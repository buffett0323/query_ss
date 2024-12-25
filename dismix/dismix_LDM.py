import os
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

from dismix_loss import ELBOLoss, BarlowTwinsLoss
from diffusers import AudioLDM2Pipeline
from diffusers.models import AutoencoderKL

from hifi_gan.inference import mel_to_wav
from hifi_gan.env import AttrDict
import argparse, json

from audioldm_train.modules.diffusionmodules.model import Encoder, Decoder
from audioldm_train.modules.diffusionmodules.distributions import DiagonalGaussianDistribution
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Could not initialize NNPACK")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class QueryEncoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=False,
        use_linear_attn=False,
        attn_type="vanilla",
        downsample_time_stride4_levels=[],
    ):
        super().__init__()

        self.encoder = Encoder(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
            double_z=double_z,
            use_linear_attn=use_linear_attn,
            attn_type=attn_type,
            downsample_time_stride4_levels=downsample_time_stride4_levels,
        ) #.half() # convert to torch float16
        self.temporal_pool = nn.AdaptiveAvgPool2d((1, 16))

    def forward(self, x):
        h = self.encoder(x)
        h = self.temporal_pool(h)
        return h
    
# Use the pre-trained encoder and freeze
class MixtureEncoder(nn.Module):
    def __init__(
        self,
        vae,
    ):
        super().__init__()
        self.encoder = vae.encoder
        self.freeze_encoder()
        self.conv = nn.Conv2d(
            in_channels=16,  # Input channels
            out_channels=8,  # Output channels
            kernel_size=(1, 1),  # Kernel size to maintain spatial dimensions
            stride=(1, 1),  # Stride to maintain spatial dimensions
            padding=(0, 0)  # No padding needed for (1, 1) kernel
        )

    def freeze_encoder(self):
        """Freeze the parameters of the encoder to prevent training."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = x.to(torch.float16)
            h = self.encoder(x).permute(0, 1, 3, 2)
            h = h.to(torch.float32)
        return self.conv(h)
    
    
    
class StochasticBinarizationLayer(nn.Module):
    def __init__(self):
        super(StochasticBinarizationLayer, self).__init__()
    
    def forward(self, logits):
        """
        Forward pass of the stochastic binarization layer.
        """
        prob = torch.sigmoid(logits)
        if self.training:
            h = torch.rand_like(prob)  # Use random threshold during training
        else:
            h = torch.full_like(prob, 0.5)  # Fixed threshold of 0.5 during inference
            
        return (prob > h).float()  # Binarize based on the threshold h
        
        
class PitchEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.E_phai_nu = Decoder(
            ch=64,  # Base number of channels; can be adjusted based on model capacity
            out_ch=129,  # Desired number of output channels (pitch values)
            ch_mult=(1, 2, 4),  # Channel multipliers for each resolution level
            num_res_blocks=2,  # Number of residual blocks per resolution level
            attn_resolutions=[],  # Resolutions at which to apply attention (can be adjusted as needed)
            dropout=0.0,  # Dropout rate
            resamp_with_conv=True,  # Whether to use convolution in upsampling
            in_channels=8,  # Number of input channels
            resolution=100,  # Spatial resolution corresponding to the Height dimension
            z_channels=8,  # Latent space channels (should match in_channels)
            give_pre_end=False,  # Whether to return the tensor before the final layer
            tanh_out=False,  # Whether to apply Tanh activation at the output
            use_linear_attn=False,  # Whether to use linear attention
            downsample_time_stride4_levels=[],  # Levels at which to perform stride-4 downsampling
            attn_type="vanilla",  # Type of attention mechanism
        )

        self.sb_layer = StochasticBinarizationLayer()
        self.f_phai_nu = Encoder(
            ch=64,
            out_ch=8,  # Not directly used in the current implementation but kept for consistency
            ch_mult=(1, 2, 4),
            num_res_blocks=2,  # Number of ResNet blocks per resolution
            attn_resolutions=[],  # Add resolutions where you want attention if needed
            dropout=0.0,
            resamp_with_conv=True,
            in_channels=1,  # Adjust based on your input
            resolution=129,  # Input height
            z_channels=8,  # Desired output channels
            double_z=False,
            use_linear_attn=False,
            attn_type="vanilla",
            downsample_time_stride4_levels=[1],  # Apply stride=4 on the second downsampling layer
        )


        
    def forward(self, x):
        y_hat = self.E_phai_nu(x) # BS, 8, 100, 16 --> BS, 129, 400
        y_hat_sb = self.sb_layer(y_hat)
        tau = self.f_phai_nu(y_hat_sb.unsqueeze(1)) # BS, 129, 400 --> BS, 8, 100, 16
        return y_hat, tau


class TimbreEncoder(nn.Module):
    def __init__(self):
        super(TimbreEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=0),
            nn.GroupNorm(num_groups=1, num_channels=128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=0),
            nn.GroupNorm(num_groups=1, num_channels=128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=0),
            nn.GroupNorm(num_groups=1, num_channels=128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0),
        )

        # Temporal pooling layer to reduce time dimension from 100 to 1
        self.temporal_pool = nn.AdaptiveAvgPool1d(output_size=1) # nn.AdaptiveAvgPool2d((1, 16))  # Output size: (time=1, feature=16)


    def reparameterize(self, mean, logvar):
        """
            Reparameterization trick to sample from N(mean, var)
            Sampling by μφτ (·) + ε σφτ (·)
        """
        std = torch.exp(logvar) # std = torch.exp(0.5 * torch.clamp(logvar, min=-10, max=10))
        eps = torch.randn_like(std) # Random noise ~ N(0, 1)
        return mean + (eps * std)

    def forward(self, x):
        BS, C, T, F = x.shape
        x = x.permute(0, 1, 3, 2).reshape(BS, C*F, T)
        x = self.conv_layers(x)
        x = self.temporal_pool(x)
        
        split_x = torch.split(x, 128, dim=1)
        reshaped_x = [t.view(t.size(0), 8, 1, 16) for t in split_x]
        mean, logvar = reshaped_x[0], reshaped_x[1]
        timbre_latent = self.reparameterize(mean, logvar)
        return mean, logvar, timbre_latent


# Sinusoidal Positional Encoding
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.pe.size(1)}")
        pos_encoding = self.pe[:, :seq_len, :].to(x.device)
        return x + pos_encoding

# Partition Class
class Partition(nn.Module):
    def __init__(self, patch_size, dim, num_patches, max_len=5000):
        super(Partition, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = num_patches
        self.positional_encoding = SinusoidalPositionalEncoding(dim, max_len)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, C, D, T], where:
               - C: Channels (8)
               - D: Feature dimensions (16)
               - T: Time frames (100)
        """
        B, C, T, D = x.shape  # B=batch_size, C=8, T=100, D=16: torch.Size([4, 8, 100, 16]) torch.Size([4, 8, 100, 32])
        assert T % self.num_patches == 0, "Time frames must be divisible by the number of patches."

        # Step 1: Permute and reshape the input
        x = x.permute(0, 2, 3, 1).contiguous()  # Shape: [B, T, D, C]
        x = x.view(B, self.num_patches, self.patch_size, D, C)  # Split into patches: [B, L=25, 4, D, C]

        # Step 2: Flatten the patch dimensions (4, D, C) -> (4 * 16 * 8)
        x = x.reshape(B, self.num_patches, -1)  # Shape: [B, L=25, 4*16*8]

        # Step 3: Add positional encoding
        x = self.positional_encoding(x)  # Shape: [B, L=25, D_z']
        x = x.view(B * self.num_patches, -1) # Shape: [B*L, D_z']

        return x


# DiTPatchPartitioner Class
class DiTPatchPartitioner(nn.Module):
    def __init__(
        self, 
        z_patch_size=4, 
        z_num_patches=25, 
        z_dim=512,
        s_patch_size=4, 
        s_num_patches=25, 
        s_dim=1024,
        max_len=5000
    ):
        super(DiTPatchPartitioner, self).__init__()
        self.z_partition = Partition(
            patch_size=z_patch_size, 
            dim=z_dim, 
            num_patches=z_num_patches,
            max_len=max_len
        )
        self.s_partition = Partition(
            patch_size=s_patch_size, 
            dim=s_dim, 
            num_patches=s_num_patches,
            max_len=max_len
        )

    def forward(self, z_s, s):
        z_m = self.z_partition(z_s)  # (N_s * L, 512)
        s_c = self.s_partition(s)  # (N_s * L, 1024)
        return z_m, s_c


class AdaLayerNorm(nn.Module):
    def __init__(
        self, 
        normalized_shape, 
        conditioning_dim
    ):
        """
        Args:
            normalized_shape (int): Input shape from an expected input of size
                                     [*, normalized_shape].
            conditioning_dim (int): Dimension of the conditioning vector.
        """
        super(AdaLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)
        
        # Projection layers for scaling (gamma) and shifting (beta)
        self.gamma_proj = nn.Linear(conditioning_dim, normalized_shape)
        self.beta_proj = nn.Linear(conditioning_dim, normalized_shape)

    def forward(self, x, condition):
        """
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, normalized_shape].
            condition (Tensor): Conditioning tensor of shape [batch_size, conditioning_dim].
        Returns:
            Tensor: Normalized and conditioned tensor.
        """
        normalized = self.layer_norm(x)
        gamma = self.gamma_proj(condition)
        beta = self.beta_proj(condition)
        return gamma * normalized + beta



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, conditioning_dim, ff_dim, dropout=0.1):
        """
        Args:
            dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            conditioning_dim (int): Dimension of the conditioning vector.
            ff_dim (int): Dimension of the feedforward network.
            dropout (float): Dropout rate.
        """
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.ada_ln1 = AdaLayerNorm(dim, conditioning_dim) # adaLN
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        self.ada_ln2 = AdaLayerNorm(dim, conditioning_dim) # adaLN

    def forward(self, x, condition):
        """
        Args:
            x (Tensor): Input tensor of shape [seq_len, batch_size, dim].
            condition (Tensor): Conditioning tensor of shape [batch_size, conditioning_dim].
        Returns:
            Tensor: Output tensor of shape [seq_len, batch_size, dim].
        """
        # Self-Attention
        attn_output, _ = self.self_attn(x, x, x)  # [seq_len, batch, dim]
        x = x + attn_output  # Residual connection

        # Adaptive Layer Norm after Residual
        x = self.ada_ln1(x, condition)#.transpose(0, 1)  # [seq_len, batch, dim]
        
        # Feedforward Network
        ff_output = self.ff(x)  # [seq_len, batch, dim]
        x = x + ff_output  # Residual connection

        # Adaptive Layer Norm after Residual
        x = self.ada_ln2(x, condition)#.transpose(0, 1)  # [seq_len, batch, dim]

        return x



class DiT(nn.Module):
    def __init__(
        self, 
        vae,
        batch_size,
        N_s,
        dim=512, 
        num_blocks=3,
        num_heads=4, 
        condition_dim=1024, 
        ff_dim=2048, 
        dropout=0.1,
    ):
        """
        Ref: https://github.com/facebookresearch/DiT/blob/main/models.py
        Args:
            dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer blocks.
            conditioning_dim (int): Dimension of the conditioning vector.
            ff_dim (int): Dimension of the feedforward network.
            max_patch_seq_len (int): Maximum sequence length for positional encoding.
            dropout (float): Dropout rate.
        """
        super(DiT, self).__init__()
        self.pt_E_VAE = MixtureEncoder(vae=vae) # pipe.vae.encoder
        self.pt_D_VAE = vae.decoder
        self.partitioner = DiTPatchPartitioner()
        self.N_s = N_s # 4
        self.batch_size = batch_size
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, condition_dim, ff_dim, dropout) 
            for _ in range(num_blocks)
        ])
        self.layer_norm = nn.LayerNorm(dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim)
        )
        
        
    def unpatchify(self, x):
        """
        Unpatchifies a tensor from patch representation to the original input format.

        Args:
            x (Tensor): Input tensor of shape [N*L, Dz'], where
                        - N: Batch size
                        - L: Number of patches
                        - Dz': Flattened patch dimension (Tz/L x Dz x C).

        Returns:
            Tensor: Output tensor of shape [N, C, Tz, Dz], where
                    - C: Number of channels
                    - Tz: Original temporal dimension
                    - Dz: Original feature dimension.
        """
        B = self.batch_size * self.N_s      # Batch size
        L = 25                              # Number of patches
        Tz = 100                            # Original temporal dimension
        Dz = 16                             # Feature dimension
        C = 8                               # Number of channels

        # Flattened patch dimension must match
        D_prime = (Tz // L) * Dz * C
        assert x.shape[1] == D_prime, f"Flattened dimension mismatch: expected {D_prime}, got {x.shape[1]}"

        # Reshape from [N*L, D_prime] to [N, L, Tz/L, Dz, C]
        x = x.view(B, L, Tz // L, Dz, C)  # Shape: [B, L, Tz/L, Dz, C]

        # Rearrange to [B, Tz, Dz, C]
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # Shape: [B, Tz, Dz, C]
        x = x.view(B, Tz, Dz, C)

        # Final rearrangement to [B, C, Tz, Dz]
        x = x.permute(0, 3, 1, 2).contiguous()  # Shape: [B, C, Tz, Dz]

        return x


        
    def forward(self, x_s, s_i, t):
        """
        x_s.shape: torch.Size([4, 1, 400, 64]) 
        s_i.shape: torch.Size([4, 8, 100, 32])
        Args:
            x (Tensor): Input tensor of shape [seq_len, batch_size, dim].
            condition (Tensor): Conditioning tensor of shape [batch_size, conditioning_dim].
        Returns:
            Tensor: Output tensor of shape [seq_len, batch_size, dim].
        """
        x_s = x_s.to(torch.float16)
        z_s = self.pt_E_VAE(x_s).to(torch.float32) # [batch*N_s, C=8, H=16, W=100]
        z_s = z_s.permute(0, 1, 3, 2)
        
        # Partition
        z_m_t, s_c = self.partitioner(z_s, s_i)  # z_m0: [batch*N_s*L (100), 512], s_c_patched: [batch*N_s*L, 1024]

        # Embed the diffusion step t and combine with sc
        t_embed = self.time_embed(t)  # Shape: [batch_size, condition_dim]
        condition = s_c + t_embed
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            z_m_t = block(z_m_t, condition)

        z_m_t = self.unpatchify(z_m_t)  
        
        # Decoder
        z_m_t = z_m_t.to(torch.float16)
        z_m_t = self.pt_D_VAE(z_m_t).to(torch.float32)
        return z_m_t.permute(0, 1, 3, 2)  


class DisMix_LDM(nn.Module):
    def __init__(
        self,
        vae,
        D_z=16,
        D_s=32,
        L=25,
        diffusion_steps=1000, 
        beta_start=1e-4, 
        beta_end=0.02, 
        N_s=4,
        batch_size=1,
        device='cuda',
    ):
        """
        Args:
            encoder (nn.Module): Encoder module.
            partitioner (nn.Module): DiT Patch Partitioner module.
            dit (nn.Module): Diffusion Transformer module.
            decoder (nn.Module): Decoder module.
            diffusion_steps (int): Number of diffusion steps.
            beta_start (float): Starting value of beta for noise schedule.
            beta_end (float): Ending value of beta for noise schedule.
            device (str): Device to run the model on.
        """
        super().__init__()
        self.Q_Encoder = QueryEncoder(
            ch=64,  # Initial number of channels, can be adjusted as needed
            out_ch=8,  # Desired number of output channels
            ch_mult=(1, 2, 4),  # Two resolution levels for two downsampling steps
            num_res_blocks=2,  # Number of residual blocks per resolution level
            attn_resolutions=[],  # Adjust based on whether you want attention
            dropout=0.0,
            resamp_with_conv=True,
            in_channels=1,  # Set according to your input data (e.g., 1 for mono audio)
            resolution=64,  # Frequency dimension of input
            z_channels=8,  # Latent space channels
            double_z=False,  # To keep z_channels as 8
            use_linear_attn=False,
            attn_type="vanilla",
            downsample_time_stride4_levels=[],  # No special downsampling for time
        )
        
        self.M_Encoder = MixtureEncoder(vae=vae) # Pre-trained model
        
        self.combine_conv = nn.Conv2d(
            in_channels=32,  # 16 from em and 16 from eq after concatenation
            out_channels=16,  # Transform back to original feature dimension
            kernel_size=1,
            stride=1,
            padding=0,
        )
        
        self.pitch_encoder = PitchEncoder()
        self.timbre_encoder = TimbreEncoder()
        self.vae = vae
        self.D_z = D_z
        self.D_s = D_s
        self.L = L
        self.device = device
        self.partitioner = DiTPatchPartitioner()
        self.N_s = N_s # 4
        self.batch_size = batch_size
        self.dit = DiT(
            vae=vae,
            batch_size=batch_size,
            N_s=N_s,
        )

        # Define the noise schedule
        self.diffusion_steps = diffusion_steps
        self.beta = torch.linspace(beta_start, beta_end, diffusion_steps).to(device)  # (T,)
        self.alpha = 1.0 - self.beta  # (T,)
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)  # (T,)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alpha_cumprod[:-1]], dim=0)  # (T,)

        
    def hifigan(self, mel):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_wavs_dir', default='test_files')
        parser.add_argument('--output_dir', default='generated_files')
        parser.add_argument('--checkpoint_file', default='LJ_V3/generator_v3')
        a = parser.parse_args()

        config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
        with open(config_file) as f:
            data = f.read()
            
        json_config = json.loads(data)
        h = AttrDict(json_config)
        return mel_to_wav(mel, a, h, mel.device)
    
    
    def forward(self, x_m, x_s): # x_q exactly is x_s in inference
        e_m = self.M_Encoder(x_m)
        e_q = self.Q_Encoder(x_s)
        
        # Concat
        eq_broadcast = e_q.expand(-1, -1, e_m.size(2), -1)
        combined = torch.cat([e_m, eq_broadcast], dim=3).permute(0, 3, 2, 1)
        combined = self.combine_conv(combined)
        combined = combined.permute(0, 3, 2, 1) # BS, 8, 100, 16
        
        # Pitch Encoder
        y_hat, pitch_latent = self.pitch_encoder(combined)
        pitch_latent = pitch_latent.permute(0, 1, 3, 2)
        
        # Timbre Encoder
        timbre_mean, timbre_logvar, timbre_latent = self.timbre_encoder(combined)
        timbre_latent = timbre_latent.expand(-1, -1, pitch_latent.shape[2], -1)
        
        # Concat: f_phi_s
        s_i = torch.cat((pitch_latent, timbre_latent), dim=3) # s_c: batch*N_s, 8, 100, 32
        x_s = x_s.permute(0, 1, 3, 2)
        
        """ Latent Diffusion Model """
        t = torch.randint(0, 1000, (self.batch_size * self.N_s * self.L, 1)).float().to(s_i.device)  # Diffusion step
        dit_mel = self.dit(x_s, s_i, t)
        
        # Transform back to audio
        dit_mel = dit_mel.squeeze(1)
        dit_mel_80 = F.interpolate(dit_mel.unsqueeze(1), size=(80, dit_mel.shape[2]), mode='bilinear', align_corners=False).squeeze(1)
        res_audio = self.hifigan(dit_mel_80)
        print("res_audio", res_audio.shape)

        return y_hat, timbre_mean, timbre_logvar, res_audio
    
    
class DisMix_LDM_Model(pl.LightningModule):
    def __init__(
        self,
        vae,
        D_z=16,
        D_s=32,
        L=25,
        diffusion_steps=1000, 
        beta_start=1e-4, 
        beta_end=0.02, 
        N_s=4,
        batch_size=1,
        device='cuda',
    ):
        super(DisMix_LDM_Model, self).__init__()
        self.model = DisMix_LDM(
            vae=vae,
            D_z=D_z,
            D_s=D_s,
            L=L,
            diffusion_steps=diffusion_steps, 
            beta_start=beta_start, 
            beta_end=beta_end, 
            N_s=N_s,
            batch_size=batch_size,
            device=device,
        )
        self.save_hyperparameters()
        
    def forward(self, x_m, x_s):
        return self.model(x_m, x_s)
    
    def training_step(self, batch, batch_idx):
        self()
        
    def evaluate(self, batch, stage='val'):
        pass
    
    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, stage='val')

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, stage='test')


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    
        # Gradient Clipping
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-0.5, 0.5)
        
        # Learning Rate Scheduler
        warmup_steps = 308000  # 308k steps
        total_steps = 4092000  # 4,092k steps
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / warmup_steps  # Linear warmup
            return 0.5 * (1 + math.cos(math.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        return [optimizer], [scheduler]

        
    def on_validation_epoch_end(self):
        return self.plotting(stage='val')
    
    def on_test_epoch_end(self):
        return self.plotting(stage='test')
    
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BS, N_s = 1, 4
    x_q = torch.randn(BS*N_s, 1, 64, 400).to(device)#.to(torch.float16)
    x_m = torch.randn(BS*N_s, 1, 64, 400).to(device)#.to(torch.float16)
    
    # choice = ["ema", "mse"]
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{choice[0]}").to(device)
    pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=torch.float16).to(device)
    vae = pipe.vae 
    
    model = DisMix_LDM(
        batch_size=BS,
        N_s=N_s,
        vae=vae,
    ).to(device)
    
    res = model(x_m, x_q)