import os
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt

# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from matplotlib.colors import ListedColormap
from tqdm import tqdm

from dismix_loss import ELBOLoss, BarlowTwinsLoss
from diffusers import AudioLDM2Pipeline
from diffusers.models import AutoencoderKL

# from hifi_gan.inference import mel_to_wav
# from hifi_gan.env import AttrDict
# import argparse, json

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
        )
        self.temporal_pool = nn.AdaptiveAvgPool2d((1, 16))

    def forward(self, x):
        h = self.encoder(x)
        h = self.temporal_pool(h)
        return h
    
# Use the pre-trained encoder and freeze
class MixtureEncoder(nn.Module):
    def __init__(
        self,
        repo_id,
    ):
        super().__init__()
        self.encoder = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float32).vae.encoder
        
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
            h = self.encoder(x).permute(0, 1, 3, 2)

        h = self.conv(h)            
        return h
    
    
    
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
        pitch_latent = self.f_phai_nu(y_hat_sb.unsqueeze(1)) # BS, 1, 129, 400 --> BS, 8, 100, 16
        return y_hat, pitch_latent


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



    def forward(self, x):
        BS, C, T, F = x.shape
        x = x.permute(0, 1, 3, 2).reshape(BS, C*F, T)
        x = self.conv_layers(x)
        x = self.temporal_pool(x)
        
        split_x = torch.split(x, 128, dim=1)
        reshaped_x = [t.view(t.size(0), 8, 1, 16) for t in split_x]
        mean, logvar = reshaped_x[0], reshaped_x[1]
        
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=1e-6)
        q = torch.distributions.Normal(mean, std)
        timbre_latent = q.rsample()
        return timbre_latent, mean, std


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
    def __init__(self, patch_size, dim, num_patches, batch_size, N_s=4, max_len=5000):
        super(Partition, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = num_patches
        self.batch_size = batch_size
        self.N_s = N_s
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
        x = x.view(self.batch_size, self.N_s * self.num_patches, -1) # Shape: [B*L, D_z']

        return x


# DiTPatchPartitioner Class
class DiTPatchPartitioner(nn.Module):
    def __init__(
        self, 
        batch_size,
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
            batch_size=batch_size,
            max_len=max_len
        )
        self.s_partition = Partition(
            patch_size=s_patch_size, 
            dim=s_dim, 
            num_patches=s_num_patches,
            batch_size=batch_size,
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
        repo_id,
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
        self.N_s = N_s # 4
        self.batch_size = batch_size
        self.condition_dim = condition_dim
        
        self.pt_E_VAE = MixtureEncoder(repo_id=repo_id) # pipe.vae.encoder
        self.pt_D_VAE = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float32).vae.decoder
        self.freeze_decoder()
        
        self.partitioner = DiTPatchPartitioner(batch_size=batch_size)
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
        

    def freeze_decoder(self):
        """Freeze the parameters of the decoder to prevent training."""
        for param in self.pt_D_VAE.parameters():
            param.requires_grad = False
        
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
        B = self.batch_size                 # Batch size
        L = 25                              # Number of patches
        T_z = 100                           # Original temporal dimension
        D_z = 16                            # Feature dimension
        C = 8                               # Number of channels

        # Flattened patch dimension must match
        patch_size = T_z // L  # Frames per patch
        D_z_prime = patch_size * D_z * C  # Flattened patch dimension
        
        # Ensure patched_tensor shape matches the expected input
        assert x.shape[1] == 4 * L and x.shape[2] == D_z_prime, \
            "Patched tensor shape mismatch with expected dimensions."

        x = x.view(B, self.N_s, L, D_z_prime)

        # Step 3: Flatten patch dimensions back into [batch_size, T_z, D_z, C]
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, L, 4, patch_size * D_z * C]
        x = x.reshape(B*self.N_s, T_z, D_z, C)  # [batch_size, T_z, D_z, C]

        return x.permute(0, 3, 1, 2)


        
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
        with torch.no_grad():
            z_s = self.pt_E_VAE(x_s) # [batch*N_s, C=8, H=16, W=100]
            z_s = z_s.permute(0, 1, 3, 2)
        
        # Partition
        z_m_t, s_c = self.partitioner(z_s, s_i)  # z_m0: [batch*N_s*L (100), 512], s_c_patched: [batch*N_s*L, 1024]

        # Embed the diffusion step t and combine with sc
        t_embed = self.time_embed(t).view(self.batch_size, -1, self.condition_dim)  # Shape: [batch_size, condition_dim]
        condition = s_c + t_embed
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            z_m_t = block(z_m_t, condition)

        # Unpatchify
        z_m_t = self.unpatchify(z_m_t)
        
        # Decoder
        with torch.no_grad():
            z_m_t = self.pt_D_VAE(z_m_t).permute(0, 1, 3, 2)
        return z_m_t
    
    
    
    def evaluate(self, noise, s_i, num_steps=1000):
        """
        Evaluation/inference method.

        Args:
            noise (Tensor): Starting random noise, shape [batch_size, C, Tz, Dz].
            s_i (Tensor): Source conditioning tensor.
            num_steps (int): Number of diffusion steps.

        Returns:
            Tensor: Generated sample.
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            z_m_t = self.pt_E_VAE(noise) #.to(torch.float32) # [batch*N_s, C=8, H=16, W=100]
            z_m_t = z_m_t.permute(0, 1, 3, 2)

            for t in reversed(range(1, num_steps + 1)):
                # Compute t_embed for the current step
                t_tensor = torch.tensor([t / num_steps], device=z_m_t.device).float().view(1, -1)
                t_embed = self.time_embed(t_tensor).expand(self.batch_size, z_m_t.shape[2], -1)
                
                # Partition
                z_m_t_patched, s_c = self.partitioner(z_m_t, s_i)
                condition = s_c + t_embed

                # Apply transformer blocks
                for block in self.transformer_blocks:
                    z_m_t_patched = block(z_m_t_patched, condition)

                # Unpatchify and decode
                z_m_t = self.unpatchify(z_m_t_patched)

            # Decoder
            z_m_t = self.pt_D_VAE(z_m_t)

        return z_m_t# ([8, 1, 400, 64])


    
class DisMix_LDM_Model(pl.LightningModule):
    def __init__(
        self,
        repo_id="cvssp/audioldm2-music",
        learning_rate=1e-4,
        D_z=16,
        D_s=32,
        L=25,
        diffusion_steps=1000, 
        beta_start=1e-4, 
        beta_end=0.02, 
        N_s=4,
        batch_size=1,
        pitch_labels=129,
    ):
        super(DisMix_LDM_Model, self).__init__()
        # self.save_hyperparameters()
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
        
        self.M_Encoder = MixtureEncoder(repo_id=repo_id) # Pre-trained model
        
        self.combine_conv = nn.Conv2d(
            in_channels=32,  # 16 from em and 16 from eq after concatenation
            out_channels=16,  # Transform back to original feature dimension
            kernel_size=1,
            stride=1,
            padding=0,
        )
        
        self.pitch_encoder = PitchEncoder()
        self.timbre_encoder = TimbreEncoder()
        self.dit = DiT(
            repo_id=repo_id,
            batch_size=batch_size,
            N_s=N_s,
        )
        
        # Params
        self.D_z = D_z
        self.D_s = D_s
        self.L = L
        self.N_s = N_s # 4
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pitch_labels = pitch_labels

        # Define the noise schedule
        self.diffusion_steps = diffusion_steps
        self.beta = torch.linspace(beta_start, beta_end, diffusion_steps)#.to(device)  # (T,)
        self.alpha = 1.0 - self.beta  # (T,)
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)  # (T,)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alpha_cumprod[:-1]], dim=0)  # (T,)

        # Loss functions
        self.elbo_loss_fn = ELBOLoss() # For ELBO
        self.ce_loss_fn = nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss()  # For pitch supervision
        self.bt_loss_fn = BarlowTwinsLoss() # Barlow Twins
        
        
    def forward(self, x_m, x_s, evaluate=False): # x_q exactly is x_s in inference
        # Reshape
        _, _, F, T = x_m.shape
        x_s = x_s.view(-1, 1, F, T) # Shape: [BS*N_s, 1, 64, 400]
        
        e_q = self.Q_Encoder(x_s)
        e_m = self.M_Encoder(x_m)
        e_m = e_m.repeat(self.N_s, 1, 1, 1)  # Shape: [BS*4, 8, 100, 16]
        
        # BroadCast & Concat
        eq_broadcast = e_q.expand(-1, -1, e_m.size(2), -1)
        combined = torch.cat([e_m, eq_broadcast], dim=3).permute(0, 3, 2, 1)
        combined = self.combine_conv(combined)
        combined = combined.permute(0, 3, 2, 1) # BS, 8, 100, 16
        
        # Pitch Encoder
        y_hat, pitch_latent = self.pitch_encoder(combined)
        pitch_latent = pitch_latent.permute(0, 1, 3, 2)
        
        # Timbre Encoder
        timbre_latent, timbre_mean, timbre_std = self.timbre_encoder(combined)
        timbre_latent_expand = timbre_latent.expand(-1, -1, pitch_latent.shape[2], -1)
        
        # Concat: f_phi_s
        s_i = torch.cat((pitch_latent, timbre_latent_expand), dim=3) # s_c: batch*N_s, 8, 100, 32
        x_s = x_s.permute(0, 1, 3, 2) # shape: torch.Size([batch*N_s, 1, 400, 64])

        # Latent Diffusion Model
        t = torch.randint(0, self.diffusion_steps, (self.batch_size * self.N_s * self.L, 1)).float().to(s_i.device)  # Diffusion step
        if evaluate:
            x_s_recon = self.dit.evaluate(x_s, s_i)
        else:
            x_s_recon = self.dit(x_s, s_i, t)

        x_s_recon = x_s_recon.permute(0, 1, 3, 2).view(self.batch_size, self.N_s, F, T)

        return e_q, y_hat, timbre_latent, timbre_mean, timbre_std, x_s_recon #res_audio
    
    
    def training_step(self, batch, batch_idx):
        x_m, x_s_i, y_gt = batch
        e_q, y_hat, timbre_latent, timbre_mean, timbre_std, x_s_recon = self(x_m, x_s_i, evaluate=False)
        
        # Pitch annotation ohe for BCE Loss
        y_gt = y_gt.view(self.batch_size*self.N_s, -1)
        bs, tf = y_gt.shape
        y_gt_ohe = F.one_hot(y_gt, num_classes=self.pitch_labels).float().permute(0, 2, 1)  # Shape: [Batch_size, 129, Time_frames]
        
        # Compute losses
        elbo_loss = self.elbo_loss_fn(
            None, None, # x_m, x_s_recon.sum(dim=1),
            x_s_i, x_s_recon,
            timbre_latent, timbre_mean, timbre_std,
        )
        bce_loss = F.binary_cross_entropy_with_logits(y_hat, y_gt_ohe) # ce_loss = self.ce_loss_fn(y_hat, y_gt)
        bt_loss = self.bt_loss_fn(e_q, timbre_latent)
        
        # Total loss
        total_loss = elbo_loss['loss'] + bce_loss + bt_loss #elbo_loss['loss'] + ce_loss + bt_loss
        
        # Log losses with batch size
        self.log('train_loss', total_loss, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        self.log('train_elbo_loss', elbo_loss['loss'], on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        self.log('train_elbo_recon_x_loss', elbo_loss['recon_x'], on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        self.log('train_elbo_kld_loss', elbo_loss['kld'], on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        self.log('train_bce_loss', bce_loss, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        self.log('train_bt_loss', bt_loss, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        
        return total_loss
    
    
        
    def evaluate(self, batch, stage='val'):
        x_m, x_s_i, y_gt = batch
        e_q, y_hat, timbre_latent, timbre_mean, timbre_std, x_s_recon = self(x_m, x_s_i, evaluate=True)
        
        # Pitch annotation ohe for BCE Loss
        y_gt = y_gt.view(self.batch_size*self.N_s, -1)
        bs, tf = y_gt.shape
        y_gt_ohe = F.one_hot(y_gt, num_classes=self.pitch_labels).float().permute(0, 2, 1)  # Shape: [Batch_size, 129, Time_frames]
        
        # Compute losses
        elbo_loss = self.elbo_loss_fn(
            None, None, # x_m, x_s_recon.sum(dim=1),
            x_s_i, x_s_recon,
            timbre_latent, timbre_mean, timbre_std,
        )
        bce_loss = F.binary_cross_entropy_with_logits(y_hat, y_gt_ohe) # ce_loss = self.ce_loss_fn(y_hat, y_gt)
        bt_loss = self.bt_loss_fn(e_q, timbre_latent)
        
        # Total loss
        total_loss = elbo_loss['loss'] + bce_loss + bt_loss #elbo_loss['loss'] + ce_loss + bt_loss
        
        # Get accuracy
        predicted_pitches = torch.argmax(y_hat, dim=1)  # Shape: [Batch_size, Time_frames]
        correct_predictions = (predicted_pitches == y_gt).float().sum()
        accuracy = correct_predictions / (bs * tf)
        
        # Log losses and metrics
        self.log(f'{stage}_loss', total_loss, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        self.log(f'{stage}_elbo_loss', elbo_loss['loss'], on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        self.log(f'{stage}_elbo_recon_x_loss', elbo_loss['recon_x'], on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        self.log(f'{stage}_elbo_kld_loss', elbo_loss['kld'], on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        self.log(f'{stage}_bce_loss', bce_loss, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        self.log(f'{stage}_bt_loss', bt_loss, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        self.log(f'{stage}_acc', accuracy.item(), on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        return total_loss
        
    
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

        
    # def on_validation_epoch_end(self):
    #     return self.plotting(stage='val')
    
    # def on_test_epoch_end(self):
    #     return self.plotting(stage='test')
    
    

    
if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    BS, N_s = 2, 4
    x_q = torch.randn(BS, N_s, 64, 400).to(device)
    x_m = torch.randn(BS, 1, 64, 400).to(device)
    

    
    # choice = ["ema", "mse"]
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{choice[1]}").to(device)
    pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=torch.float32).to(device)
    vae = pipe.vae 
    
    model = DisMix_LDM_Model(
        batch_size=BS,
        N_s=N_s,
        vae=vae,
    ).to(device)
    
    res = model(x_m, x_q)