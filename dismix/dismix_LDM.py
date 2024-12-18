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
# from dismix_model import Conv1DEncoder, MixtureQueryEncoder, \
    # StochasticBinarizationLayer, TimbreEncoder#, PitchEncoder

from diffusers import AudioLDM2Pipeline
from audioldm_train.modules.diffusionmodules.model import Encoder, Decoder
from audioldm_train.modules.diffusionmodules.distributions import DiagonalGaussianDistribution
import warnings
warnings.filterwarnings('ignore')

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
        repo_id="cvssp/audioldm2",
    ):
        super().__init__()
        pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        self.encoder = pipe.vae.encoder

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
        return tau


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
        std = torch.exp(0.5 * logvar) # std = torch.exp(0.5 * torch.clamp(logvar, min=-10, max=10))
        eps = torch.randn_like(std)
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
        return timbre_latent
    


# class D_theta_m(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout=0.1):
#         super(D_theta_m, self).__init__()
#         # Transformer configuration
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=input_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim,
#             dropout=dropout,
#             activation='relu'
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.layer_norm = nn.LayerNorm(input_dim)
#         self.output_layer = nn.Linear(input_dim, input_dim)  # Predict clean latent

#     def forward(self, zm_t, sc):
#         """
#         Args:
#             zm_t: Noised latent input (batch_size, sequence_length, input_dim)
#             sc: Condition embeddings (batch_size, sequence_length, input_dim)

#         Returns:
#             zm_t_minus_1: Denoised latent (batch_size, sequence_length, input_dim)
#         """
#         # Combine inputs and normalize
#         combined_input = zm_t + sc
#         combined_input = self.layer_norm(combined_input)

#         # Transformer processing
#         processed = self.transformer(combined_input)

#         # Predict next denoised latent
#         zm_t_minus_1 = self.output_layer(processed)
#         return zm_t_minus_1


# class D_vae(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(D_vae, self).__init__()
#         # A simple linear decoder for reconstruction
#         self.decoder = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, output_dim)
#         )

#     def forward(self, z_s):
#         """
#         Args:
#             z_s: Latent representation for a single source (batch_size, latent_dim)

#         Returns:
#             x_s: Reconstructed source (batch_size, output_dim)
#         """
#         return self.decoder(z_s)


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
        B, C, D, T = x.shape  # B=batch_size, C=8, D=16, T=100
        assert T % self.num_patches == 0, "Time frames must be divisible by the number of patches."

        # Step 1: Permute and reshape the input
        x = x.permute(0, 3, 1, 2).contiguous()  # Shape: [B, T, C, D]
        x = x.view(B, self.num_patches, self.patch_size, C, D)  # Split into patches: [B, L=25, 4, C, D]

        # Step 2: Flatten the patch dimensions (4, C, D) -> (4 * 16 * 8)
        x = x.reshape(B, self.num_patches, -1)  # Shape: [B, L=25, 4*16*8]

        # Step 3: Add positional encoding
        x = self.positional_encoding(x)  # Shape: [B, L=25, D_z']
        x = x.view(B * self.num_patches, -1) # Shape: [B*L, D_z']

        return x


# DiTPatchPartitioner Class
class DiTPatchPartitioner(nn.Module):
    def __init__(self, 
                 z_patch_size=4, 
                 z_num_patches=25, 
                 z_dim=512,
                 s_patch_size=4, 
                 s_num_patches=25, 
                 s_dim=1024,
                 max_len=5000):
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
    def __init__(self, normalized_shape, conditioning_dim):
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
        gamma = self.gamma_proj(condition).unsqueeze(1)  # [batch, 1, normalized_shape]
        beta = self.beta_proj(condition).unsqueeze(1)    # [batch, 1, normalized_shape]
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
        self.ada_ln1 = AdaLayerNorm(dim, conditioning_dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        self.ada_ln2 = AdaLayerNorm(dim, conditioning_dim)

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
        x = self.ada_ln1(x.transpose(0, 1), condition).transpose(0, 1)  # [seq_len, batch, dim]

        # Feedforward Network
        ff_output = self.ff(x)  # [seq_len, batch, dim]
        x = x + ff_output  # Residual connection

        # Adaptive Layer Norm after Residual
        x = self.ada_ln2(x.transpose(0, 1), condition).transpose(0, 1)  # [seq_len, batch, dim]

        return x



class DiT(nn.Module):
    def __init__(
        self, 
        dim=512, 
        num_heads=4, 
        num_layers=3, 
        conditioning_dim=1024 + 512,  # s_c_patched + timestep_embedding
        ff_dim=2048, 
        max_patch_seq_len=25*4,  # Assuming N_s=4 and L=25
        dropout=0.1,
    ):
        """
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
        self.positional_encoding = SinusoidalPositionalEncoding(dim, max_len=max_patch_seq_len)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, conditioning_dim, ff_dim, dropout) 
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, condition):
        """
        Args:
            x (Tensor): Input tensor of shape [seq_len, batch_size, dim].
            condition (Tensor): Conditioning tensor of shape [batch_size, conditioning_dim].
        Returns:
            Tensor: Output tensor of shape [seq_len, batch_size, dim].
        """
        x = self.positional_encoding(x.transpose(0,1)).transpose(0,1)  # Apply positional encoding
        for block in self.transformer_blocks:
            x = block(x, condition)
        x = self.layer_norm(x.transpose(0,1)).transpose(0,1)
        return x


class DisMix_LDM(nn.Module):
    def __init__(
        self,
        partitioner=DiTPatchPartitioner(), 
        dit=DiT(), 
        diffusion_steps=1000, 
        beta_start=1e-4, 
        beta_end=0.02, 
        N_s=4,
        batch_size=1,
        device='cuda',
        repo_id="cvssp/audioldm2",
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
        
        self.M_Encoder = MixtureEncoder() # Pre-trained model
        
        self.combine_conv = nn.Conv2d(
            in_channels=32,  # 16 from em and 16 from eq after concatenation
            out_channels=16,  # Transform back to original feature dimension
            kernel_size=1,
            stride=1,
            padding=0,
        )
        
        self.pitch_encoder = PitchEncoder()
        self.timbre_encoder = TimbreEncoder()
        self.device = device
        self.partitioner = partitioner
        self.dit = dit
        self.N_s = N_s # 4
        self.batch_size = batch_size

        # Define the noise schedule
        self.diffusion_steps = diffusion_steps
        self.beta = torch.linspace(beta_start, beta_end, diffusion_steps).to(device)  # (T,)
        self.alpha = 1.0 - self.beta  # (T,)
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)  # (T,)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alpha_cumprod[:-1]], dim=0)  # (T,)

        pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        self.pt_E_VAE = MixtureEncoder() # pipe.vae.encoder
        self.pt_D_VAE = pipe.vae.decoder
        
        # # Load HiFi-GAN via torch.hub
        # try:
        #     self.hifigan, self.vocoder_train_setup, self.denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan')
        #     CHECKPOINT_SPECIFIC_ARGS = [
        #         'sampling_rate', 'hop_length', 'win_length', 'p_arpabet', 'text_cleaners',
        #         'symbol_set', 'max_wav_value', 'prepend_space_to_text',
        #         'append_space_to_text']

        #     for k in CHECKPOINT_SPECIFIC_ARGS:
        #         self.vocoder_train_setup.get(k, None)
                
        #     print("HiFi-GAN loaded successfully.")
            
        # except Exception as e:
        #     print("Error loading HiFi-GAN via torch.hub. Ensure the repository and model name are correct.")
        #     print(e)
        #     self.hifigan = None

    def forward_diffusion_sample(self, z0, t):
        """
        Adds noise to the latent z0 at step t.
        Args:
            z0 (Tensor): Original latent tensor.
            t (Tensor): Diffusion step indices.
        Returns:
            Tensor: Noised latent tensor z_t.
        """
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t]).view(-1, 1, 1)
        noise = torch.randn_like(z0)
        z_t = sqrt_alpha_cumprod * z0 + sqrt_one_minus_alpha_cumprod * noise
        return z_t, noise
    
    
    def sample_timesteps(self, batch_size):
        """
        Randomly samples diffusion steps for each instance in the batch.
        Args:
            batch_size (int): Number of samples.
        Returns:
            Tensor: Random diffusion steps.
        """
        return torch.randint(0, self.diffusion_steps, (batch_size,), device=self.device).long()


    def get_timestep_embedding(self, t, dim):
        """
        Creates sinusoidal timestep embeddings.
        Args:
            t (Tensor): Timestep indices.
            dim (int): Embedding dimension.
        Returns:
            Tensor: Timestep embeddings of shape [batch, dim].
        """
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=self.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(t.size(0), 1, device=self.device)], dim=1)
        return emb  # [batch, dim]


    def sample(self, x_cond, N_s=4):
        """
        Sampling from the model using the reverse diffusion process.
        Args:
            x_cond (Tensor): Conditioning input tensor [batch_size, N_s, in_channels, height, width]
            N_s (int): Number of sources.
        Returns:
            Tensor: [batch_size, audio_length] - Mixture audio x_m
        """
        batch_size = x_cond.size(0)
        # Encode conditioning input
        _, s_c = self.encoder(x_cond)  # s_c: [batch, N_s, 32, 8, 100]
        # Reshape
        s_c = s_c.view(batch_size * N_s, 32, 8, 100)  # [batch*N_s, 32, 8, 100]
        # Partition
        _, s_c_patched = self.partitioner(None, s_c)  # [batch*N_s*L, 1024]

        # Initialize z_T as standard normal
        L = self.partitioner.z_partition.num_patches  # Number of patches
        z = torch.randn(batch_size * N_s * L, 512).to(self.device)  # [batch*N_s*L, 512]

        for t in reversed(range(self.diffusion_steps)):
            # Predict z0
            timestep = torch.full((z.size(0),), t, device=self.device, dtype=torch.long)  # [batch*N_s*L]
            timestep_embedding = self.get_timestep_embedding(timestep, self.dit.transformer_blocks[0].self_attn.self_attn.embed_dim)  # [batch*N_s*L, dim]
            condition = torch.cat([s_c_patched, timestep_embedding], dim=-1)  # [batch*N_s*L, 1024 + dim]

            z_input = z.unsqueeze(0)  # [1, batch*N_s*L, 512]
            pred_z0 = self.dit(z_input, condition)  # [1, batch*N_s*L, 512]
            pred_z0 = pred_z0.squeeze(0)  # [batch*N_s*L, 512]

            # Update z for the next step
            alpha = self.alpha[t]
            alpha_cumprod = self.alpha_cumprod[t]
            z = (z * torch.sqrt(alpha)) + (pred_z0 * (1 - alpha_cumprod))  # [batch*N_s*L, 512]

        # Decode z to get x_s
        x_s = self.decoder(z)  # [batch*N_s*L, out_channels=80, H', W']

        # Reverse partition to reconstruct full mel spectrograms per source
        # Assuming L=25 patches, W'=4 (patch_size=4)
        batch_Ns_L, C, H_prime, W_prime = x_s.shape  # [batch*N_s*L, C=80, H', W'=4]
        batch = batch_size
        N_s = N_s
        L = self.partitioner.z_partition.num_patches  # 25
        mel_bins = C
        patch_size = W_prime  # 4

        # Reshape x_s to [batch, N_s, L, mel_bins, patch_size]
        x_s = x_s.view(batch, N_s, L, mel_bins, patch_size)
        # Permute to [batch, N_s, mel_bins, L, patch_size]
        x_s = x_s.permute(0, 1, 3, 2, 4).contiguous()
        # Merge patches to reconstruct full mel spectrogram: [batch, N_s, mel_bins, L * patch_size]
        x_s = x_s.view(batch, N_s, mel_bins, L * patch_size)

        # Move to HiFi-GAN's expected input format
        # HiFi-GAN expects [batch*N_s, mel_bins, T]
        x_s = x_s.view(batch * N_s, mel_bins, L * patch_size)  # [batch*N_s, 80, 100]

        # Convert mel spectrograms to audio using HiFi-GAN
        if self.hifigan is not None:
            with torch.no_grad():
                x_s_audio = self.hifigan(x_s)  # [batch*N_s, audio_length]
        else:
            raise ValueError("HiFi-GAN model is not loaded.")

        # Reshape to [batch, N_s, audio_length]
        x_s_audio = x_s_audio.view(batch, N_s, -1)  # [batch, N_s, audio_length]

        # Sum across sources to get mixture audio x_m
        x_m = torch.sum(x_s_audio, dim=1)  # [batch, audio_length]

        return x_m  # [batch, audio_length]
    
    
    def forward(self, x_m, x_s): # x_q exactly is x_s in inference
        e_m = self.M_Encoder(x_m)
        e_q = self.Q_Encoder(x_s)
        
        # Concat
        eq_broadcast = e_q.expand(-1, -1, e_m.size(2), -1)
        combined = torch.cat([e_m, eq_broadcast], dim=3).permute(0, 3, 2, 1)
        combined = self.combine_conv(combined)
        combined = combined.permute(0, 3, 2, 1) # BS, 8, 100, 16
        
        # Pitch Encoder
        pitch_latent = self.pitch_encoder(combined).permute(0, 1, 3, 2)
        
        # Timbre Encoder
        timbre_latent = self.timbre_encoder(combined)
        timbre_latent = timbre_latent.expand(-1, -1, pitch_latent.shape[2], -1)
        
        # Concat: f_phi_s
        s_i = torch.cat((pitch_latent, timbre_latent), dim=3) # s_c: batch*N_s, 8, 100, 32
        s_i = s_i.view(self.batch_size * self.N_s, 32, 8, 100)    # [batch*N_s, C=32, H=8, W=100]
        
        # Get Zs_i from E_vae(x_si)
        x_s = x_s.to(torch.float16)
        z_s0 = self.pt_E_VAE(x_s).to(torch.float32) # [batch*N_s, C=8, H=16, W=100]
        z_s0 = z_s0.permute(0, 1, 3, 2)
        
        # Partition
        z_m0, s_c = self.partitioner(z_s0, s_i)  # z_m0: [batch*N_s*L, 512], s_c_patched: [batch*N_s*L, 1024]
        print(z_m0.shape, s_c.shape)
        # # Sample timesteps
        # t = self.sample_timesteps(z_m0.size(0))  # [batch*N_s*L]
        # # Get noise
        # z_t, noise = self.forward_diffusion_sample(z_m0, t)
        
        # # Prepare conditioning vector (concatenate s_c_patched and timestep embedding)
        # timestep_embedding = self.get_timestep_embedding(t, self.dit.transformer_blocks[0].self_attn.self_attn.embed_dim)
        # condition = torch.cat([s_c_patched, timestep_embedding], dim=-1)  # [batch*N_s*L, 1024 + embed_dim]
        # # Predict z0 using DiT
        # z_t = z_t.unsqueeze(-1)  # [batch*N_s*L, 512, 1]
        # z_t = z_t.permute(2, 0, 1)  # [1, batch*N_s*L, 512]
        # pred_z0 = self.dit(z_t, condition)  # [1, batch*N_s*L, 512]
        # pred_z0 = pred_z0.squeeze(0).permute(1, 0)  # [batch*N_s*L, 512]
        # # Compute loss
        # z_s0_flat = z_s0.view(batch_size * N_s, -1)  # [batch*N_s, 8*16*100]
        # z_s0_flat = z_s0_flat.view(batch_size * N_s * 512)  # Adjust as per actual dimensions
        # loss = nn.MSELoss()(pred_z0, z_s0_flat)  # Ensure shapes align
        
        return s_c
    
    
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BS, N_s = 1, 4
    x_q = torch.randn(BS*N_s, 1, 64, 400).to(device)#.to(torch.float16)
    x_m = torch.randn(BS*N_s, 1, 64, 400).to(device)#.to(torch.float16)
    model = DisMix_LDM(
        batch_size=BS,
        N_s=N_s,
    ).to(device)
    
    res = model(x_m, x_q)
    print(res.shape)
