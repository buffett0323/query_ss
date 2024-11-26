import os
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
    
from audioldm_train.modules.diffusionmodules.model import Encoder, Decoder
from audioldm_train.modules.diffusionmodules.distributions import DiagonalGaussianDistribution


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
    
    
class MixtureEncoder(nn.Module):
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


    def forward(self, x):
        h = self.encoder(x)
        return h.permute(0, 1, 3, 2)
    
    
    
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
        self.E_phai_tau = Decoder(
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
        self.f_phai_tau = Encoder(
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
        y_hat = self.E_phai_tau(x) # BS, 8, 100, 16 --> BS, 129, 400
        y_hat_sb = self.sb_layer(y_hat)
        tau = self.f_phai_tau(y_hat_sb.unsqueeze(1)) # BS, 129, 400 --> BS, 8, 100, 16
        return tau

    
class DisMix_LDM(nn.Module):
    def __init__(
        self,
    ):
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
        
        self.M_Encoder = MixtureEncoder(
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
        
        self.combine_conv = nn.Conv2d(
            in_channels=32,  # 16 from em and 16 from eq after concatenation
            out_channels=16,  # Transform back to original feature dimension
            kernel_size=1,
            stride=1,
            padding=0,
        )
        
        self.pitch_encoder = PitchEncoder()
        # self.timbre_encoder = TimbreEncoder(input_dim, latent_dim)
        # self.decoder = Decoder(latent_dim, output_dim)
        # self.diffusion_transformer = DiffusionTransformer(latent_dim, num_heads, num_layers)

    def forward(self, x_m, x_q):
        # Encoder for Mixture and Query
        e_m = self.M_Encoder(x_m)
        e_q = self.Q_Encoder(x_q)
        
        # Concat
        eq_broadcast = e_q.expand(-1, -1, e_m.size(2), -1)
        combined = torch.cat([e_m, eq_broadcast], dim=3).permute(0, 3, 2, 1)
        combined = self.combine_conv(combined)
        combined = combined.permute(0, 3, 2, 1) # BS, 8, 100, 16
        
        # Pitch Encoder
        tau = self.pitch_encoder(combined)
        
        
        return combined
    
    
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BS = 2
    x_q = torch.randn(BS, 1, 64, 400).to(device)
    x_m = torch.randn(BS, 1, 64, 400).to(device)
    model = DisMix_LDM().to(device)
    res = model(x_m, x_q)
    print(res.shape)
