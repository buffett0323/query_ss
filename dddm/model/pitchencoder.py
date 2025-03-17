from modules_sf.attentions import *
from torch import nn
import torch
from torch.nn import functional as F
from model.styleencoder import Conv1dGLU, Mish

from diffusers import AudioLDM2Pipeline
from diffusers.models import AutoencoderKL
from audioldm_train.modules.diffusionmodules.model import Encoder, Decoder

class StochasticBinarizationLayer(nn.Module):
    def __init__(self):
        super(StochasticBinarizationLayer, self).__init__()
    
    def forward(self, logits):
        prob = torch.sigmoid(logits)  # Probabilities
        if self.training:
            h = torch.rand_like(prob)
            binary = (prob > h).float() 
        else:
            binary = (prob > 0.5).float() 
        
        # Use the STE: Forward pass uses binary, backward pass uses the probabilities
        return binary + (prob - prob.detach())
    
    

class PitchEncoder(nn.Module):
    """
    torch.Size([4, 80, 200]) --> torch.Size([4, 128, 200])
    """
    
    def __init__(
        self, 
        in_dim=64, 
        hidden_dim=256, 
        pitch_classes=129,
        out_dim=64,
        dropout_rate=0.1,
    ):
        # TODO: AudioLDM Framework Test
        super(PitchEncoder, self).__init__()

        self.in_dim = in_dim 
        self.hidden_dim = hidden_dim
        self.pitch_classes = pitch_classes
        self.out_dim = out_dim
        self.kernel_size = 5
        self.n_head = 2
        self.dropout = dropout_rate
        
        
        # Transcriber: Linear layers for pitch classification
        self.transcriber = nn.Sequential(
            nn.Conv1d(self.in_dim, self.hidden_dim, 1),
            Mish(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.hidden_dim, self.pitch_classes, 1),
            Mish(),
            nn.Dropout(self.dropout),
        )

        # Stochastic Binarization (SB) Layer: Converts pitch logits to a binary representation
        self.sb_layer = StochasticBinarizationLayer()

        # Projection Layer: Project the binarized pitch representation to the latent space
        self.fc_proj = nn.Sequential(
            nn.Conv1d(self.pitch_classes, self.out_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.out_dim, self.out_dim, 1),
        )
        

    def forward(self, x, mask=None):
        pitch_logits = self.transcriber(x) # * mask # Transcriber
        y_bin = self.sb_layer(pitch_logits) # SB Layer
        pitch_latent = self.fc_proj(y_bin) # Projection Layer
        
        return pitch_latent, pitch_logits
    


class PitchEncoder_ALDM(nn.Module):
    """
    # BS, 8, 100, 16 --> BS, 129, 400
    """
    def __init__(
        self,
        in_channels=8,
        out_channels=88,
    ):
        super().__init__()
        self.E_phai_nu = Decoder(
            ch=64,  # Base number of channels; can be adjusted based on model capacity
            out_ch=out_channels,  # Desired number of output channels (pitch values)
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
            out_ch=in_channels,  # Not directly used in the current implementation but kept for consistency
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
        y_hat = self.E_phai_nu(x)
        print("y_hat", y_hat.shape)
        y_hat_sb = self.sb_layer(y_hat)
        print("y_hat_sb", y_hat_sb.shape)
        pitch_latent = self.f_phai_nu(y_hat_sb.unsqueeze(1)) # BS, 1, 129, 400 --> BS, 8, 100, 16
        return y_hat, pitch_latent