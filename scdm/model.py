import os
import argparse
import torch
import torchvision
import librosa
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.transforms as T
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from diffusers import AudioLDM2Pipeline
from diffusers.models import AutoencoderKL
from audioldm_train.modules.diffusionmodules.model import Encoder, Decoder
from audioldm_train.modules.diffusionmodules.distributions import DiagonalGaussianDistribution
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Could not initialize NNPACK")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from loss import ELBOLoss, NT_Xent

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
        # h = h.to(torch.float32)
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
            # x = x.to(torch.float16)
            h = self.encoder(x).permute(0, 1, 3, 2)
        return self.conv(h) #.to(torch.float32)
    
    

class Conv1DEncoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, norm_layer, activation):
        super(Conv1DEncoder, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding)
        self.norm = norm_layer(output_channels) if norm_layer else None
        self.activation = activation() if activation else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = x.transpose(1, 2)  # Change shape from (batch, channels, sequence) to (batch, sequence, channels)
            x = self.norm(x)        # Apply LayerNorm to the last dimension (channels)
            x = x.transpose(1, 2)  # Change back shape to (batch, channels, sequence)
        if self.activation:
            x = self.activation(x)
        return x


class MixtureQueryEncoder(nn.Module):
    """
        R^128x10 --> R^64
    """
    def __init__(
        self,
        input_dim=128,
        hidden_dim=768,
        output_dim=64,
    ):
        super(MixtureQueryEncoder, self).__init__()
        self.encoder_layers = nn.Sequential(
            Conv1DEncoder(input_dim, hidden_dim, 3, 1, 0, nn.LayerNorm, nn.ReLU),
            Conv1DEncoder(hidden_dim, hidden_dim, 3, 1, 1, nn.LayerNorm, nn.ReLU),
            Conv1DEncoder(hidden_dim, hidden_dim, 4, 2, 1, nn.LayerNorm, nn.ReLU),
            Conv1DEncoder(hidden_dim, hidden_dim, 3, 1, 1, nn.LayerNorm, nn.ReLU),
            Conv1DEncoder(hidden_dim, hidden_dim, 3, 1, 1, nn.LayerNorm, nn.ReLU),
            Conv1DEncoder(hidden_dim, output_dim, 1, 1, 1, None, None)
        )

    def forward(self, x):
        x = self.encoder_layers(x)
        return x #torch.mean(x, dim=-1)  # Mean pooling along the temporal dimension



class TimbreEncoder(nn.Module):
    def __init__(
        self, 
        input_dim=128, 
        hidden_dim=256, 
        output_dim=64  # Latent space dimension for timbre
    ):
        super(TimbreEncoder, self).__init__()

        # Shared architecture with Eφν (PitchEncoder)
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # First layer
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # Last layer
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Gaussian parameterization layers
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.logvar_layer = nn.Linear(hidden_dim, output_dim)

    def reparameterize(self, mean, logvar):
        """
            Reparameterization trick to sample from N(mean, var)
            Sampling by μφτ (·) + ε σφτ (·)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + (eps * std)

    def forward(self, em, eq):
        # Concatenate the mixture and query embeddings
        concat_input = torch.cat([em, eq], dim=-1)  # Concatenate along feature dimension
        hidden_state = self.shared_layers(concat_input)  # Shared hidden state output

        # Calculate mean and log variance
        mean, logvar = self.mean_layer(hidden_state), self.logvar_layer(hidden_state)  # Gaussian distribution

        # Sample the timbre latent using the reparameterization trick
        timbre_latent = self.reparameterize(mean, logvar)

        return timbre_latent, mean, logvar
    
    
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
        input_dim=128, 
        hidden_dim=256, 
        pitch_classes=52, # true labels not 0-51
        output_dim=64
    ):
        super(PitchEncoder, self).__init__()

        # Transcriber: Linear layers for pitch classification
        self.transcriber = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pitch_classes),  # Output logits for pitch classification
        )

        # Stochastic Binarization (SB) Layer: Converts pitch logits to a binary representation
        self.sb_layer = StochasticBinarizationLayer()

        # Projection Layer: Project the binarized pitch representation to the latent space
        self.fc_proj = nn.Sequential(
            nn.Linear(pitch_classes, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )


    def forward(self, em, eq):
        concat_input = torch.cat([em, eq], dim=-1)  # Concatenate em and eq
        pitch_logits = self.transcriber(concat_input) # Transcriber
        y_bin = self.sb_layer(pitch_logits) # SB Layer # Apply binarisation
        pitch_latent = self.fc_proj(y_bin) # Projection Layer
        
        return pitch_latent, pitch_logits



class FiLM(nn.Module):
    def __init__(self, pitch_dim, timbre_dim):
        super(FiLM, self).__init__()
        self.scale = nn.Linear(timbre_dim, pitch_dim)
        self.shift = nn.Linear(timbre_dim, pitch_dim)
    
    def forward(self, pitch_latent, timbre_latent):
        scale = self.scale(timbre_latent)
        shift = self.shift(timbre_latent)
        return scale * pitch_latent + shift


class DisMixDecoder(nn.Module):
    def __init__(
        self, 
        pitch_dim=64, 
        timbre_dim=64, 
        gru_hidden_dim=256, 
        output_dim=128, 
        num_frames=32, #10,
        num_layers=2
    ):
        super(DisMixDecoder, self).__init__()
        self.num_frames = num_frames
        
        self.film = FiLM(pitch_dim, timbre_dim)
        self.gru = nn.GRU(input_size=pitch_dim, hidden_size=gru_hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(gru_hidden_dim * 2, output_dim)  # Bi-directional GRU output dimension is doubled
    
    def forward(self, pitch_latents, timbre_latents):
        # FiLM layer: modulates pitch latents based on timbre latents
        source_latents = self.film(pitch_latents, timbre_latents)
        source_latents = source_latents.unsqueeze(1).repeat(1, self.num_frames, 1) # Expand source_latents along time axis if necessary
        output, _ = self.gru(source_latents)
        output = self.linear(output).transpose(1, 2) # torch.Size([32, 10, 64])
        
        return output # reconstructed spectrogram


class SimCLRDisMix(pl.LightningModule):
    def __init__(
        self,
        vae,
        batch_size=4,
        temperature=0.5,
        input_dim=128, 
        latent_dim=64, 
        hidden_dim=256, 
        gru_hidden_dim=256,
        num_frames=10,
        pitch_classes=52,
        output_dim=128,
        learning_rate=4e-4,
        num_layers=2,
        clip_value=0.5,
        lambda_weight=0.005,
    ):
        super(SimCLRDisMix, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pitch_classes = pitch_classes
        self.clip_value = clip_value
        
        # Latents of mixture and query
        # self.E_m = MixtureQueryEncoder(
        #     input_dim=input_dim,
        # )
        # self.E_q = MixtureQueryEncoder(
        #     input_dim=input_dim,
        # )
        self.E_m = MixtureEncoder(vae=vae) # Pre-trained model
        self.E_q = QueryEncoder(
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
        
        # Pitch and Timbre Encoder
        self.E_Pitch = PitchEncoder(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            pitch_classes=pitch_classes,
            output_dim=latent_dim,
        )
        self.E_Timbre = TimbreEncoder(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=latent_dim,
        )
        
        # Reconstruct
        self.Decoder = DisMixDecoder(
            pitch_dim=latent_dim, 
            timbre_dim=latent_dim, 
            gru_hidden_dim=gru_hidden_dim, 
            output_dim=output_dim, 
            num_frames=num_frames,
            num_layers=num_layers,
        )
        
        # Loss functions
        self.elbo_loss_fn = ELBOLoss() # For ELBO
        self.ce_loss_fn = nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss()  # For pitch supervision
        self.nt_xent_fn = NT_Xent(
            batch_size=batch_size, 
            temperature=temperature, 
            world_size=1,    
        ) # NT-Xent Loss
        
        
    def forward(self, x_m1, x_m2, x_q1, x_q2):
        x_m1, x_m2 = x_m1.repeat_interleave(4, dim=0), x_m2.repeat_interleave(4, dim=0)
        x_q1, x_q2 = x_q1.reshape(-1, 1, 64, 200), x_q2.reshape(-1, 1, 64, 200)
        print(x_m2.shape, x_q1.shape)
        
        # x_m1, x_m2 = self.E_m(x_m1), self.E_m(x_m2)
        x_q1, x_q2 = self.E_q(x_q1), self.E_q(x_q2)
        return x_q1, x_q2 # x_m1, x_m2, x_q1, x_q2
    
        pitch_latent_i, pitch_logits_i = self.E_Pitch(x_m1, x_q1)
        timbre_latent_i, mean_i, logvar_i = self.E_Timbre(x_m1, x_q1)
        
        pitch_latent_j, pitch_logits_j = self.E_Pitch(x_m2, x_q2)
        timbre_latent_j, mean_j, logvar_j = self.E_Timbre(x_m2, x_q2)
        
        s_i = self.Decoder(pitch_latent_i, timbre_latent_i)
        s_j = self.Decoder(pitch_latent_j, timbre_latent_j)
        
        return s_i, s_j, timbre_latent_i, timbre_latent_j, \
                pitch_logits_i, pitch_logits_j, \
                mean_i, logvar_i, mean_j, logvar_j 
    
    
    def training_step(self, batch, batch_idx):
        criterion = self.configure_criterion()
        mix_i, mix_j, stem_i, stem_j, pitch_i, pitch_j = batch
        if mix_i.shape[0] != self.batch_size:
            return None
        
        s_i, s_j, timbre_latent_i, timbre_latent_j, pitch_logits_i, pitch_logits_j, mean_i, logvar_i, mean_j, logvar_j  = self(mix_i, mix_j, stem_i, stem_j)
        
        # Loss Cal
        elbo_loss = criterion["elbo_loss"](mix_latent, query_latent, reconstruction)
        ce_loss = criterion["ce_loss"](pitch_latent, pitch_i) + criterion["ce_loss"](pitch_latent, pitch_j)
        nt_xent_loss = criterion["nt_xent_loss"](timbre_latent_i, timbre_latent_j)
        
        train_loss = elbo_loss + ce_loss + nt_xent_loss
        
        # Log the individual losses and total loss
        self.log("elbo_loss", elbo_loss, on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size, sync_dist=True)
        self.log("ce_loss", ce_loss, on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size, sync_dist=True)
        self.log("nt_xent_loss", nt_xent_loss, on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size, sync_dist=True)
        self.log('train_loss', train_loss, on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size, sync_dist=True)
        return train_loss
    
    
    def configure_criterion(self):
        return {
            "elbo_loss": self.elbo_loss_fn,
            "ce_loss": self.ce_loss_fn,
            "nt_xent_loss": self.nt_xent_fn
        }
    

if __name__ == "__main__":
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    model = SimCLRDisMix(       
        vae=None,
        batch_size=4,
        input_dim=64, 
    ).to(device)

    mix_i, mix_j, stem_i, stem_j, pitch_i, pitch_j = torch.randn([4, 1, 64, 200]).to(device), torch.randn([4, 1, 64, 200]).to(device), \
            torch.randn([4, 4, 64, 200]).to(device), torch.randn([4, 4, 64, 200]).to(device), \
            torch.randn([4, 4, 200]).to(device), torch.randn([4, 4, 200]).to(device)
    
    res = model(mix_i, mix_j, stem_i, stem_j)
    print(res[0].shape, res[-1].shape)