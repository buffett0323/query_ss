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
from dismix_model import Conv1DEncoder, MixtureQueryEncoder, \
    StochasticBinarizationLayer, TimbreEncoder, PitchEncoder
    
    
    
    
class DisMix_LDM(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, num_heads, num_layers):
        super().__init__()
        self.pitch_encoder = PitchEncoder(input_dim, latent_dim)
        self.timbre_encoder = TimbreEncoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, output_dim)
        self.diffusion_transformer = DiffusionTransformer(latent_dim, num_heads, num_layers)

    def forward(self, mixture, query):
        # Encode pitch and timbre
        pitch_latent = self.pitch_encoder(query)
        timbre_mean, timbre_logvar = self.timbre_encoder(mixture)
        timbre_latent = timbre_mean + torch.exp(0.5 * timbre_logvar) * torch.randn_like(timbre_logvar)

        # Combine latents for diffusion
        combined_latents = torch.cat([pitch_latent, timbre_latent], dim=-1)
        transformed_latents = self.diffusion_transformer(combined_latents)

        # Decode to reconstruct the mixture
        return self.decoder(transformed_latents)