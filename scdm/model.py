import os
import argparse
import torch
import torchvision
import librosa
import numpy as np
import torch.nn as nn
from pytorch_lightning import Trainer, LightningModule
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.transforms as T
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models.vision_transformer import VisionTransformer


class SimCLR(nn.Module):
    def __init__(
        self,
        args,
        n_features=512, 
        projection_dim=128,
    ):
        super(SimCLR, self).__init__()
        self.args = args
        self.encoder = None
            
        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features), #, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, projection_dim), #, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i, h_j = self.encoder(x_i), self.encoder(x_j)
        z_i, z_j = self.projector(h_i), self.projector(h_j)
        
        return h_i, h_j, z_i, z_j
    
    
class DisMix(nn.Module):
    def __init__(
        self,
        args,
    ):
        super(DisMix, self).__init__()
        self.args = args
        
        # Latents of mixture and query
        self.E_m = None
        self.E_qi = None
        
        # Pitch and Timbre Encoder
        self.E_Pitch = None
        self.E_Timbre = None
        
        # Reconstruct
        self.Decoder = None
        
        
    def forward(self, x_m, x_q):
        x_m = self.E_m(x_m)
        x_q = self.E_q(x_q)
        
        pitch_latent, pitch_logits = self.E_Pitch(x_m, x_q)
        timbre_latent, mean, logvar = self.E_Timbre(x_m, x_q)
        
        return self.Decoder(pitch_latent, timbre_latent)

        