from modules_sf.attentions import *
from torch import nn
import torch
from torch.nn import functional as F
from model.styleencoder import Conv1dGLU, Mish

    
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
        pitch_logits = self.transcriber(x)*mask # Transcriber
        y_bin = self.sb_layer(pitch_logits) # SB Layer
        pitch_latent = self.fc_proj(y_bin) # Projection Layer
        
        return pitch_latent, pitch_logits