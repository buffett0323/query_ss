import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, TimeStretch, PitchShift, AdditiveNoise
from tqdm import tqdm


# Timbre Encoder with Projection Head
class TimbreSimCLR(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, projection_dim=64):
        super(TimbreSimCLR, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(input_dim // 4 * input_dim // 4 * 64, hidden_dim),
            nn.ReLU()
        )
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections