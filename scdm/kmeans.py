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
from sklearn.cluster import KMeans
from pytorch_lightning import Trainer, LightningModule
from diffusers import AudioLDM2Pipeline
from diffusers.models import AutoencoderKL
from audioldm_train.modules.diffusionmodules.model import Encoder, Decoder
from audioldm_train.modules.diffusionmodules.distributions import DiagonalGaussianDistribution
import warnings
warnings.filterwarnings("ignore")



class AudioFeatureExtractor(nn.Module):
    def __init__(self, sample_rate=16000, n_mfcc=13):
        super().__init__()
        self.mfcc = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)

    def forward(self, waveform):
        return self.mfcc(waveform)
    
    

class KMeansClustering(nn.Module):
    def __init__(
        self, 
        n_clusters=100
    ):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters)

    def fit(self, features):
        self.kmeans.fit(features)

    def predict(self, features):
        return self.kmeans.predict(features)

