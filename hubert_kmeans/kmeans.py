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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import load_audio, get_mel_spec, get_log_mel_spec, get_cqt
import warnings
warnings.filterwarnings("ignore")

    
    

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

