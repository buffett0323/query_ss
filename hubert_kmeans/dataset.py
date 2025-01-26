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



class FMA_BP_DS(Dataset):
    def __init__(
        self, 
        sr=44100,
        fma_path='/mnt/gestalt/database/FMA/fma_track/audio',
        bp_path='/mnt/gestalt/database/beatport/audio/audio',
    ):
        """
            Both in 44100
        """
        super(FMA_BP_DS, self).__init__()
        
        self.sr = sr

        # FMA + Beatport
        self.fma_list = [os.path.join(fma_path, f) for f in tqdm(os.listdir(fma_path))]
        self.bp_list = [
            os.path.join(bp_path, style, track)
            for style in tqdm(os.listdir(bp_path))
                for track in os.listdir(os.path.join(bp_path, style))
        ]
        self.file_path =  self.bp_list

    def __len__(self):
        return len(self.file_path)
    
    
    def __getitem__(self, idx):
        audio_path = self.file_path[idx]
        _, sr = librosa.load(audio_path, sr=None)  # Load audio without resampling
        
        return
    




if __name__ == "__main__":
    DS = FMA_BP_DS()
    print(len(DS))
    DS[0]
    
    
    