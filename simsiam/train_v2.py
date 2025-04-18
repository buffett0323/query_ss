import os
import random
import math
import torch
import torchaudio
import json
import librosa
import librosa.display
import argparse
import scipy.interpolate
import scipy.stats

import numpy as np
import torch.nn as nn
import nnAudio.features

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torchaudio.transforms as T
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchaudio.transforms import MelSpectrogram
from pytorch_lightning import LightningDataModule
from typing import Optional
from librosa import effects
from tqdm import tqdm
from torchaudio.functional import pitch_shift
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, TimeMask


from transforms import CLARTransform, RandomFrequencyMasking
from utils import yaml_config_hook, plot_spec_and_save, resize_spec
from spec_aug.spec_augment_pytorch import spec_augment, visualization_spectrogram
from spec_aug.spec_augment_pytorch import SpecAugment

from dataset import SimpleBPDataset
from augmentation import AugmentationModuleTT
from model import SimSiam




if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Simsiam_BP")

    config = yaml_config_hook("config/ssbp_byola.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    train_dataset = SimpleBPDataset(
        sample_rate=args.sample_rate,
        data_dir=args.data_dir,
        piece_second=args.piece_second,
        segment_second=args.segment_second,
        random_slice=args.random_slice,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        # drop_last=args.drop_last,
        # persistent_workers=args.persistent_workers,
        # prefetch_factor=8, #4,
    )
    
    
    to_spec = nnAudio.features.MelSpectrogram(
        sr=args.sample_rate,
        n_fft=args.n_fft,
        win_length=args.window_size,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        center=True,
        power=2,
        verbose=False,
    ).to(device)
    
    
    print("=> Creating model with backbone encoder: '{}'".format(args.encoder_name))
    model = SimSiam(
        args=args,
        dim=args.dim,
        pred_dim=args.pred_dim,
    ).to(device)
    
    for batch in tqdm(train_loader):
        x1, x2, _ = batch
        x1 = x1.to(device)
        x2 = x2.to(device)
        x1 = to_spec(x1).unsqueeze(1)
        x2 = to_spec(x2).unsqueeze(1)
        print("x1.shape", x1.shape, "x2.shape", x2.shape)
        
        p1, p2, z1, z2 = model(x1, x2)
        print("p1.shape", p1.shape, "p2.shape", p2.shape, \
            "z1.shape", z1.shape, "z2.shape", z2.shape)
        
        
        break