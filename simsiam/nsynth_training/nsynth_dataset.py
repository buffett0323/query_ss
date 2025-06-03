# Using original track, without .dbfss
import os
import random
import math
import pickle
import torch
import torchaudio
import json
import librosa
import librosa.display
import argparse
import scipy.interpolate
import scipy.stats
import nnAudio.features

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torchaudio.transforms as T
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchaudio.transforms import MelSpectrogram
from pytorch_lightning import LightningDataModule
from typing import Optional
from tqdm import tqdm
from scipy.stats import entropy
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, TimeMask

from transforms import CLARTransform, RandomFrequencyMasking
from utils import yaml_config_hook, plot_spec_and_save, resize_spec
from spec_aug.spec_augment_pytorch import spec_augment, visualization_spectrogram
from spec_aug.spec_augment_pytorch import SpecAugment


# Beatport Dataset
class NsynthDataset(Dataset):
    """ For 4 seconds audio data """
    def __init__(
        self,
        sample_rate,
        data_dir,
        piece_second=4,
        segment_second=0.95,
        window_size=1024,
        hop_length=160,
        split="train",
    ):

        if split != "train":
            data_dir = data_dir.replace("nsynth-train", f"nsynth-{split}")

        self.data_path_list = [
            os.path.join(data_dir, file)
            for file in tqdm(os.listdir(data_dir), desc="Loading NSynth Dataset")
        ]

        self.sample_rate = sample_rate
        self.segment_second = segment_second
        self.window_size = window_size
        self.hop_length = hop_length
        self.piece_length = int(piece_second * sample_rate)
        self.duration = int(segment_second * sample_rate)
        self.split = split

    def __len__(self):
        return len(self.data_path_list)


    def __getitem__(self, idx):
        # Load audio data from .npy
        path = self.data_path_list[idx]
        x = np.load(path).squeeze(0) #, mmap_mode='r').squeeze(0)
        return x[:self.duration], path




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simsiam_BP")

    config = yaml_config_hook("config/nsynth_convnext.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    train_dataset = NsynthDataset(
        sample_rate=args.sample_rate,
        data_dir=args.data_dir,
        piece_second=args.piece_second,
        segment_second=args.segment_second,
        window_size=args.window_size,
        hop_length=args.hop_length,
    )

    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)

    segment_dict = dict()
    energy_track, eliminated_track = [], []

    for i in tqdm(range(len(train_dataset))):
        x, path = train_dataset[i]
        print(x.shape, path)
        break
