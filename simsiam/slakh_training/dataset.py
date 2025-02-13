# Using original track, without .dbfss
import os
import random
import math
import torch
import torchaudio
import json
import librosa
import argparse
import scipy.interpolate
import scipy.stats
import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchaudio.transforms import MelSpectrogram
from pytorch_lightning import LightningDataModule
from typing import Optional
from librosa import effects
from tqdm import tqdm
from torchaudio.functional import pitch_shift
from utils import yaml_config_hook, train_test_split_BPDataset


# Beatport Dataset
class SlakhDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        duration,
        data_dir="/mnt/gestalt/home/ddmanddman/slakh2100_buffett/",
        split="train",
        need_transform=True,
        random_slice=True,
    ):
        # Load split files from txt file
        self.data_dir = os.path.join(data_dir, split)
        self.data_path_list = [
            os.path.join(self.data_dir, folder, file)
            for folder in os.listdir(self.data_dir)
                for file in os.listdir(os.path.join(self.data_dir, folder))
                    if file.endswith(".npy")
        ]
        with open("info/slakh_label.txt", "r") as f:
            self.labels = [line.strip() for line in f.readlines()]
            
        self.transform = CLARTransform(
            sample_rate=sample_rate,
            duration=int(duration/2),
        )
        self.sample_rate = sample_rate
        self.duration = duration
        self.slice_duration = sample_rate * duration
        self.split = split
        self.need_transform = need_transform
        self.random_slice = random_slice


    def __len__(self):
        return len(self.data_path_list)
    

    def __getitem__(self, idx):
        path = self.data_path_list[idx]
        label = self.labels.index(path.split('/')[-1].split('.npy')[0])
        
        x = np.load(path)
        
        if self.split == "test":
            return x[int(x.shape[0]/4) : int(x.shape[0]*3/4)], \
                torch.tensor(label, dtype=torch.int64)
        
        # Augmentation for training
        x_i, x_j = x[:x.shape[0]//2], x[x.shape[0]//2:]
        
        if self.need_transform:
            x_i, x_j = self.transform(x_i, x_j)

        return torch.tensor(x_i, dtype=torch.float32), \
                torch.tensor(x_j, dtype=torch.float32), \
                torch.tensor(label, dtype=torch.int64)


class SlakhDataModule(LightningDataModule):
    def __init__(
        self,
        args,
        data_dir="/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_4secs_npy", 
    ):
        super(SlakhDataModule, self).__init__()
        self.args = args
        self.data_dir = data_dir
        self.pin_memory = args.pin_memory
        self.drop_last = args.drop_last
        self.num_workers = args.workers #args.num_workers
        
    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_ds = SlakhDataset(
            sample_rate=self.args.sample_rate,
            duration=self.args.segment_second,
            data_dir=self.data_dir,
            split="train",
            need_transform=self.args.need_clar_transform,
            random_slice=self.args.random_slice,
        )   
        self.val_ds = SlakhDataset(
            sample_rate=self.args.sample_rate,
            duration=self.args.segment_second,
            data_dir=self.data_dir,
            split="validation",
            need_transform=self.args.need_clar_transform,
            random_slice=self.args.random_slice,
        )
        self.test_ds = SlakhDataset(
            sample_rate=self.args.sample_rate,
            duration=self.args.segment_second,
            data_dir=self.data_dir,
            split="test",
            need_transform=self.args.need_clar_transform,
            random_slice=self.args.random_slice,
        )
        self.memory_ds = SlakhDataset(
            sample_rate=self.args.sample_rate,
            duration=self.args.segment_second,
            data_dir=self.data_dir,
            split="train",
            need_transform=self.args.need_clar_transform,
            random_slice=self.args.random_slice,
        )
            
            
    def train_dataloader(self):
        """Train DataLoader with Distributed Sampler for DDP"""
        train_sampler = DistributedSampler(self.train_ds) if self.trainer.world_size > 1 else None
        return DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size,
            sampler=train_sampler,  # Use DistributedSampler instead of shuffle
            shuffle=(train_sampler is None),
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.args.persistent_workers,  # Keep workers alive to reduce loading overhead
            prefetch_factor=4 if self.num_workers > 0 else None,  # Prefetch data in advance
        )

    def val_dataloader(self):
        """Validation DataLoader"""
        return DataLoader(
            self.val_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.args.persistent_workers,  # Keep workers alive to reduce loading overhead
            prefetch_factor=4 if self.num_workers > 0 else None,  # Prefetch data in advance
        )

    def test_dataloader(self):
        """Test DataLoader"""
        return DataLoader(
            self.test_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.args.persistent_workers,  # Keep workers alive to reduce loading overhead
            prefetch_factor=4 if self.num_workers > 0 else None,  # Prefetch data in advance
        )
    
    @property
    def num_samples(self) -> int:
        self.setup(stage = 'fit')
        return len(self.train_ds)