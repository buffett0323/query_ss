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
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchaudio.transforms import MelSpectrogram
from pytorch_lightning import LightningDataModule
from typing import Optional
from librosa import effects
from tqdm import tqdm
from torchaudio.functional import pitch_shift

from transforms import CLARTransform, AudioFXAugmentation
from utils import yaml_config_hook, train_test_split_BPDataset

# Beatport Dataset
class BPDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        duration,
        data_dir,
        augment_func,
        n_mels=128,
        split="train",
        melspec_transform=False,
        data_augmentation=True,
        random_slice=True,
        stems=["other"], #["vocals", "bass", "drums", "other"], # VBDO
    ):
        # Load split files from txt file
        with open(f"info/{split}_bp.txt", "r") as f:
            bp_listdir = [line.strip() for line in f.readlines()]
            
        with open("info/labels.txt", "r") as f:
            self.labels = [line.strip() for line in f.readlines()]
            
        with open("info/class_dict.json", "r", encoding="utf-8") as f:
            self.class_dict = json.load(f)

        self.stems = stems
        self.data_path_list = [
            os.path.join(data_dir, folder, f"{stem}.npy")
            for folder in bp_listdir
                for stem in stems
        ]
        self.label_list = [
            self.get_label(folder, stem)
            for folder in bp_listdir
                for stem in stems
        ]
        
        self.augment_func = augment_func # CLARTransform
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.slice_duration = sample_rate * duration
        self.split = split
        self.melspec_transform = melspec_transform
        self.data_augmentation = data_augmentation
        self.random_slice = random_slice
    
    def get_label(self, folder, stem):
        style = self.class_dict[folder.split('_')[0]]
        return self.labels.index(style) * len(self.stems) + self.stems.index(stem)

    
    def mel_spec_transform(self, x):
        mel_spec = librosa.feature.melspectrogram(
            y=x, sr=self.sample_rate, n_mels=self.n_mels, 
            n_fft=1024, hop_length=256, fmax=8000,
        )    
        return librosa.power_to_db(mel_spec, ref=np.max)
    
        
    def __len__(self): #""" Total we got 175698 files * 4 tracks """
        return len(self.data_path_list)
    
    
    def __getitem__(self, idx):
        path = self.data_path_list[idx]
        x = np.load(path)
        
        x_i, x_j = x[:x.shape[0]//2], x[x.shape[0]//2:]
        
        # Augmentation for training
        if self.data_augmentation:
            x_i, x_j = self.augment_func(x_i, x_j)
            
        if self.melspec_transform:
            x_i, x_j = self.mel_spec_transform(x_i), self.mel_spec_transform(x_j)
            
        x_i = torch.tensor(x_i, dtype=torch.float32)
        x_j = torch.tensor(x_j, dtype=torch.float32)
        
        return x_i, x_j, torch.tensor(self.label_list[idx], dtype=torch.int64)


class BPDataModule(LightningDataModule):
    def __init__(
        self,
        args,
        data_dir="/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_4secs_npy", 
    ):
        super(BPDataModule, self).__init__()
        self.args = args
        self.data_dir = data_dir
        self.pin_memory = args.pin_memory
        self.drop_last = args.drop_last
        self.num_workers = args.workers #args.num_workers
        
    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_ds = BPDataset(
            sample_rate=self.args.sample_rate,
            duration=self.args.segment_second,
            data_dir=self.data_dir,
            split="train",
            need_transform=self.args.need_clar_transform,
            random_slice=self.args.random_slice,
        )   
        self.val_ds = BPDataset(
            sample_rate=self.args.sample_rate,
            duration=self.args.segment_second,
            data_dir=self.data_dir,
            split="valid",
            need_transform=self.args.need_clar_transform,
            random_slice=self.args.random_slice,
        )
        self.test_ds = BPDataset(
            sample_rate=self.args.sample_rate,
            duration=self.args.segment_second,
            data_dir=self.data_dir,
            split="test",
            need_transform=self.args.need_clar_transform,
            random_slice=self.args.random_slice,
        )
        self.memory_ds = BPDataset(
            sample_rate=self.args.sample_rate,
            duration=self.args.segment_second,
            data_dir=self.data_dir,
            split="memory",
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




if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="SimCLR_BP")

    config = yaml_config_hook("config/ssbp_6secs.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    
    # ds = SlakhDataset(args.sample_rate, args.segment_second, split="train")
    # for i in range(10):
    #     print(ds[i][2])
    
    
    train_dataset = BPDataset(
        sample_rate=args.sample_rate, 
        duration=args.segment_second, 
        data_dir=args.data_dir,
        augment_func=CLARTransform(
            sample_rate=args.sample_rate,
            duration=int(args.segment_second/2),
            n_mels=args.n_mels,
        ),
        n_mels=args.n_mels,
        split="train",
        melspec_transform=args.melspec_transform,
        data_augmentation=args.data_augmentation,
        resize=True,
        resize_target_size=224,
        random_slice=False,
        stems=['other'],
    )
    ts = train_dataset[0]
    print(ts[0].shape, ts[1].shape, ts[2])
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=4, shuffle=True,
    #     num_workers=args.workers, pin_memory=True, drop_last=True)
    
    # for ds in train_loader:
    #     print(ds[0].shape, ds[1].shape) # 256, 128, 94
    #     break
    
    
    # for tr in tqdm(dm.train_dataloader()):
    #     pass; #print(tr[0].shape)
