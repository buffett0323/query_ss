# Using original track, without .dbfss
import os
import random
import math
import torch
import json
import librosa
import argparse
import scipy.signal
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




# Beatport Dataset
class BPDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        duration,
        data_dir="/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_4secs_npy",
        split="train",
        need_transform=True,
        random_slice=True,
        stems=["vocals", "bass", "drums", "other"], # VBDO
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

    
    def get_label(self, folder, stem):
        style = self.class_dict[folder.split('_')[0]]
        return self.labels.index(style) * len(self.stems) + self.stems.index(stem)

    
    def __len__(self): #""" Total we got 175698 files * 4 tracks """
        return len(self.data_path_list)
    

    def __getitem__(self, idx):
        path = self.data_path_list[idx]
        x = np.load(path)
        if self.split == "test":
            return x[int(x.shape[0]/4) : int(x.shape[0]*3/4)], \
                torch.tensor(self.label_list[idx], dtype=torch.int64)
        
        # Augmentation for training
        x_i, x_j = x[:x.shape[0]//2], x[x.shape[0]//2:]
        
        if self.need_transform:
            x_i, x_j = self.transform(x_i, x_j)
        
        return torch.tensor(x_i, dtype=torch.float32), \
                torch.tensor(x_j, dtype=torch.float32), \
                torch.tensor(self.label_list[idx], dtype=torch.int64)


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




class CLARTransform(nn.Module):
    def __init__(
        self, 
        sample_rate,
        duration,
    ):
        super(CLARTransform, self).__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.transforms = [
            self.pitch_shift_transform,
            self.add_fade_transform,
            self.add_noise_transform,
            self.time_masking_transform,
            self.time_shift_transform,
            self.time_stretch_transform,
        ]
        
        
    def pitch_shift_transform(self, x, n_steps=15):
        return effects.pitch_shift(x, sr=self.sample_rate, n_steps=torch.randint(low=-n_steps, high=n_steps, size=[1]).item())

    def add_fade_transform(self, x, max_fade_size=.5):
        def _fade_in(fade_shape, waveform_length, fade_in_len):
            fade = np.linspace(0, 1, fade_in_len)
            ones = np.ones(waveform_length - fade_in_len)
            if fade_shape == 0:
                fade = fade
            if fade_shape == 1:
                fade = np.power(2, (fade - 1)) * fade
            if fade_shape == 2:
                fade = np.log10(.1 + fade) + 1
            if fade_shape == 3:
                fade = np.sin(fade * math.pi / 2)
            if fade_shape == 4:
                fade = np.sin(fade * math.pi - math.pi / 2) / 2 + 0.5
            return np.clip(np.concatenate((fade, ones)), 0, 1)

        def _fade_out(fade_shape, waveform_length, fade_out_len):
            fade = torch.linspace(0, 1, fade_out_len)
            ones = torch.ones(waveform_length - fade_out_len)
            if fade_shape == 0:
                fade = - fade + 1
            if fade_shape == 1:
                fade = np.power(2, - fade) * (1 - fade)
            if fade_shape == 2:
                fade = np.log10(1.1 - fade) + 1
            if fade_shape == 3:
                fade = np.sin(fade * math.pi / 2 + math.pi / 2)
            if fade_shape == 4:
                fade = np.sin(fade * math.pi + math.pi / 2) / 2 + 0.5
            return np.clip(np.concatenate((ones, fade)), 0, 1)

        waveform_length = x.shape[0]
        fade_shape = np.random.randint(5)
        fade_out_len = np.random.randint(int(x.shape[0] * max_fade_size))
        fade_in_len = np.random.randint(int(x.shape[0] * max_fade_size))
        return np.float32(
            _fade_in(fade_shape, waveform_length, fade_in_len) * 
            _fade_out(fade_shape, waveform_length, fade_out_len) * 
            x
        )


    def add_noise_transform(self, x):
        noise_type = random.choice(['white', 'brown', 'pink'])
        snr = random.uniform(0.5, 1.5)  # Signal-to-noise ratio
        
        if noise_type == 'white':
            noise = np.random.normal(0, 1, len(x))
        elif noise_type == 'brown':
            noise = np.cumsum(np.random.normal(0, 1, len(x)))
            noise = noise / np.max(np.abs(noise))
        else:  # pink noise
            freqs = np.fft.rfftfreq(len(x))
            noise = np.fft.irfft(np.random.randn(len(freqs)) / (freqs + 1e-6))
        
        noise = noise / np.max(np.abs(noise))
        x = x + noise / snr
        return np.clip(x, -1, 1)


    def time_masking_transform(self, x, sr=0.125):
        sr = int(x.shape[0] * sr)
        start = np.random.randint(x.shape[0] - sr)
        x[start: start + sr] = np.float32(np.random.normal(0, 0.01, sr))
        return x

    def time_shift_transform(self, x, shift_rate=8000):
        return np.roll(x, torch.randint(low=-shift_rate, high=shift_rate, size=[1]).item())

    def time_stretch_transform(self, x):
        x = effects.time_stretch(x, rate=random.uniform(0.5, 1.5))
        x = librosa.resample(x, orig_sr=x.shape[0] / self.duration, target_sr=self.sample_rate)
        if x.shape[0] > (self.sample_rate * self.duration):
            return x[:(self.sample_rate * self.duration)]
        return np.pad(x, [0, (self.sample_rate * self.duration) - x.shape[0]])


    def __call__(self, x1, x2):        
        # Apply random augmentations
        transform1, transform2 = random.sample(self.transforms, 2)
        x1 = transform1(x1)
        x2 = transform2(x2)
        return x1, x2




if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="SimCLR_BP")

    config = yaml_config_hook("ssbp_pl_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    
    ds = SlakhDataset(args.sample_rate, args.segment_second, split="train")
    for i in range(10):
        print(ds[i][2])
    
    # dm = BPDataModule(
    #     args=args,
    #     data_dir=args.data_dir, 
    # )
    # dm.setup()
    
    
    # for tr in tqdm(dm.train_dataloader()):
    #     pass; #print(tr[0].shape)
