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
from librosa import effects
from tqdm import tqdm
from torchaudio.functional import pitch_shift
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, TimeMask


from transforms import CLARTransform, RandomFrequencyMasking
from utils import yaml_config_hook, plot_spec_and_save, resize_spec
from spec_aug.spec_augment_pytorch import spec_augment, visualization_spectrogram
from spec_aug.spec_augment_pytorch import SpecAugment


# Beatport Dataset
class SimpleBPDataset(Dataset):
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
        random_slice=True,
        stems=["other"], #["vocals", "bass", "drums", "other"], # VBDO
        top_k=1, #2,
    ):
        # Load split files from txt file
        with open(f"info/{split}_by_song_name_4secs.txt", "r") as f:
            bp_listdir = [line.strip() for line in f.readlines()]

        self.stems = stems
        self.data_path_list = [
            os.path.join(data_dir, folder, f"{stem}.npy")
            for folder in bp_listdir
                for stem in stems
        ]

        self.sample_rate = sample_rate
        self.segment_second = segment_second
        self.window_size = window_size
        self.hop_length = hop_length
        self.piece_length = int(piece_second * sample_rate)
        self.duration = int(segment_second * sample_rate)
        self.split = split
        self.random_slice = random_slice
        self.top_k = top_k

    def __len__(self): #""" Total we got 175698 files * 4 tracks """
        return len(self.data_path_list)
    
    
    
    def cal_high_energy_crop(self, x):
        # TODO: Process it before loading
        """Return top-k high-energy segments (non-overlapping)"""
        
        # Frame RMS energy (center=False to align frame with start)
        rms = librosa.feature.rms(
            y=x, center=False, 
            frame_length=self.window_size, hop_length=self.hop_length
        )[0]
        
        # Find top-k indices with highest energy
        top_indices = np.argsort(rms)[-self.top_k:][::-1][0]
        return top_indices * self.hop_length

 
    def random_crop(self, x):
        """ Random crop for given segmented seconds """
        # TODO: Sound Event Detection / voice activity detection (VAD)
        max_idx = self.piece_length - self.duration
        idx = random.randint(0, max_idx)
        return x[idx : idx+self.duration]


    def __getitem__(self, idx):
        # Load audio data from .npy
        path = self.data_path_list[idx]
        x = np.load(path, mmap_mode='r')

        # Random Crop
        if self.random_slice:
            x_i, x_j = self.random_crop(x), self.random_crop(x)
        else:
            half = int(x.shape[0] // 2)
            segment1 = self.cal_high_energy_crop(x[:half])
            segment2 = self.cal_high_energy_crop(x[half:self.piece_length-self.duration])
            return segment1, half+segment2, path.split("/")[-2]

        # Faster conversion if .npy is already float32
        # return torch.from_numpy(x_i), torch.from_numpy(x_j), path
        return torch.from_numpy(x_i.copy()), torch.from_numpy(x_j.copy()), path




class BPDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        segment_second,
        data_dir,
        augment_func,
        piece_second=3,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        split="train",
        melspec_transform=False,
        data_augmentation=True,
        random_slice=False,
        stems=["other"], #["vocals", "bass", "drums", "other"], # VBDO
        fmax=8000,
        img_size=256,
        img_mean=0,
        img_std=0,
        clap_use=False,
    ):
        # Load split files from txt file
        with open(f"info/{split}_bp_8secs.txt", "r") as f:
            bp_listdir = [line.strip() for line in f.readlines()]

        self.stems = stems
        self.data_path_list = [
            os.path.join(data_dir, folder, f"{stem}.npy")
            for folder in bp_listdir
                for stem in stems
        ]

        self.sample_rate = sample_rate
        self.segment_second = segment_second
        self.duration = sample_rate * piece_second # 4 seconds for each piece
        self.augment_func = augment_func # CLARTransform
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.split = split
        self.melspec_transform = melspec_transform
        self.data_augmentation = data_augmentation
        self.random_slice = random_slice
        self.clap_use = clap_use
        
        # Mel-spec transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_max=fmax,
        )
        self.db_transform = T.AmplitudeToDB(
            stype="power"
        )
        self.resizer = transforms.Resize((img_size, img_size))
        self.img_mean = img_mean
        self.img_std = img_std
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5), 
            # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.4), # S3T settings
            # PitchShift(min_semitones=-4, max_semitones=4, p=0.4), # S3T settings
            # Shift(p=1),
        ])
    
    
    def __len__(self): #""" Total we got 175698 files * 4 tracks """
        return len(self.data_path_list)
    
    
    def mel_spec_transform(self, x):
        x = x.float()
        mel_spec = self.mel_transform(x).float()
        return self.db_transform(mel_spec)
        
    
    
    def data_pipeline(self, x):
        x = torch.tensor(x)
        x = self.mel_spec_transform(x).unsqueeze(0)
        x = self.resizer(x) # transform to 1, img_size, img_size
        return (x - self.img_mean) / self.img_std


    def random_crop(self, x):
        """ Random crop for 4 seconds """
        max_idx = int(x.shape[0]) - self.duration
        idx = random.randint(0, max_idx)
        return x[idx:idx+self.duration]


    def __getitem__(self, idx):
        """ 
            1. Mel-Spectrogram Transformation
            2. Resize
            3. Normalization
            4. Data Augmentation
        """
        # Load audio data
        path = self.data_path_list[idx]
        x = np.load(path)
        
        
        # Random Crop
        if self.random_slice:
            x_i, x_j = self.random_crop(x), self.random_crop(x)
        else:
            x_i, x_j = x[:self.duration], x[self.duration:]
        
        if self.clap_use:
            x_i_audio = torch.tensor(x_i)#.clone().detach()
            x_j_audio = torch.tensor(x_j)#.clone().detach()
        else:
            x_i_audio = None
            x_j_audio = None
        
        # Augmentation
        if self.data_augmentation:
            x_i, x_j = self.augment(x_i, sample_rate=self.sample_rate), \
                self.augment(x_j, sample_rate=self.sample_rate)
        
        # Mel-spectrogram transformation and Melspec's Augmentation
        x_i, x_j = self.data_pipeline(x_i), self.data_pipeline(x_j)
        
        return x_i.float(), x_j.float(), x_i_audio, x_j_audio, path



class NewBPDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        segment_second,
        data_dir,
        piece_second=3,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        split="train",
        melspec_transform=False,
        data_augmentation=True,
        random_slice=False,
        stems=["other"], #["vocals", "bass", "drums", "other"], # VBDO
        fmax=8000,
        img_size=256,
        img_mean=0,
        img_std=0,
    ):
        # Load split files from txt file
        with open(f"info/{split}_bp_8secs.txt", "r") as f:
            bp_listdir = [line.strip() for line in f.readlines()]

        self.stems = stems
        self.data_path_list = [
            os.path.join(data_dir, folder, f"{stem}.npy")
            for folder in bp_listdir
                for stem in stems
        ]

        self.sample_rate = sample_rate
        self.segment_second = segment_second
        self.duration = sample_rate * piece_second # 4 seconds for each piece
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.split = split
        self.melspec_transform = melspec_transform
        self.data_augmentation = data_augmentation
        self.random_slice = random_slice
        self.fmax = fmax
        
        self.resizer = transforms.Resize((img_size, img_size))
        self.img_size = img_size
        self.img_mean = img_mean
        self.img_std = img_std
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5), 
            # TimeMask(min_band_part=0.01, max_band_part=0.125, fade=True, p=0.5),
            Shift(p=0.5), # Time shift
            # TimeStretch(min_rate=0.8, max_rate=1.25, leave_length_unchanged=True, p=0.4),
            # PitchShift(min_semitones=-4, max_semitones=4, p=0.4), # S3T settings
            # Shift(p=1),
        ])
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_max=self.fmax,
        )

        self.db_transform = T.AmplitudeToDB()
    
    
    def __len__(self): #""" Total we got 175698 files * 4 tracks """
        return len(self.data_path_list)
    
    
    def mel_spec_transform(self, x):
        x = torch.tensor(x)
        x = self.mel_transform(x)
        x = self.db_transform(x)
        return x
    
    def data_pipeline(self, x):
        x = self.mel_spec_transform(x).unsqueeze(0)
       
        # Added mel spec augmentation
        time_warping_para = random.randint(1, 10) # (0, 10)
        freq_mask_num = random.randint(1, 3) # (1, 5)
        time_mask_num = random.randint(1, 5) # (1, 10)
        freq_masking_para = random.randint(5, 15) # (5, 30)
        time_masking_para = random.randint(5, 15) # (5, 30)
        p1, p2 = 0.4, 0.5
        
        x = spec_augment(
            x, 
            time_warping_para=time_warping_para, 
            frequency_masking_para=freq_masking_para,
            time_masking_para=time_masking_para, 
            frequency_mask_num=freq_mask_num, 
            time_mask_num=time_mask_num,
            p1=p1,
            p2=p2,
        )
        
        x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False).squeeze(0)
        return (x - self.img_mean) / self.img_std
    

    def random_crop(self, x):
        """ Random crop for 4 seconds """
        max_idx = int(x.shape[0]) - self.duration
        idx = random.randint(0, max_idx)
        return x[idx:idx+self.duration]


    def __getitem__(self, idx):
        """ 
            1. Mel-Spectrogram Transformation
            2. Resize
            3. Normalization
            4. Data Augmentation
        """
        # Load audio data
        path = self.data_path_list[idx]
        x = np.load(path)
        
        # Random Crop
        if self.random_slice:
            x_i, x_j = self.random_crop(x), self.random_crop(x)
        else:
            x_i, x_j = x[:self.duration], x[self.duration:]
        
        # Augmentation
        if self.data_augmentation:
            x_i, x_j = self.augment(x_i, sample_rate=self.sample_rate), \
                self.augment(x_j, sample_rate=self.sample_rate)
        return torch.tensor(x_i).float(), torch.tensor(x_j).float(), path
    
    
        # # Mel-spectrogram transformation and Melspec's Augmentation
        # x_i, x_j = self.data_pipeline(x_i), self.data_pipeline(x_j)
        
        # return x_i.float(), x_j.float(), path



class MixedBPDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        segment_second,
        data_dir,
        augment_func,
        piece_second=3,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        split="train",
        melspec_transform=False,
        data_augmentation=True,
        random_slice=False,
        stems=["other"], #["vocals", "bass", "drums", "other"], # VBDO
        fmax=8000,
        img_size=256,
        img_mean=0,
        img_std=0,
    ):
        # Load split files from txt file
        with open(f"info/{split}_bp_8secs.txt", "r") as f:
            bp_listdir1 = [line.strip() for line in f.readlines()]
        
        with open(f"info/{split}_bp_verse_8secs.txt", "r") as f:
            bp_listdir2 = [line.strip() for line in f.readlines()]
        
        
        data_dir2 = data_dir.replace("chorus", "verse")
        self.stems = stems
        self.data_path_list = [
            os.path.join(data_dir, folder, f"{stem}.npy")
            for folder in bp_listdir1
                for stem in stems
        ] + [
            os.path.join(data_dir2, folder, f"vocals.npy")
                for folder in bp_listdir2
        ]

        self.sample_rate = sample_rate
        self.segment_second = segment_second
        self.duration = sample_rate * piece_second # 4 seconds for each piece
        self.augment_func = augment_func # CLARTransform
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.split = split
        self.melspec_transform = melspec_transform
        self.data_augmentation = data_augmentation
        self.random_slice = random_slice
        self.fmax = fmax
        
        self.resizer = transforms.Resize((img_size, img_size))
        self.img_mean = img_mean
        self.img_std = img_std
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5), 
            # TimeMask(min_band_part=0.01, max_band_part=0.125, fade=True, p=0.5),
            Shift(p=0.5), # Time shift
            # TimeStretch(min_rate=0.8, max_rate=1.25, leave_length_unchanged=True, p=0.4),
            # PitchShift(min_semitones=-4, max_semitones=4, p=0.4), # S3T settings
            # Shift(p=1),
        ])
        
    
    def __len__(self): #""" Total we got 175698 files * 4 tracks """
        return len(self.data_path_list)
    
    
    def mel_plotting(self, mel, title, save_path):
        if mel.ndim == 3:
            mel = np.squeeze(mel, axis=0)
        mel = mel.numpy()
        fig, ax = plt.subplots()
        img = librosa.display.specshow(
            mel, x_axis='time', y_axis='mel',
            sr=self.sample_rate, fmax=self.fmax, ax=ax
        )
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title=title)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')  # Save as PNG

    
    def mel_spec_transform(self, x):
        x = librosa.feature.melspectrogram(
            y=x, 
            sr=self.sample_rate, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmax=self.fmax,
        )
        x = librosa.power_to_db(np.abs(x))
        return torch.tensor(x)
        # x = x.float()
        # mel_spec = self.mel_transform(x).float()
        # return self.db_transform(mel_spec)
        
    
    
    def data_pipeline(self, x):
        x = self.mel_spec_transform(x).unsqueeze(0)
        
        # Added mel spec augmentation
        time_warping_para = random.randint(1, 10) # (0, 10)
        freq_mask_num = random.randint(1, 3) # (1, 5)
        time_mask_num = random.randint(1, 5) # (1, 10)
        freq_masking_para = random.randint(5, 15) # (5, 30)
        time_masking_para = random.randint(5, 15) # (5, 30)
        p1, p2 = 0.4, 0.5
        
        # self.mel_plotting(x, "original mel spectrogram", "visualization/original_mel_spec.png")
        
        x = spec_augment(
            x, 
            time_warping_para=time_warping_para, 
            frequency_masking_para=freq_masking_para,
            time_masking_para=time_masking_para, 
            frequency_mask_num=freq_mask_num, 
            time_mask_num=time_mask_num,
            p1=p1,
            p2=p2,
        )
        # self.mel_plotting(x, "augmented mel spectrogram", "visualization/augmented_mel_spec.png")
        
        x = self.resizer(x) # transform to 1, img_size, img_size
        return (x - self.img_mean) / self.img_std


    def random_crop(self, x):
        """ Random crop for 4 seconds """
        max_idx = int(x.shape[0]) - self.duration
        idx = random.randint(0, max_idx)
        return x[idx:idx+self.duration]


    def __getitem__(self, idx):
        """ 
            1. Mel-Spectrogram Transformation
            2. Resize
            3. Normalization
            4. Data Augmentation
        """
        # Load audio data
        path = self.data_path_list[idx]
        x = np.load(path)
        
        
        # Random Crop
        if self.random_slice:
            x_i, x_j = self.random_crop(x), self.random_crop(x)
        else:
            x_i, x_j = x[:self.duration], x[self.duration:]
            
        # Data Augmentation for audio waveform
        if self.data_augmentation:
            x_i, x_j = self.augment(x_i, sample_rate=self.sample_rate), \
                self.augment(x_j, sample_rate=self.sample_rate)
        
        # Mel spectrogram transformation and Melspec's Augmentation
        x_i, x_j = self.data_pipeline(x_i), self.data_pipeline(x_j)
        
        return x_i.float(), x_j.float(), path
        




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




class MixedBPDataModule(LightningDataModule):
    def __init__(self, args):
        super(MixedBPDataModule, self).__init__()
        self.args = args

    def setup(self, stage=None):
        self.train_dataset = MixedBPDataset(
            sample_rate=self.args.sample_rate, 
            segment_second=self.args.segment_second, 
            piece_second=self.args.piece_second,
            data_dir=self.args.data_dir,
            augment_func=CLARTransform(
                sample_rate=self.args.sample_rate,
                duration=int(self.args.piece_second),
            ),
            n_mels=self.args.n_mels,
            n_fft=self.args.n_fft,
            hop_length=self.args.hop_length,
            split="train",
            melspec_transform=self.args.melspec_transform,
            data_augmentation=self.args.data_augmentation,
            random_slice=self.args.random_slice,
            stems=['other'],
            fmax=self.args.fmax,
            img_size=self.args.img_size,
            img_mean=self.args.img_mean,
            img_std=self.args.img_std,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=self.args.workers, 
            pin_memory=self.args.pin_memory, 
            drop_last=self.args.drop_last,
            persistent_workers=self.args.persistent_workers,  # Keep workers alive to reduce loading overhead
            prefetch_factor=self.args.prefetch_factor,
        )

    @property
    def num_samples(self) -> int:
        self.setup(stage='fit')
        return len(self.train_dataset)




class Transform_Pipeline(nn.Module):
    def __init__(
        self, 
        sample_rate, 
        n_fft, 
        hop_length, 
        n_mels, 
        fmax, 
        img_size, 
        img_mean, 
        img_std, 
        device=torch.device("cuda"),
        p_time_warp=0.4,
        p_mask=0.5,
    ):
        super(Transform_Pipeline, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmax = fmax
        self.img_size = img_size
        self.img_mean = img_mean
        self.img_std = img_std
        self.device = device
        self.p_time_warp = p_time_warp
        self.p_mask = p_mask
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_max=self.fmax,
        ).to(device)

        self.db_transform = T.AmplitudeToDB().to(device)
        

    def mel_spec_transform(self, x):
        x = self.mel_transform(x)
        x = self.db_transform(x)
        return x
    
    def __call__(self, x):
        x = self.mel_spec_transform(x) #.unsqueeze(0)
       
        # Added mel spec augmentation
        time_warping_para = random.randint(1, 10) # (0, 10)
        freq_mask_num = random.randint(1, 3) # (1, 5)
        time_mask_num = random.randint(1, 5) # (1, 10)
        freq_masking_para = random.randint(5, 15) # (5, 30)
        time_masking_para = random.randint(5, 15) # (5, 30)
        
        spec_augment = SpecAugment(
            time_warping_para=time_warping_para,
            frequency_masking_para=freq_mask_num,
            time_masking_para=time_mask_num,
            frequency_mask_num=freq_masking_para,
            time_mask_num=time_masking_para,
            p_time_warp=self.p_time_warp,
            p_mask=self.p_mask,
        ).to(self.device)
        
        x = spec_augment(x)
        x = x.unsqueeze(1)
        
        x = F.interpolate(
            x, 
            size=(self.img_size, self.img_size), 
            mode='bilinear', 
            align_corners=False
        )  # x shape: [B, 1, img_size, img_size]
        return (x - self.img_mean) / self.img_std
    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Simsiam_BP")

    config = yaml_config_hook("config/ssbp_convnext.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    train_dataset = SimpleBPDataset(
        sample_rate=args.sample_rate,
        data_dir=args.data_dir,
        piece_second=args.piece_second,
        segment_second=args.segment_second,
        window_size=args.window_size,
        hop_length=args.hop_length,
        random_slice=False, #args.random_slice,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16, #args.batch_size,
        shuffle=True,
        num_workers=args.workers,
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



    segment_dict = dict()
    for start1, start2, song_name in tqdm(train_loader):
        segment_dict[song_name] = [start1, start2]
    
    
    with open("info/segment_dict_train.pkl", "wb") as f:
        pickle.dump(segment_dict, f)
        
        
    # Later loading
    with open("data.pkl", "rb") as f:
        my_dict = pickle.load(f)
    
    # X = []
    # for (x_i, x_j, _) in tqdm(train_loader):
    #     x_i = x_i.to(device)
    #     x_j = x_j.to(device)

    #     lms_i = (to_spec(x_i) + torch.finfo().eps).log().unsqueeze(1)
    #     lms_j = (to_spec(x_j) + torch.finfo().eps).log().unsqueeze(1)
    #     X.extend([x for x in lms_i.detach().cpu().numpy()])
    #     X.extend([x for x in lms_j.detach().cpu().numpy()])
        

    # X = np.stack(X)
    # norm_stats = np.array([X.mean(), X.std()])
    # print(norm_stats)



    # for i in range(10):
    #     ts1, ts2, _ = train_dataset[i]
    #     print(ts1.shape, ts2.shape)

    # # Experiment 1: Mel-spectrogram
    # mel_transform = T.MelSpectrogram(
    #     sample_rate=args.sample_rate,
    #     n_mels=args.n_mels,
    #     n_fft=args.n_fft,
    #     hop_length=args.hop_length,
    #     f_max=args.fmax,
    # )
    # db_transform = T.AmplitudeToDB(stype="power")
    # i = 0
    # xi, xj = train_dataset[i][0], train_dataset[i][1]
    
    # # original dataset
    # x = np.load(train_dataset.data_path_list[i])
    
    # torchaudio.save("visualization/x.wav", torch.tensor(x).clone().detach().unsqueeze(0), args.sample_rate)
    # torchaudio.save("visualization/xi.wav", xi.clone().detach().unsqueeze(0), args.sample_rate)
    # torchaudio.save("visualization/xj.wav", xj.clone().detach().unsqueeze(0), args.sample_rate)

    
    # x1 = db_transform(mel_transform(xi))
    # x2 = db_transform(mel_transform(xj))
    # print(x1.shape, x2.shape)
    
    # plot_spec_and_save(x1, "mel_spectrogram_x1_256.png", sr=args.sample_rate)
    # plot_spec_and_save(x2, "mel_spectrogram_x2_256.png", sr=args.sample_rate)
    

    # # Experiment 2: Resize spectrogram
    # res_x1 = resize_spec(x1, target_size=(256, 256))
    # res_x2 = resize_spec(x2, target_size=(256, 256))
    
    # plot_spec_and_save(res_x1, "resized_x1_256.png", sr=args.sample_rate)
    # plot_spec_and_save(res_x2, "resized_x2_256.png", sr=args.sample_rate)


    # for i in range(10):
    #     print(train_dataset[i][0].shape, train_dataset[i][1].shape)
    
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=4, shuffle=True,
    #     num_workers=args.workers, pin_memory=True, drop_last=True)
    
    # for ds in tqdm(train_loader):
    #     print(ds[0].shape, ds[1].shape)
    #     pass
    