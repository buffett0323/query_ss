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




# Beatport Dataset
class BPDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        duration,
        data_dir,
        split="train",
        need_transform=True,
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
        
        self.transform = AudioFXAugmentation( #CLARTransform(
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
        # transform1, transform2 = random.sample(self.transforms, 2)
        # x1 = transform1(x1)
        # x2 = transform2(x2)
        
        # Apply augmentations
        x1 = self.add_noise_transform(x1)
        x1 = self.time_stretch_transform(x1)
        x1 = self.time_masking_transform(x1)
        
        x2 = self.add_noise_transform(x2)
        x2 = self.time_stretch_transform(x2)
        x2 = self.time_masking_transform(x2)
        
        return x1, x2




class AudioFXAugmentation(nn.Module):
    def __init__(
        self, 
        sample_rate,
        duration,
        n_mels=128,
    ):
        super(AudioFXAugmentation, self).__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        
    def sample_tau(self):
        """
        Sample τ from the given probability distribution: τ ∼ 1/(τ log(1.5/0.75))
        within the range [0.75, 1.5].
        """
        def pdf(tau):
            return 1 / (tau * np.log(1.5 / 0.75))
        
        tau_range = np.linspace(0.75, 1.5, 1000)
        probs = pdf(tau_range)
        probs /= probs.sum()  # Normalize to create a proper probability distribution
        return np.random.choice(tau_range, p=probs)
    

    def time_stretch_augmentation(self, x):
        """
        Apply Time Stretching (TS) Augmentation following the TSPS methodology.
        
        Steps:
        1. Load an audio file and ensure it is at least `context_length` seconds long.
        2. Randomly crop a 4.5s segment (if necessary).
        3. Sample a time-stretch factor τ from the given probability distribution.
        4. Apply time stretching using cubic spline interpolation.
        5. Truncate the resulting signal to `target_length` seconds.
        """
        target_length = 3.0
        
        # Sample τ from the given probability distribution and apply Time Stretching (TS)
        tau = self.sample_tau()
        x = librosa.effects.time_stretch(x, rate=tau)

        # Truncate to `target_length` seconds
        target_samples = int(self.sample_rate * target_length)
        if len(x) > target_samples:
            x = x[:target_samples]
        else:
            x = np.pad(x, (0, target_samples - len(x)), mode='constant')

        return x #torch.tensor(x).float().unsqueeze(0)  # Add batch dimension

    
    def sample_mu(self):
        """
        Sample μ from the given probability distribution: μ ∼ 1/(μ log(1.335/0.749))
        within the range [0.749, 1.335].
        """
        def pdf(mu):
            return 1 / (mu * np.log(1.335 / 0.749))
        
        mu_range = np.linspace(0.749, 1.335, 1000)
        probs = pdf(mu_range)
        probs /= probs.sum()  # Normalize to create a proper probability distribution
        return np.random.choice(mu_range, p=probs)


    def pitch_shift_augmentation(self, x):
        """
        Apply Pitch Shifting (PS) Augmentation following the TSPS methodology.

        Steps:
        1. Compute the mel spectrogram of the input audio.
        2. Sample a pitch shift factor μ.
        3. Apply cubic spline interpolation on the frequency axis.
        4. If μ < 1.0, zero out frequency bins above μ * max frequency.
        """

        # Compute Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=x, 
            sr=self.sample_rate, 
            n_mels=self.n_mels
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)  # Convert to dB

        # Sample μ
        mu = self.sample_mu()

        # Compute frequency bin warping
        U = self.n_mels
        SU = U / np.log10(1 + self.sample_rate / 700)  # Compute scaling factor
        mel_bins = np.linspace(0, U - 1, U)  # Original mel bins
        warped_bins = SU * np.log10(1 + mu * (10 ** (mel_bins / SU) - 1))  # Apply pitch shift transformation

        # Apply cubic spline interpolation
        interpolator = scipy.interpolate.interp1d(mel_bins, mel_spectrogram_db, axis=0, kind='cubic', fill_value="extrapolate")
        shifted_spectrogram = interpolator(warped_bins)

        # Zero out bins if μ < 1.0
        if mu < 1.0:
            cutoff_bin = int(mu * U)
            shifted_spectrogram[cutoff_bin:] = -80.0  # Set to silence (approximate dB floor)

        return shifted_spectrogram
    
    
    def butterworth_filter(self, filter_type="low", cutoff_freq=None):
        """
        Apply a third-order Butterworth filter (lowpass/highpass).

        Parameters:
            sr (int): Sampling rate.
            n_mels (int): Number of mel bins.
            filter_type (str): "low" or "high".
            cutoff_freq (float): Cutoff frequency in Hz.

        Returns:
            np.ndarray: Butterworth filter response.
        """
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff_freq / nyquist  # Normalize w.r.t Nyquist frequency
        b, a = scipy.signal.butter(N=3, Wn=normal_cutoff, btype=filter_type, analog=False)
        return b, a


    def apply_equalization_filter(self, x):
        """
        Apply Equalization Augmentation (EQ) using Butterworth filters.

        Steps:
        1. Randomly select a filter type (lowpass, highpass, or none).
        2. Design a Butterworth filter based on sampled cutoff frequency.
        3. Apply the filter to the spectrogram.
        4. Add the log-scaled filter response to the original spectrogram.

        Parameters:
            mel_spectrogram (np.ndarray): Input mel spectrogram.
            sr (int): Sampling rate.
            n_mels (int): Number of mel bins.

        Returns:
            np.ndarray: Augmented spectrogram.
            str: Applied filter type.
            float: Chosen cutoff frequency.
        """
        # Choose filter type with equal probability
        filter_choice = np.random.choice(["lowpass", "highpass", "none"], p=[1/3, 1/3, 1/3])

        if filter_choice == "none":
            return x

        # Sample cutoff frequency
        if filter_choice == "lowpass":
            cutoff_freq = np.random.uniform(2200, 4000)  # Hz
        else:  # Highpass
            cutoff_freq = np.random.uniform(200, 1200)  # Hz

        # Get Butterworth filter and Apply filter along the mel bins axis
        b, a = self.butterworth_filter(filter_type=filter_choice, cutoff_freq=cutoff_freq)
        filtered_spectrogram = scipy.signal.filtfilt(b, a, x, axis=0)

        # Add the log-transformed filter response
        return x + np.log10(np.abs(filtered_spectrogram) + 1e-6)



    def __call__(self, x1, x2):        
        # Apply augmentations
        x1 = self.time_stretch_augmentation(x1)
        x1 = self.pitch_shift_augmentation(x1)
        x1 = self.apply_equalization_filter(x1)
        
        x2 = self.time_stretch_augmentation(x2)
        x2 = self.pitch_shift_augmentation(x2)
        x2 = self.apply_equalization_filter(x2)
        return x1, x2



if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="SimCLR_BP")

    config = yaml_config_hook("config/ssbp_6secs.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    
    # ds = SlakhDataset(args.sample_rate, args.segment_second, split="train")
    # for i in range(10):
    #     print(ds[i][2])
    
    
    train_dataset = BPDataset(args.sample_rate, args.segment_second, data_dir=args.data_dir, split="train")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    
    for ds in train_loader:
        print(ds[0].shape, ds[1].shape) # 256, 128, 94
    
    
    # for tr in tqdm(dm.train_dataloader()):
    #     pass; #print(tr[0].shape)
