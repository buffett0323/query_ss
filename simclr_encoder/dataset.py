# Using original track, without .dbfss
import os
import random
import math
import torch
import torchaudio
import librosa
import argparse
import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram
from pytorch_lightning import LightningDataModule
from typing import Optional
from librosa import effects
from tqdm import tqdm
from torchaudio.functional import pitch_shift
from utils import yaml_config_hook

class NSynthDataset(Dataset):
    def __init__(
        self, 
        data_dir="/mnt/gestalt/home/ddmanddman/nsynth_dataset/",
        split="train",
        need_transform=True,
    ):
        self.data_path_list = [
            os.path.join(data_dir, f"nsynth-{split}", "npy", i)
            for i in os.listdir(os.path.join(data_dir, f"nsynth-{split}", "npy"))
        ]
        self.transform = CLARTransform()
        self.need_transform = need_transform
        self.split = split
        
        
    def get_spec_features(self, x, sr=16000):
        # Convert audio to mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=x, sr=sr, hop_length=128)
        return librosa.power_to_db(mel_spec)


    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        path = self.data_path_list[idx]
        x = np.load(path).squeeze(0)
        x_i, x_j = x[:x.shape[0]//2], x[x.shape[0]//2:]
        
        if self.need_transform:
            x_i, x_j = self.transform(x_i, x_j)
            
        if self.split == "test":
            return torch.tensor(x, dtype=torch.float32), path
        return torch.tensor(x_i, dtype=torch.float32), torch.tensor(x_j, dtype=torch.float32)


class NSynthDataModule(LightningDataModule):
    def __init__(
        self,
        args,
        data_dir="/mnt/gestalt/home/ddmanddman/nsynth_dataset/", 
    ):
        super(NSynthDataModule, self).__init__()
        self.args = args
        self.data_dir = data_dir
        self.pin_memory = True
        self.drop_last = False
        
    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_ds = NSynthDataset(
                data_dir=self.data_dir,
                split="train",
                need_transform=self.args.need_clar_transform,
            )
            
            self.val_ds = NSynthDataset(
                data_dir=self.data_dir,
                split="valid",
                need_transform=self.args.need_clar_transform,
            )
        
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = NSynthDataset(
                data_dir=self.data_dir,
                split="test",
                need_transform=self.args.need_clar_transform,
            )
            
            
    def train_dataloader(self):
        """The train dataloader."""
        return self._data_loader(
            self.train_ds,
            shuffle=True
        )

    def val_dataloader(self):
        """The val dataloader."""
        return self._data_loader(
            self.val_ds,
            shuffle=False
        )

    def test_dataloader(self):
        """The test dataloader."""
        return self._data_loader(
            self.test_ds,
            shuffle=False
        )
        
    
    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
    
    @property
    def num_samples(self) -> int:
        self.setup(stage = 'fit')
        return len(self.train_ds)



class CLARTransform(nn.Module):
    def __init__(self, sample_rate=16000):
        super(CLARTransform, self).__init__()
        self.sample_rate = sample_rate
        self.length = 2
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
        x = librosa.resample(x, orig_sr=x.shape[0] / self.length, target_sr=self.sample_rate)
        if x.shape[0] > (self.sample_rate * self.length):
            return x[:(self.sample_rate * self.length)]
        return np.pad(x, [0, (self.sample_rate * self.length) - x.shape[0]])


    def __call__(self, x1, x2):        
        # Apply random augmentations
        transform1, transform2 = random.sample(self.transforms, 2)
        x1 = transform1(x1)
        x2 = transform2(x2)
        return x1, x2




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")

    config = yaml_config_hook("config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    dm = NSynthDataModule(
        args=args,
    )
    dm.setup()
    
    ds = NSynthDataset()
    for i in range(30):
        x, y = ds[i]
        print(x.shape, y.shape)