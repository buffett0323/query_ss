# Using original track, without .dbfss
import os
import random
import shutil
import torch
import torchaudio
import argparse
import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram
from pytorch_lightning import LightningDataModule
from typing import Optional
from tqdm import tqdm
from mutagen.mp3 import MP3
from scipy.signal import fade
from torchaudio.functional import pitch_shift







class NSynthDataset(Dataset):
    def __init__(
        self, 
        data_dir="/mnt/gestalt/home/ddmanddman/nsynth_dataset/",
        split="train",
    ):
        self.data_path_list = os.listdir(os.path.join(data_dir, f"nsynth-{split}", "npy"))
        

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        path = self.data_path_list[idx]
        return np.load(path)


class CLARTransform(nn.Module):
    def __init__(self, sample_rate=16000):
        super(CLARTransform, self).__init__()
        self.sample_rate = sample_rate
        self.transforms = random.choice([
            self.pitch_shift_transform,
            self.fade_in_out_transform,
            self.noise_injection_transform,
            self.time_masking_transform,
            self.time_shift_transform,
            self.time_stretch_transform,
        ])
        
        
    def pitch_shift_transform(self, audio):
        # Randomly pick a shift in the range [-15, 15]
        shift = random.uniform(-15, 15)
        return pitch_shift(audio, self.sample_rate, shift)

    def fade_in_out_transform(self, audio):
        fade_type = random.choice(['linear', 'logarithmic', 'exponential'])
        fade_size = random.uniform(0, 0.5) * len(audio)
        fade_size = int(fade_size)
        
        if fade_type == 'linear':
            fade_in = np.linspace(0, 1, fade_size)
            fade_out = np.linspace(1, 0, fade_size)
        elif fade_type == 'logarithmic':
            fade_in = np.logspace(-1, 0, fade_size, base=10)
            fade_in = (fade_in - fade_in.min()) / (fade_in.max() - fade_in.min())
            fade_out = fade_in[::-1]
        else:  # exponential
            fade_in = np.linspace(0, 1, fade_size)**2
            fade_out = fade_in[::-1]

        audio[:fade_size] *= fade_in
        audio[-fade_size:] *= fade_out
        return audio

    def noise_injection_transform(self, audio):
        noise_type = random.choice(['white', 'brown', 'pink'])
        snr = random.uniform(0.5, 1.5)  # Signal-to-noise ratio
        
        if noise_type == 'white':
            noise = np.random.normal(0, 1, len(audio))
        elif noise_type == 'brown':
            noise = np.cumsum(np.random.normal(0, 1, len(audio)))
            noise = noise / np.max(np.abs(noise))
        else:  # pink noise
            freqs = np.fft.rfftfreq(len(audio))
            noise = np.fft.irfft(np.random.randn(len(freqs)) / (freqs + 1e-6))

        noise = noise / np.max(np.abs(noise))
        audio = audio + noise / snr
        return np.clip(audio, -1, 1)

    def time_masking_transform(self, audio):
        max_segment = int(len(audio) / 8)
        mask_size = random.randint(1, max_segment)
        start = random.randint(0, len(audio) - mask_size)
        audio[start:start + mask_size] = 0
        return audio

    def time_shift_transform(self, audio):
        shift = random.randint(-len(audio) // 2, len(audio) // 2)
        return np.roll(audio, shift)

    def time_stretch_transform(self, audio):
        rate = random.uniform(0.5, 1.5)
        stretched = torchaudio.functional.time_stretch(torch.tensor(audio).unsqueeze(0), self.sample_rate, rate)
        if rate > 1:
            # Crop to original size
            stretched = stretched[:len(audio)]
        else:
            # Pad to original size
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)), mode='constant')
        return stretched

    def __call__(self, x):        
        # Apply random augmentations
        transform1, transform2 = random.sample(self.transforms, 2)
        x1 = transform1(x)
        x2 = transform2(x)
        return x1, x2




if __name__ == "__main__":
    db = NSynthDataset()
    print(len(db))