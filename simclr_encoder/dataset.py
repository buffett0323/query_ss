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
from torchaudio.functional import pitch_shift
from utils import yaml_config_hook

class NSynthDataset(Dataset):
    def __init__(
        self, 
        data_dir="/mnt/gestalt/home/ddmanddman/nsynth_dataset/",
        split="train",
    ):
        self.data_path_list = [
            os.path.join(data_dir, f"nsynth-{split}", "npy", i)
            for i in os.listdir(os.path.join(data_dir, f"nsynth-{split}", "npy"))
        ]
        

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        path = self.data_path_list[idx]
        return np.load(path)


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
            )
            
            self.val_ds = NSynthDataset(
                data_dir=self.data_dir,
                split="valid",
            )
        
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = NSynthDataset(
                data_dir=self.data_dir,
                split="test",
            )
            
            
    def train_dataloader(self):
        """The train dataloader."""
        return self._data_loader(
            self.train_ds,
            shuffle=True)

    def val_dataloader(self):
        """The val dataloader."""
        return self._data_loader(
            self.val_ds,
            shuffle=False)

    def test_dataloader(self):
        """The test dataloader."""
        return self._data_loader(
            self.test_ds,
            shuffle=False)
        
    
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
    parser = argparse.ArgumentParser(description="SimCLR")

    config = yaml_config_hook("config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    dm = NSynthDataModule(
        args=args,
    )
    dm.setup()
    
    for i in dm.train_dataloader():
        print(i.shape); break