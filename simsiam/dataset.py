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
from utils import yaml_config_hook, plot_spec_and_save, resize_spec

# Beatport Dataset
class BPDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        segment_second,
        data_dir,
        augment_func,
        piece_second=3,
        n_fft=1024,
        hop_length=320,
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
        
        
        self.sample_rate = sample_rate
        self.segment_second = segment_second
        self.duration = sample_rate * piece_second # 3 seconds for each piece
        self.augment_func = augment_func # CLARTransform
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.split = split
        self.melspec_transform = melspec_transform
        self.data_augmentation = data_augmentation
        self.random_slice = random_slice
    
    
    def random_sample(self, audio_length):
        # TODO: minimum repetivie 1.5 seconds (x), maximum 5 seconds (v)
        f1 = random.randint(0, audio_length - self.duration)
        
        # Define valid range for f2 relative to f1
        min_offset = self.sample_rate * 1.5
        max_offset = self.sample_rate * 5

        # Ensure f2 stays within bounds
        min_f2 = max(0, f1 - max_offset)
        max_f2 = min(audio_length - self.duration, f1 + max_offset)

        # Directly sample f2 within valid range
        f2 = random.randint(min_f2, max_f2)
        return f1, f2
    
    
    def get_label(self, folder, stem):
        style = self.class_dict[folder.split('_')[0]]
        return self.labels.index(style) * len(self.stems) + self.stems.index(stem)

    
    def mel_spec_transform(self, x):
        mel_spec = librosa.feature.melspectrogram(
            y=x, sr=self.sample_rate, n_mels=self.n_mels, 
            n_fft=self.n_fft, hop_length=self.hop_length, fmax=8000,
        )    
        return librosa.power_to_db(mel_spec, ref=np.max)
    
        
    def __len__(self): #""" Total we got 175698 files * 4 tracks """
        return len(self.data_path_list)
    
    
    def __getitem__(self, idx):
        path = self.data_path_list[idx]
        
        # Read data and segment
        x = np.load(path)
        audio_length = x.shape[0]

        # Random Crop for 3 seconds
        if self.random_slice:
            f1, f2 = self.random_sample(audio_length)
            x_i, x_j = x[f1: f1+self.duration], x[f2: f2+self.duration]
        else:
            x_i, x_j = x[:int(audio_length/2)], x[int(audio_length/2):]
        
        
        # Augmentation
        if self.data_augmentation:
            x_i, x_j = self.augment_func(x_i, x_j)
            
        # Mel-spectrogram and add channel
        if self.melspec_transform:
            x_i, x_j = self.mel_spec_transform(x_i), self.mel_spec_transform(x_j)
            return torch.tensor(x_i, dtype=torch.float32).unsqueeze(0), torch.tensor(x_j, dtype=torch.float32).unsqueeze(0)
            
        return torch.tensor(x_i, dtype=torch.float32), torch.tensor(x_j, dtype=torch.float32)


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
    parser = argparse.ArgumentParser(description="Simsiam_BP")

    config = yaml_config_hook("config/ssbp_resnet50.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    
    train_dataset = BPDataset(
        sample_rate=args.sample_rate, 
        segment_second=args.segment_second, 
        piece_second=args.piece_second,
        data_dir=args.data_dir,
        augment_func=CLARTransform(
            sample_rate=args.sample_rate,
            duration=int(args.piece_second),
        ),
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        split="train",
        melspec_transform=args.melspec_transform,
        data_augmentation=args.data_augmentation,
        random_slice=args.random_slice,
        stems=['other'],
    )

    # Experiment 1: Mel-spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        f_max=args.fmax,
    )
    db_transform = T.AmplitudeToDB(stype="power")
    i = 0
    xi, xj = train_dataset[i][0], train_dataset[i][1]
    
    # original dataset
    x = np.load(train_dataset.data_path_list[i])
    
    torchaudio.save("visualization/x.wav", torch.tensor(x).clone().detach().unsqueeze(0), args.sample_rate)
    torchaudio.save("visualization/xi.wav", xi.clone().detach().unsqueeze(0), args.sample_rate)
    torchaudio.save("visualization/xj.wav", xj.clone().detach().unsqueeze(0), args.sample_rate)

    
    x1 = db_transform(mel_transform(xi))
    x2 = db_transform(mel_transform(xj))
    print(x1.shape, x2.shape)
    
    plot_spec_and_save(x1, "mel_spectrogram_x1.png", sr=args.sample_rate)
    plot_spec_and_save(x2, "mel_spectrogram_x2.png", sr=args.sample_rate)
    

    # Experiment 2: Resize spectrogram
    res_x1 = resize_spec(x1, target_size=(256, 256))
    res_x2 = resize_spec(x2, target_size=(256, 256))
    
    plot_spec_and_save(res_x1, "resized_x1.png", sr=args.sample_rate)
    plot_spec_and_save(res_x2, "resized_x2.png", sr=args.sample_rate)


    # for i in range(10):
    #     print(train_dataset[i][0].shape, train_dataset[i][1].shape)
    
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=4, shuffle=True,
    #     num_workers=args.workers, pin_memory=True, drop_last=True)
    
    # for ds in tqdm(train_loader):
    #     print(ds[0].shape, ds[1].shape)
    #     pass
    