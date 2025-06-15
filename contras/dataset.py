import os
import random
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
from utils import yaml_config_hook


class BeatportDataset(Dataset):
    def __init__(
        self,
        args,
        data_path_list,
        split="train",
        n_fft=2048,
        hop_length=1024,
    ):
        self.split = split
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.data_path_list = data_path_list


    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        path = self.data_path_list[idx]
        return np.load(path)




class BeatportDataModule(LightningDataModule):
    def __init__(
        self,
        args,
        npy_list,
    ):
        super(BeatportDataModule, self).__init__()
        self.args = args
        self.pin_memory = True
        self.drop_last = False

        random.shuffle(npy_list)
        valid_size = args.batch_size * round(int(len(npy_list)*0.1) / args.batch_size)

        # Split the data
        self.valid_data = npy_list[:valid_size]
        self.test_data = npy_list[valid_size:valid_size*2]
        self.train_data = npy_list[valid_size*2:]

        print("Total dataset size:", len(npy_list))
        print("Train dataset size:", len(self.train_data))
        print("Valid dataset size:", len(self.valid_data))
        print("Test dataset size:", len(self.test_data))


    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_ds = BeatportDataset(
                args=self.args,
                data_path_list=self.train_data,
                split="train",
            )

            self.val_ds = BeatportDataset(
                args=self.args,
                data_path_list=self.valid_data,
                split="valid",
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = BeatportDataset(
                args=self.args,
                data_path_list=self.test_data,
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


# TODO: transform random
class SimCLRTransform(nn.Module):
    def __init__(self, args):
        super(SimCLRTransform, self).__init__()
        self.amplitude_to_db = T.AmplitudeToDB()#.to(self.my_device)
        self.transforms = [
            lambda spectrogram: self.time_mask(spectrogram, mask_param=args.tm_param),
            lambda spectrogram: self.frequency_mask(spectrogram, mask_param=args.fm_param),
            lambda spectrogram: self.random_crop(spectrogram, crop_size=args.crop_size),
            lambda spectrogram: self.add_noise(spectrogram, noise_level=args.noise_level),
        ]

    def time_mask(self, spectrogram, mask_param=30):
        time_mask = T.TimeMasking(time_mask_param=mask_param)#.to(self.my_device)
        return time_mask(spectrogram)


    def frequency_mask(self, spectrogram, mask_param=15):
        freq_mask = T.FrequencyMasking(freq_mask_param=mask_param)#.to(self.my_device)
        return freq_mask(spectrogram)


    def random_crop(self, spectrogram, crop_size):
        max_start = spectrogram.size(-1) - crop_size
        if max_start > 0:
            start = random.randint(0, max_start)
            return spectrogram[:, :, start:start + crop_size]
        return spectrogram


    def add_noise(self, spectrogram, noise_level=0.005):
        noise = noise_level * torch.randn_like(spectrogram)#.to(self.my_device)
        return spectrogram + noise


    def __call__(self, mel_spec):
        mel_spec = self.amplitude_to_db(mel_spec)

        # Apply random augmentations
        transform1, transform2 = random.sample(self.transforms, 2)
        mel_spec1 = transform1(mel_spec)
        mel_spec2 = transform2(mel_spec)

        # Ensure the output shapes match the input shape
        mel_spec1 = self._preserve_shape(mel_spec1, mel_spec)
        mel_spec2 = self._preserve_shape(mel_spec2, mel_spec)

        return mel_spec1, mel_spec2

    def _preserve_shape(self, transformed, original):
        """
        Ensures the transformed spectrogram retains the original shape.
        """
        # Add batch and channel dimensions if missing
        if len(transformed.shape) == 3:  # If shape is [C, H, W]
            transformed = transformed.unsqueeze(0)  # Add batch dimension

        # Perform interpolation
        transformed = torch.nn.functional.interpolate(
            transformed,
            size=original.shape[-2:],  # Only match spatial dimensions (H, W)
            mode='nearest'
        )

        # Remove batch dimension if it was added
        if transformed.shape[0] == 1:  # If batch dimension was added
            transformed = transformed.squeeze(0)

        return transformed



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")

    config = yaml_config_hook("config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    npy_list = [
        os.path.join(args.npy_dir, folder_name, file_name)
        for folder_name in os.listdir(args.npy_dir)
            for file_name in os.listdir(os.path.join(args.npy_dir, folder_name))
                if file_name.endswith(".npy")
    ]
    random.shuffle(npy_list)
    n = len(npy_list)
    train_end = int(n * 0.8)  # 80%
    test_end = train_end + int(n * 0.1)  # 80% + 10%

    # Split the data
    train_data = npy_list[:train_end]
    test_data = npy_list[train_end:test_end]
    valid_data = npy_list[test_end:]

    print("Train dataset:", len(train_data))
    print("Valid dataset:", len(valid_data))
    print("Test dataset:", len(test_data))


    train_dataset = BeatportDataset(
        args=args,
        data_path_list=train_data,
        split="train",
    )

    for i in range(100):
        print(train_dataset[i].shape)
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.workers,
    #     pin_memory=True,        # Faster transfer to GPU
    #     prefetch_factor=2,      # Prefetch 2 batches per worker
    #     persistent_workers=True # Keep workers alive between epochs
    # )
    # for t in tqdm(train_loader):
    #     pass
