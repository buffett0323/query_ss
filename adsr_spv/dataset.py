import os
import json
import random
import torch
import torchaudio
import h5py
import numpy as np
import nnAudio.features
import pytorch_lightning as pl
from util import get_memory_usage, format_memory_size

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional

class AudioADSRDataset(Dataset):
    """Loads (mel, ADSR[, audio]) triplets from disk."""

    def __init__(
        self,
        data_dir: Path,
        split: str = "train", # val, test
        unit_sec: float = 2.97,
        sr: int = 44100,
    ):
        self.unit_length = int(unit_sec*sr)

        with open(os.path.join(data_dir, split, "metadata.json"), "r") as f:
            metadata = json.load(f)

        self.items = []
        for chunk in tqdm(metadata, desc="Loading metadata"):
            item = [
                os.path.join(data_dir, split, chunk["file"]),
                torch.tensor([chunk["attack"],
                              chunk["decay"],
                              chunk["sustain"], # *1000,  # Scale sustain to 0-1000
                              chunk["release"]],
                             dtype=torch.float),  # [4]
            ]
            self.items.append(item)

        random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, adsr = self.items[idx]
        wav, _ = torchaudio.load(wav_path)
        wav = wav.mean(dim=0, keepdim=True)

        current_length = wav.shape[1]
        if current_length < self.unit_length:
            # Pad with zeros
            padding = self.unit_length - current_length
            wav = torch.nn.functional.pad(wav, (0, padding))
        elif current_length > self.unit_length:
            # Truncate
            wav = wav[:, :self.unit_length]

        return wav, adsr


class H5ADSRDataset(Dataset):
    """Loads pre-computed mel spectrograms and ADSR parameters from HDF5 files."""

    def __init__(
        self,
        h5_path: Path,
        split: str = "train",
        shuffle: bool = True,
        keep_open: bool = True,
    ):
        """
        Args:
            h5_path: Path to the HDF5 file containing mel spectrograms
            split: Data split (train/val/test) - used for logging
            shuffle: Whether to shuffle the dataset indices
            keep_open: Whether to keep the HDF5 file open for faster access
        """
        self.h5_path = h5_path
        self.split = split
        self.keep_open = keep_open

        if keep_open:
            # Keep file open for faster access
            self.h5_file = h5py.File(h5_path, 'r')
            self.num_samples = len(self.h5_file['mel_spectrograms'])
            self.mel_shape = self.h5_file['mel_spectrograms'].shape
            self.adsr_shape = self.h5_file['adsr_parameters'].shape
            self.metadata = dict(self.h5_file.attrs)
        else:
            # Get info without keeping file open
            with h5py.File(h5_path, 'r') as h5_file:
                self.num_samples = len(h5_file['mel_spectrograms'])
                self.mel_shape = h5_file['mel_spectrograms'].shape
                self.adsr_shape = h5_file['adsr_parameters'].shape
                self.metadata = dict(h5_file.attrs)

        # Create indices for shuffling
        self.indices = list(range(self.num_samples))
        if shuffle:
            random.shuffle(self.indices)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Map shuffled index to actual index
        actual_idx = self.indices[idx]

        if self.keep_open:
            # Use the already open file
            mel = torch.from_numpy(self.h5_file['mel_spectrograms'][actual_idx]).float()
            adsr = torch.from_numpy(self.h5_file['adsr_parameters'][actual_idx]).float()
        else:
            # Open file for each access (slower but uses less memory)
            with h5py.File(self.h5_path, 'r') as h5_file:
                mel = torch.from_numpy(h5_file['mel_spectrograms'][actual_idx]).float()
                adsr = torch.from_numpy(h5_file['adsr_parameters'][actual_idx]).float()

        return mel, adsr

    def get_metadata(self):
        """Return the metadata stored in the HDF5 file."""
        return self.metadata.copy()

    def close(self):
        """Close the HDF5 file if it's open."""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __del__(self):
        """Ensure the file is closed when the object is destroyed."""
        self.close()


class Mel_ADSRDataset(Dataset):
    def __init__(
        self,
        npy_path: Path,
        split: str = "train",
    ):
        self.npy_path = npy_path
        self.split = split

        # Load metadata
        metadata_path = self.npy_path / self.split / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.metadata = [
            [
                str(self.npy_path / self.split / chunk["file"]), #chunk["npy_file"]),
                torch.tensor([chunk["attack"],
                            chunk["decay"],
                            chunk["sustain"], # *1000,  # Scale sustain to 0-1000
                            chunk["release"]],
                            dtype=torch.float),  # [4]
            ]
            for chunk in tqdm(metadata, desc="Pre-Loading metadata")
        ]
        random.shuffle(self.metadata)

        # # Load parameters
        # params_path = self.npy_path / self.split / "params.json"
        # with open(params_path, "r") as f:
        #     self.params = json.load(f)


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        mel_path, adsr = self.metadata[idx]
        mel = np.load(mel_path)
        return mel, adsr


class Mel_ADSRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: Dict[str, Any]
    ):
        super().__init__()
        self.config = config
        self.data_dir = Path(config['data_dir'])

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = Mel_ADSRDataset(
                npy_path=self.data_dir,
                split="train",
            )
            self.val_dataset = Mel_ADSRDataset(
                npy_path=self.data_dir,
                split="val",
            )

        if stage == 'test' or stage is None:
            self.test_dataset = Mel_ADSRDataset(
                npy_path=self.data_dir,
                split="test",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.config['prefetch_factor'],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.config['prefetch_factor'],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.config['prefetch_factor'],
        )




class ADSRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: Dict[str, Any]
    ):
        super().__init__()
        self.config = config
        self.data_dir = Path(config['data_dir'])

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = AudioADSRDataset(
                data_dir=self.data_dir,
                split="train",
            )
            self.val_dataset = AudioADSRDataset(
                data_dir=self.data_dir,
                split="val",
            )

        if stage == 'test' or stage is None:
            self.test_dataset = AudioADSRDataset(
                data_dir=self.data_dir,
                split="test",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.config['prefetch_factor'],
            # cache_size=self.config['cache_size'],
            # use_memory_mapping=self.config['use_memory_mapping'],
            # drop_last=self.config['drop_last_train'],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.config['prefetch_factor'],
            # cache_size=self.config['cache_size'],
            # use_memory_mapping=self.config['use_memory_mapping'],
            # drop_last=self.config['drop_last_train'],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.config['prefetch_factor'],
            # cache_size=self.config['cache_size'],
            # use_memory_mapping=self.config['use_memory_mapping'],
            # drop_last=self.config['drop_last_train'],
        )




class Mel_ADSR_h5_Dataset(Dataset):
    """Ultra-fast mel spectrogram dataset using HDF5 with memory mapping."""

    def __init__(
        self,
        h5_path: Path,
        split: str = "train",
        load_to_memory: bool = True,  # Load all data to memory for maximum speed
    ):
        self.split = split
        self.load_to_memory = load_to_memory

        # Track initial memory usage
        initial_memory = get_memory_usage()
        print(f"Initial memory usage: {initial_memory:.2f} GB")

        # Load HDF5 file
        with h5py.File(h5_path, 'r') as h5_file:
            # Get dataset info
            self.num_samples = len(h5_file[f'{split}_mel'])
            self.mel_shape = h5_file[f'{split}_mel'].shape
            self.adsr_shape = h5_file[f'{split}_adsr'].shape

            # Calculate expected memory usage
            mel_size_bytes = h5_file[f'{split}_mel'].nbytes
            adsr_size_bytes = h5_file[f'{split}_adsr'].nbytes
            total_size_bytes = mel_size_bytes + adsr_size_bytes

            print(f"Dataset info for {split}:")
            print(f"  - Samples: {self.num_samples}")
            print(f"  - Mel shape: {self.mel_shape}")
            print(f"  - ADSR shape: {self.adsr_shape}")
            print(f"  - Expected memory usage: {format_memory_size(total_size_bytes)}")

            if load_to_memory:
                # Load all data to memory for maximum speed
                print(f"Loading {split} data to memory...")

                # Load mel data
                mel_memory_before = get_memory_usage()
                self.mel_data = torch.from_numpy(h5_file[f'{split}_mel'][:]).float()
                mel_memory_after = get_memory_usage()
                mel_memory_used = mel_memory_after - mel_memory_before

                # Load ADSR data
                adsr_memory_before = get_memory_usage()
                self.adsr_data = torch.from_numpy(h5_file[f'{split}_adsr'][:]).float()
                adsr_memory_after = get_memory_usage()
                adsr_memory_used = adsr_memory_after - adsr_memory_before

                # Calculate total memory usage
                final_memory = get_memory_usage()
                total_memory_used = final_memory - initial_memory

                print(f"Memory usage breakdown:")
                print(f"  - Mel data loaded: {format_memory_size(mel_size_bytes)} (actual: {mel_memory_used:.2f} GB)")
                print(f"  - ADSR data loaded: {format_memory_size(adsr_size_bytes)} (actual: {adsr_memory_used:.2f} GB)")
                print(f"  - Total memory used: {format_memory_size(total_size_bytes)} (actual: {total_memory_used:.2f} GB)")
                print(f"  - Final memory usage: {final_memory:.2f} GB")

            else:
                # Memory mapping for large datasets
                self.h5_file = h5py.File(h5_path, 'r')
                self.mel_dataset = self.h5_file[f'{split}_mel']
                self.adsr_dataset = self.h5_file[f'{split}_adsr']
                print(f"Using memory mapping (no data loaded to RAM)")

        # Create shuffled indices
        self.indices = list(range(self.num_samples))
        random.shuffle(self.indices)

    def get_memory_stats(self):
        """Get current memory usage statistics."""
        if not self.load_to_memory:
            return "Memory mapping mode - no data loaded to RAM"

        current_memory = get_memory_usage()
        if hasattr(self, 'mel_data') and hasattr(self, 'adsr_data'):
            mel_size = self.mel_data.element_size() * self.mel_data.nelement()
            adsr_size = self.adsr_data.element_size() * self.adsr_data.nelement()
            total_size = mel_size + adsr_size

            return {
                'current_memory_gb': current_memory,
                'mel_data_size': format_memory_size(mel_size),
                'adsr_data_size': format_memory_size(adsr_size),
                'total_data_size': format_memory_size(total_size),
                'samples_loaded': self.num_samples
            }
        return "Data not loaded yet"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        if self.load_to_memory:
            # Ultra-fast memory access
            mel = self.mel_data[actual_idx]
            adsr = self.adsr_data[actual_idx]
        else:
            # Memory-mapped access
            mel = torch.from_numpy(self.mel_dataset[actual_idx]).float()
            adsr = torch.from_numpy(self.adsr_dataset[actual_idx]).float()

        return mel, adsr

    def close(self):
        """Close HDF5 file if using memory mapping."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

    def __del__(self):
        self.close()


class Mel_ADSR_h5_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: Dict[str, Any]
    ):
        super().__init__()
        self.config = config
        self.h5_path = Path(config['h5_path'])  # Path to HDF5 file
        self.load_to_memory = config.get('load_to_memory', True)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            print("\n" + "="*50)
            print("Loading Training Dataset")
            print("="*50)
            self.train_dataset = Mel_ADSR_h5_Dataset(
                h5_path=self.h5_path,
                split="train",
                load_to_memory=self.load_to_memory,
            )

            print("\n" + "="*50)
            print("Loading Validation Dataset")
            print("="*50)
            self.val_dataset = Mel_ADSR_h5_Dataset(
                h5_path=self.h5_path,
                split="val",
                load_to_memory=self.load_to_memory,
            )

        if stage == 'test' or stage is None:
            print("\n" + "="*50)
            print("Loading Test Dataset")
            print("="*50)
            self.test_dataset = Mel_ADSR_h5_Dataset(
                h5_path=self.h5_path,
                split="test",
                load_to_memory=self.load_to_memory,
            )

        # Print final memory summary
        if stage == 'fit' or stage is None:
            print("\n" + "="*50)
            print("MEMORY USAGE SUMMARY")
            print("="*50)
            print("Training dataset:")
            train_stats = self.train_dataset.get_memory_stats()
            if isinstance(train_stats, dict):
                for key, value in train_stats.items():
                    print(f"  {key}: {value}")

            print("\nValidation dataset:")
            val_stats = self.val_dataset.get_memory_stats()
            if isinstance(val_stats, dict):
                for key, value in val_stats.items():
                    print(f"  {key}: {value}")

            print(f"\nTotal memory usage: {get_memory_usage():.2f} GB")
            print("="*50)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.config['prefetch_factor'],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.config['prefetch_factor'],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.config['prefetch_factor'],
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = {
        "batch_size": 32,
        "num_workers": 16,
        "prefetch_factor": 4,
        "h5_path": "/home/buffett/dataset/rendered_adsr_unpaired_mel_h5/adsr_mel.h5",
        "load_to_memory": True,
    }

    dl = Mel_ADSR_h5_DataModule(config=config)
    dl.setup()
    train_loader = dl.train_dataloader()
    for batch in tqdm(train_loader, desc="Train"):
        pass

    val_loader = dl.val_dataloader()
    for batch in tqdm(val_loader, desc="Validation"):
        pass

    test_loader = dl.test_dataloader()
    for batch in tqdm(test_loader, desc="Test"):
        pass
