import os
import json
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional

class NPYADSRDataset(Dataset):
    """Loads pre-computed mel spectrograms from .npy files efficiently."""

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        shuffle: bool = True,
        preload: bool = False,
        max_preload: int = 1000
    ):
        """
        Args:
            data_dir: Directory containing the .npy files and metadata
            split: Data split (train/val/test)
            shuffle: Whether to shuffle the dataset indices
            preload: Whether to preload data into memory
            max_preload: Maximum number of samples to preload
        """
        self.data_dir = data_dir / split
        self.split = split
        self.preload = preload
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        # Load parameters
        params_path = self.data_dir / "params.json"
        with open(params_path, "r") as f:
            self.params = json.load(f)
        
        self.num_samples = len(self.metadata)
        
        # Preload data if requested
        if preload and self.num_samples <= max_preload:
            print(f"Preloading {self.num_samples} samples into memory...")
            self.preloaded_data = []
            for item in tqdm(self.metadata, desc="Preloading"):
                npy_path = self.data_dir / item['npy_file']
                mel = np.load(npy_path)
                adsr = np.array([item['attack'], item['decay'], item['sustain'], item['release']], dtype=np.float32)
                self.preloaded_data.append((mel, adsr))
            print("Preloading complete!")
        else:
            self.preloaded_data = None
        
        # Create indices for shuffling
        self.indices = list(range(self.num_samples))
        if shuffle:
            random.shuffle(self.indices)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Map shuffled index to actual index
        actual_idx = self.indices[idx]
        
        if self.preloaded_data is not None:
            # Use preloaded data
            mel, adsr = self.preloaded_data[actual_idx]
        else:
            # Load from disk
            item = self.metadata[actual_idx]
            npy_path = self.data_dir / item['npy_file']
            mel = np.load(npy_path)
            adsr = np.array([item['attack'], item['decay'], item['sustain'], item['release']], dtype=np.float32)
        
        # Convert to torch tensors
        mel = torch.from_numpy(mel).float()
        adsr = torch.from_numpy(adsr).float()
        
        return mel, adsr
    
    def get_metadata(self):
        """Return the metadata stored in the dataset."""
        return self.metadata.copy()
    
    def get_params(self):
        """Return the processing parameters."""
        return self.params.copy()

class NPYADSRDataLoader:
    """Convenience class for creating DataLoaders with common configurations."""
    
    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        preload: bool = False,
        max_preload: int = 1000
    ):
        """
        Args:
            data_dir: Directory containing the .npy files
            split: Data split (train/val/test)
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the dataset
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
            preload: Whether to preload data into memory
            max_preload: Maximum number of samples to preload
        """
        self.dataset = NPYADSRDataset(
            data_dir=data_dir,
            split=split,
            shuffle=shuffle,
            preload=preload,
            max_preload=max_preload
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    def get_dataset_info(self):
        """Get information about the dataset."""
        return {
            'num_samples': len(self.dataset),
            'batch_size': self.dataloader.batch_size,
            'num_batches': len(self.dataloader),
            'mel_shape': self.dataset.params.get('n_mels', 'unknown'),
            'mel_frames': self.dataset.params.get('expected_mel_frames', 'unknown'),
            'preloaded': self.dataset.preloaded_data is not None
        }

def test_npy_dataset():
    """Test the .npy dataset functionality."""
    
    data_dir = Path("/home/buffett/dataset/rendered_adsr_unpaired_npy")
    
    if not data_dir.exists():
        print(f"Dataset directory not found: {data_dir}")
        return
    
    print("Testing .npy dataset...")
    
    # Test basic dataset
    dataset = NPYADSRDataset(data_dir, split="train", shuffle=False)
    print(f"Dataset size: {len(dataset)}")
    print(f"Parameters: {dataset.get_params()}")
    
    # Test loading a few samples
    for i in range(min(3, len(dataset))):
        mel, adsr = dataset[i]
        print(f"Sample {i}: mel shape {mel.shape}, adsr {adsr}")
    
    # Test DataLoader
    dataloader = NPYADSRDataLoader(
        data_dir=data_dir,
        split="train",
        batch_size=4,
        shuffle=False,
        num_workers=2
    )
    
    print(f"\nDataLoader info: {dataloader.get_dataset_info()}")
    
    # Test batch loading
    for i, (mel_batch, adsr_batch) in enumerate(dataloader):
        print(f"Batch {i}: mel shape {mel_batch.shape}, adsr shape {adsr_batch.shape}")
        if i >= 2:  # Test first 3 batches
            break

if __name__ == "__main__":
    test_npy_dataset() 