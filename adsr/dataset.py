import os
import json
import random
import h5py
import torch
from torch.utils.data import Dataset
from util import MAX_AUDIO_LENGTH


class ADSRDataset(Dataset):
    def __init__(
        self, 
        h5_path,
        env_amount=600,
        pair_amount=1547,
    ):
        super().__init__()
        self.env_amount = env_amount
        self.pair_amount = pair_amount
        
        # Load h5 file
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"h5 file not found at {h5_path}")
            
        self.h5_file = h5py.File(h5_path, 'r')
        
        # Verify required datasets exist
        required_datasets = ['mel_specs', 'env_ids', 'file_paths']
        for dataset in required_datasets:
            if dataset not in self.h5_file:
                raise KeyError(f"Required dataset '{dataset}' not found in h5 file")
        
        self.mel_specs = self.h5_file['mel_specs']
        self.env_ids = self.h5_file['env_ids']
        self.file_paths = self.h5_file['file_paths']
        
        # Verify data shapes
        if len(self.mel_specs) != len(self.env_ids) or len(self.mel_specs) != len(self.file_paths):
            raise ValueError("Inconsistent dataset lengths in h5 file")
            
        print(f"Loaded dataset with {len(self.mel_specs)} samples")
        print(f"Mel spectrogram shape: {self.mel_specs.shape}")
        
        # Create environment ID to index mapping
        self.env_id_to_indices = {i: [] for i in range(self.env_amount)}
        for i, env_id in enumerate(self.env_ids):
            self.env_id_to_indices[env_id].append(i)
            
        # Verify environment IDs
        for env_id in range(self.env_amount):
            if not self.env_id_to_indices[env_id]:
                print(f"Warning: No samples found for environment ID {env_id}")
    
    
    def __len__(self):
        return len(self.mel_specs)
    
    
    def __get_random_pair(self, env_id):
        indices = self.env_id_to_indices[env_id]
        if not indices:
            raise ValueError(f"No samples found for environment ID {env_id}")
        return random.choice(indices)
    
    
    def __getitem__(self, index):
        try:
            # Get first mel spectrogram
            mel1 = torch.from_numpy(self.mel_specs[index]).float()  # Ensure float type
            env_id = int(self.env_ids[index])  # Ensure int type
            
            # Get random pair from same environment
            pair_idx = self.__get_random_pair(env_id)
            mel2 = torch.from_numpy(self.mel_specs[pair_idx]).float()  # Ensure float type
            
            result = {
                "env_id": env_id,
                "mel1": mel1,
                "mel2": mel2,
                "path1": self.file_paths[index].decode('utf-8'),  # Convert bytes to string
                "path2": self.file_paths[pair_idx].decode('utf-8')  # Convert bytes to string
            }
            
            return result
            
        except Exception as e:
            print(f"Error loading sample at index {index}: {str(e)}")
            # Return a valid sample as fallback
            return self.__getitem__((index + 1) % len(self))
    
    
    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
            
    
    def get_env_stats(self):
        """Return statistics about environment distribution"""
        env_counts = {i: len(indices) for i, indices in self.env_id_to_indices.items()}
        return {
            'total_samples': len(self.mel_specs),
            'env_counts': env_counts,
            'min_samples_per_env': min(env_counts.values()),
            'max_samples_per_env': max(env_counts.values()),
            'avg_samples_per_env': sum(env_counts.values()) / len(env_counts)
        }
        
        
if __name__ == "__main__":
    path = "/mnt/gestalt/home/buffett/adsr_h5/adsr_mel.h5"
    dataset = ADSRDataset(h5_path=path)
    print(dataset.get_env_stats())
    