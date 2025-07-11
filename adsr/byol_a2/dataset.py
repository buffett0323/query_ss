"""BYOL for Audio: Dataset class definition."""
import os
import json
import random
import h5py
import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
from tqdm import tqdm

class ADSR_h5_Dataset(Dataset):
    def __init__(
        self,
        h5_path,
        env_amount=600,
        pair_amount=1547,
        cache_size=1000,  # Add caching
    ):
        super().__init__()
        self.env_amount = env_amount
        self.pair_amount = pair_amount
        self.cache_size = cache_size

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

        # Create environment ID to index mapping - pre-compute for speed
        self.env_id_to_indices = {i: [] for i in range(self.env_amount)}
        for i, env_id in enumerate(self.env_ids):
            self.env_id_to_indices[env_id].append(i)

        # Pre-compute random pairs for faster access
        self._precompute_pairs()

        # Initialize cache
        self.cache = {}
        self.cache_keys = []

        # Verify environment IDs
        for env_id in range(self.env_amount):
            if not self.env_id_to_indices[env_id]:
                print(f"Warning: No samples found for environment ID {env_id}")

    def _precompute_pairs(self):
        """Pre-compute random pairs for faster access"""
        self.pair_indices = []
        for i in tqdm(range(len(self.mel_specs))):
            env_id = int(self.env_ids[i])
            indices = self.env_id_to_indices[env_id]
            if len(indices) > 1:
                # Exclude self from possible pairs
                possible_pairs = [idx for idx in indices if idx != i]
                if possible_pairs:
                    pair_idx = random.choice(possible_pairs)
                else:
                    pair_idx = i  # Fallback to self if no other pairs
            else:
                pair_idx = i
            self.pair_indices.append(pair_idx)

    def _get_cached_sample(self, index):
        """Get sample from cache or load from H5"""
        if index in self.cache:
            return self.cache[index]

        # Load from H5
        sample = torch.from_numpy(self.mel_specs[index]).float()

        # Add to cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = self.cache_keys.pop(0)
            del self.cache[oldest_key]

        self.cache[index] = sample
        self.cache_keys.append(index)

        return sample

    def __len__(self):
        return len(self.mel_specs)

    def __getitem__(self, index):
        # Get first mel spectrogram
        mel1 = self._get_cached_sample(index)
        file_path1 = self.file_paths[index]

        # Get pre-computed pair
        pair_idx = self.pair_indices[index]
        mel2 = self._get_cached_sample(pair_idx)
        file_path2 = self.file_paths[pair_idx]

        return mel1, mel2, file_path1, file_path2

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

class ADSRDataset(Dataset):
    def __init__(
        self,
        metadata_dir,
        data_dir,
        unit_sec=2.97,
        env_amount=600,
        pair_amount=1547,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.unit_sec = unit_sec
        self.unit_samples = int(unit_sec * 44100)
        self.env_amount = env_amount
        self.pair_amount = pair_amount

        with open(os.path.join(metadata_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        paths = [chunk["file"] for chunk in metadata]
        paths = [path.replace(".wav", "_mel.npy").replace(
            "rendered_adsr_dataset_npy", "rendered_adsr_dataset_npy_new_mel"
            ) for path in paths]
        random.shuffle(paths)
        self.paths = paths

        print(f"Loaded dataset with {len(self.paths)} samples")

        # Create environment ID to index mapping
        self.env_id_to_indices = {i: [] for i in range(self.env_amount)}
        for path in self.paths:
            env_id = int(path.split("_")[1])
            self.env_id_to_indices[env_id].append(path)


    def __len__(self):
        return len(self.paths)


    def _get_random_pair(self, env_id):
        return random.choice(self.env_id_to_indices[env_id])


    def _get_audio(self, path):
        wav = np.load(os.path.join(self.data_dir, path))
        return wav


    def _get_mel(self, path):
        mel = np.load(os.path.join(self.data_dir, path))
        return mel

    def __getitem__(self, index):
        path1 = self.paths[index]

        # Get random pair from same envelope
        env_id = int(path1.split("_")[1])
        path2 = self._get_random_pair(env_id)

        mel1 = self._get_mel(path1)
        mel2 = self._get_mel(path2)
        return mel1, mel2

        # wav1 = self._get_audio(path1)
        # wav2 = self._get_audio(path2)

        # return wav1, wav2




if __name__ == "__main__":
    path = "/mnt/gestalt/home/buffett/adsr_h5/adsr_new_mel.h5"
    dataset = ADSR_h5_Dataset(h5_path=path)
    print(dataset.get_env_stats())
    a1, a2, f1, f2 = dataset[0]
    print("a1.shape, a2.shape, f1, f2", a1.shape, a2.shape, f1, f2)
