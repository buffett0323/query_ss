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

# from .common import (np, torch, F, torchaudio)


class BaseRawAudioDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, tfms=None, random_crop=False):
        self.cfg = cfg
        self.tfms = tfms
        self.random_crop = random_crop

    def __len__(self):
        raise NotImplementedError('implement me')

    def get_audio(self, index):
        raise NotImplementedError('implement me')

    def get_label(self, index):
        return None # implement me

    def __getitem__(self, index):
        wav = self.get_audio(index) # shape is expected to be (cfg.unit_samples,)

        # Trim or stuff padding
        l = len(wav)
        unit_samples = self.cfg.unit_samples
        if l > unit_samples:
            start = np.random.randint(l - unit_samples) if self.random_crop else 0
            wav = wav[start:start + unit_samples]
        elif l < unit_samples:
            wav = F.pad(wav, (0, unit_samples - l), mode='constant', value=0)
        wav = wav.to(torch.float)

        # Apply transforms
        if self.tfms is not None:
            wav = self.tfms(wav)

        # Return item
        label = self.get_label(index)
        return wav if label is None else (wav, label)


class WavDataset(BaseRawAudioDataset):
    def __init__(self, cfg, audio_files, labels, tfms=None, random_crop=False):
        super().__init__(cfg, tfms=tfms, random_crop=random_crop)
        self.files = audio_files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def get_audio(self, index):
        filename = self.files[index]
        wav, sr = torchaudio.load(filename)
        assert sr == self.cfg.sample_rate, f'Convert .wav files to {self.cfg.sample_rate} Hz. {filename} has {sr} Hz.'
        return wav[0]

    def get_label(self, index):
        return None if self.labels is None else torch.tensor(self.labels[index])


class ADSR_h5_Dataset(Dataset):
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


class ADSRDataset(Dataset):
    def __init__(
        self,
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

        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        paths = [chunk["file"].replace(".wav", ".npy") for chunk in metadata]
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
        # wav, _ = torchaudio.load(os.path.join(self.data_dir, path))
        # wav = wav.mean(dim=0)

        # if wav.shape[-1] > self.unit_samples:
        #     start = np.random.randint(wav.shape[-1] - self.unit_samples)
        #     wav = wav[:, start:start + self.unit_samples]
        # elif wav.shape[-1] < self.unit_samples:
        #     wav = F.pad(wav, (0, self.unit_samples - wav.shape[-1]), mode='constant', value=0)
        # return wav


    def __getitem__(self, index):
        path1 = self.paths[index]

        # Get random pair from same envelope
        env_id = int(path1.split("_")[1])
        path2 = self._get_random_pair(env_id)

        wav1 = self._get_audio(path1)
        wav2 = self._get_audio(path2)

        return wav1, wav2
        # result = {
        #     "env_id": env_id,
        #     "wav1": wav1,
        #     "wav2": wav2,
        #     "path1": path1,
        #     "path2": path2,
        # }

        # return result




if __name__ == "__main__":
    # path = "/mnt/gestalt/home/buffett/adsr_h5/adsr_mel.h5"
    # dataset = ADSR_h5_Dataset(h5_path=path)
    # print(dataset.get_env_stats())

    path = "/mnt/gestalt/home/buffett/rendered_adsr_dataset_npy"
    dataset = ADSRDataset(data_dir=path)
    wav1, wav2 = dataset[0]
    print(wav1.shape)
    print(wav2.shape)
