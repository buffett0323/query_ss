import os
import json
import random
import torch
import torchaudio
import numpy as np
import nnAudio.features
import pytorch_lightning as pl

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



if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    dataset = AudioADSRDataset(
        data_dir=Path("/home/buffett/dataset/rendered_adsr_unpaired"),
        split="train",
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    to_spec = nnAudio.features.MelSpectrogram(
        sr=44100,
        n_mels=128,
        fmin=20,
        fmax=22050,
        hop_length=512,
        n_fft=2048,
        window='hann',
        center=True,
        power=2.0,
    ).to(device)

    for batch in tqdm(dataloader):
        wav, adsr = batch

        mel = to_spec(wav.to(device))
        mel = mel.unsqueeze(1) # [BS, 1, 128, 256]
        if mel.isnan().any():
            print("mel is nan")
            print(mel)
