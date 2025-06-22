import os
import json
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


class AudioADSRDataset(Dataset):
    """Loads (mel, ADSR[, audio]) triplets from disk."""

    def __init__(
        self,
        metadata_dir: Path,
        data_dir: Path,
    ):

        with open(os.path.join(metadata_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        self.items = []
        for chunk in tqdm(metadata, desc="Loading metadata"):
            item = [
                os.path.join(data_dir, chunk["file"].replace(".wav", "_mel.npy")),
                torch.tensor([chunk["attack"], chunk["decay"], chunk["sustain"], chunk["release"]], dtype=torch.float),  # [4]
            ]
            self.items.append(item)
        random.shuffle(self.items)
        print("Loaded {} items".format(len(self.items)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        mel_path, adsr = self.items[idx]
        mel = np.load(mel_path)
        return mel, adsr


if __name__ == "__main__":
    dataset = AudioADSRDataset(
        metadata_dir=Path("/mnt/gestalt/home/buffett/rendered_adsr_dataset_npy"),
        data_dir=Path("/mnt/gestalt/home/buffett/rendered_adsr_dataset_npy_new_mel"),
    )
    print(dataset[0])
