import os
import torch
import torchaudio
import argparse
import torchvision.transforms as transforms
import torchaudio.transforms as T
import numpy as np

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from utils import yaml_config_hook
from tqdm import tqdm


class BPDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        segment_second,
        data_dir,
        augment_func,
        piece_second=3,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        split="train",
        melspec_transform=False,
        data_augmentation=True,
        random_slice=False,
        stems=["other"], #["vocals", "bass", "drums", "other"], # VBDO
    ):

        self.stems = stems
        self.data_path_list = [
            os.path.join(data_dir, folder, f"{stem}.npy")
            for folder in os.listdir(data_dir)
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

    def mel_spec_transform(self, x):
        mel_transform = T.MelSpectrogram(
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            f_max=args.fmax,
        )
        db_transform = T.AmplitudeToDB(stype="power")
        return db_transform(mel_transform(x))
    
        
    def __len__(self): #""" Total we got 175698 files * 4 tracks """
        return len(self.data_path_list)
    
    
    def __getitem__(self, idx):
        path = self.data_path_list[idx]
        
        # Read data and segment
        x = torch.tensor(np.load(path))
        return self.mel_spec_transform(x).unsqueeze(0)



def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=24)

    sum_ = 0.0
    sum_sq_ = 0.0
    count = 0

    for mel_spec in tqdm(loader):
        mel_spec = mel_spec.float()  # Ensure float precision
        sum_ += mel_spec.mean()
        sum_sq_ += (mel_spec ** 2).mean()
        count += 1

    mean = sum_ / count
    std = (sum_sq_ / count - mean ** 2) ** 0.5  # Standard deviation formula

    return mean.item(), std.item()



# Example usage with your dataset
parser = argparse.ArgumentParser(description="Simsiam_BP")

config = yaml_config_hook("config/ssbp_swint.yaml")
for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))

args = parser.parse_args()

ds = BPDataset(
    sample_rate=args.sample_rate, 
    segment_second=args.segment_second, 
    piece_second=args.piece_second,
    data_dir=args.data_dir,
    augment_func=None,
    n_mels=args.n_mels,
    n_fft=args.n_fft,
    hop_length=args.hop_length,
    split="train",
    melspec_transform=False, #args.melspec_transform,
    data_augmentation=False, #args.data_augmentation,
    random_slice=False, #args.random_slice,
    stems=['other'],
)



mean, std = compute_mean_std(ds)
print(f"Mean: {mean}, Std: {std}")
