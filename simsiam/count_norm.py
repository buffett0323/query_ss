import torch
import torchaudio
import argparse
import torchvision.transforms as transforms
import torchaudio.transforms as T
import numpy as np
from torch.utils.data import DataLoader
from dataset import BPDataset
from utils import yaml_config_hook
from tqdm import tqdm

def compute_mean_std(dataset, mel_transform, db_transform):
    """
    Compute mean and standard deviation of a dataset of Mel spectrograms (single-channel).
    """
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=24)

    sum_ = 0.0
    sum_sq_ = 0.0
    count = 0

    for x1, x2 in tqdm(loader):  # Assuming dataset returns (mel_spec, label)
        x1 = db_transform(mel_transform(x1))
        x2 = db_transform(mel_transform(x2))
        
        x1 = x1.float()  # Ensure float precision
        sum_ += x1.mean()
        sum_sq_ += (x1 ** 2).mean()
        count += 1
        
        x2 = x2.float()  # Ensure float precision
        sum_ += x2.mean()
        sum_sq_ += (x2 ** 2).mean()
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

train_dataset = BPDataset(
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

# Experiment 1: Mel-spectrogram
mel_transform = T.MelSpectrogram(
    sample_rate=args.sample_rate,
    n_mels=args.n_mels,
    n_fft=args.n_fft,
    hop_length=args.hop_length,
    f_max=args.fmax,
)
db_transform = T.AmplitudeToDB(stype="power")
mean, std = compute_mean_std(train_dataset, mel_transform, db_transform)
print(f"Mean: {mean}, Std: {std}")
