import os
import torch
import torchaudio
import argparse
import torchvision.transforms as transforms
import torchaudio.transforms as T
import numpy as np
import nnAudio.features

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from utils import yaml_config_hook
from tqdm import tqdm
from dataset import SegmentBPDataset


def compute_mean_std(dataset, device, to_spec):
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=24,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    sum_ = 0.0
    sum_sq_ = 0.0
    count = 0

    for x, _, _ in tqdm(loader, desc="Computing Mean and Std"):
        x = x.to(device)
        logmel = (to_spec(x) + torch.finfo().eps).log()  # [B, n_mels, T]

        sum_ += logmel.sum().item()
        sum_sq_ += (logmel ** 2).sum().item()
        count += logmel.numel()

    mean = sum_ / count
    std = (sum_sq_ / count - mean ** 2) ** 0.5

    return mean, std



if __name__ == "__main__":
    # Example usage with your dataset
    parser = argparse.ArgumentParser(description="MoCo_BP")

    config = yaml_config_hook("config/ssbp_convnext_pairs.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    dataset = SegmentBPDataset(
        data_dir=args.seg_dir,
        split="train",
        stem="bass_other",
        eval_mode=True,
        num_seq_segments=args.num_seq_segments,
        fixed_second=args.fixed_second,
        sp_method=args.sp_method,
        p_ts=args.p_ts,
        p_ps=args.p_ps,
        p_tm=args.p_tm,
        p_tstr=args.p_tstr,
        semitone_range=args.semitone_range,
        tm_min_band_part=args.tm_min_band_part,
        tm_max_band_part=args.tm_max_band_part,
        tm_fade=args.tm_fade,
        amp_name=args.amp_name,
        loading_mode="pairs",
    )


    to_spec = nnAudio.features.MelSpectrogram(
        sr=args.sample_rate,
        n_fft=args.n_fft,
        win_length=args.window_size,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        center=True,
        power=2,
        verbose=False,
    ).to(device)


    mean, std = compute_mean_std(dataset, device, to_spec)

    print(f"Mean: {mean}, Std: {std}")
