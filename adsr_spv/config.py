from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import torch

@dataclass
class TrainConfig:
    batch_size: int = 16
    num_workers: int = 8
    device: str = "cuda:3" if torch.cuda.is_available() else "cpu"

    # Paths
    metadata_dir: Path = Path("/mnt/gestalt/home/buffett/rendered_adsr_dataset_npy")
    data_dir: Path = Path("/mnt/gestalt/home/buffett/rendered_adsr_dataset_npy_new_mel")

    # Wandb
    wandb_use: bool = False # True
    wandb_dir: Path = Path("/mnt/gestalt/home/buffett/adsr_reg")
    wandb_project: str = "adsr-reg"
    wandb_name: str = "adsr 0622"

    # Model
    d_model: int = 1024
    d_out: int = 4 # A,D,S,R
    n_heads: int = 8 # 6
    n_layers: int = 16
    patch_size: int = 16
    patch_stride: int = 10
    input_channels: int = 1
    spec_shape: Tuple[int] = (128, 256)

    # Optim
    lr: float = 1e-4
    epochs: int = 100 #50

    # Loss weights
    param_weight: float = 1.0
    spectral_weight: float = 0.5  # set 0 to disable spectral loss
