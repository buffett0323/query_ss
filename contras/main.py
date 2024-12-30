import os 
import argparse
import random
import torch
import torchaudio
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram

from utils import yaml_config_hook
from simclr import ContrastiveLearning
from dataset import BeatportDataset, SimCLRTransform

torch.set_float32_matmul_precision('high')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")

    config = yaml_config_hook("config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Train-Test Split
    npy_list = [
        os.path.join(args.npy_dir, folder_name, file_name)
        for folder_name in os.listdir(args.npy_dir)
            for file_name in os.listdir(os.path.join(args.npy_dir, folder_name))
                if file_name.endswith(".npy")
    ]
    random.shuffle(npy_list)
    n = len(npy_list)
    train_end = int(n * 0.8)  # 80%
    test_end = train_end + int(n * 0.1)  # 80% + 10%

    # Split the data
    train_data = npy_list[:train_end]
    test_data = npy_list[train_end:test_end]
    valid_data = npy_list[test_end:]

    print("Train dataset:", len(train_data))
    print("Valid dataset:", len(valid_data))
    print("Test dataset:", len(test_data))
    
    
    train_dataset = BeatportDataset(
        args=args,
        data_path_list=train_data,
        split="train",
    )
    
    valid_dataset = BeatportDataset(
        args=args,
        data_path_list=valid_data,
        split="valid",
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.workers)
    
    cl = ContrastiveLearning(args, device)
    
    trainer = Trainer(
        max_epochs=args.epoch_num,
        devices=args.gpu_ids,
        accelerator="gpu",
        sync_batchnorm=True,
        val_check_interval=5.0,
    )

    print("Starting training...")
    # trainer.fit(cl, train_loader, valid_loader)