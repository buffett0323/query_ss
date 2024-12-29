import argparse
import torch
import torchvision
import torchaudio
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

    train_dataset = BeatportDataset(
        dataset_dir=args.dataset_dir,
        split="train",
        transform=SimCLRTransform(),
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers)

    cl = ContrastiveLearning(args, device)
    
    trainer = Trainer(
        max_epochs=args.epoch_num,
        devices=1,
        accelerator="gpu",
        sync_batchnorm=True,
    )

    trainer.fit(cl, train_loader)