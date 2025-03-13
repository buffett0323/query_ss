import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from dataset import MixedBPDataset, MixedBPDataModule
from model import SimSiam
from transforms import CLARTransform
from train_swint import save_checkpoint
from utils import yaml_config_hook, AverageMeter, ProgressMeter

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



class SimSiamLightning(pl.LightningModule):
    def __init__(
        self, 
        args,
    ):
        super().__init__()
        self.args = args
        self.init_lr = args.lr * args.batch_size / 256
        self.model = SimSiam(
            args=args,
            dim=args.dim,
            pred_dim=args.pred_dim,
        )
        self.criterion = nn.CosineSimilarity(dim=1)
        
        # Setting optimizer parameters
        if args.fix_pred_lr:
            optim_params = [{'params': self.model.encoder.parameters(), 'fix_lr': False},
                            {'params': self.model.predictor.parameters(), 'fix_lr': True}]
        else:
            optim_params = self.model.parameters()

        self.optimizer = torch.optim.SGD(optim_params, self.init_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        
        self.save_hyperparameters()


    def forward(self, x1, x2):
        p1, p2, z1, z2 = self.model(x1=x1, x2=x2)
        return p1, p2, z1, z2


    def on_train_epoch_start(self):
        self.adjust_learning_rate()
    
        # Fetch dataloader from the trainer's datamodule
        if self.trainer.datamodule is not None:
            dataloader_len = len(self.trainer.datamodule.train_dataloader())
        else:
            raise ValueError("Trainer has no datamodule. Make sure you have passed a valid LightningDataModule.")

        self.batch_time = AverageMeter('Time', ':6.3f')
        self.data_time = AverageMeter('Data', ':6.3f')
        self.losses = AverageMeter('Loss', ':.4f')
        self.progress = ProgressMeter(
            len(self.trainer.datamodule.train_dataloader()),
            [self.batch_time, self.data_time, self.losses],
            prefix="Epoch: [{}]".format(self.current_epoch))
        self.end = time.time()
        
        
        
    def training_step(self, batch, batch_idx):
        self.data_time.update(time.time() - self.end)
        
        x_i, x_j, _ = batch
        p1, p2, z1, z2 = self(x_i, x_j)
        loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
        avg_std = self.cal_std(z1, z2)
        self.losses.update(loss.item(), x_i.size(0))
        
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()
        
        if batch_idx % self.args.print_freq == 0:
            self.progress.display(batch_idx)
            
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_avg_std", avg_std, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    
    def on_train_epoch_end(self):
        if self.current_epoch % 10 == 0:
            checkpoint_path = f"checkpoint_{self.current_epoch}.ckpt"
            self.trainer.save_checkpoint(os.path.join(self.args.model_dict_save_dir, checkpoint_path))
            print(f"Checkpoint saved at {checkpoint_path}")
    
    
    def adjust_learning_rate(self):
        cur_lr = self.init_lr * 0.5 * (1. + math.cos(math.pi * self.current_epoch / self.args.epochs))
        for param_group in self.optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = self.init_lr
            else:
                param_group['lr'] = cur_lr


    def cal_std(self, z1, z2):
        z1_normalized = F.normalize(z1, dim=1)
        z2_normalized = F.normalize(z2, dim=1)
        z1_std = z1_normalized.std(dim=0).mean()
        z2_std = z2_normalized.std(dim=0).mean()
        return (z1_std + z2_std) / 2
    
    
    def configure_optimizers(self):
        return self.optimizer




def main():
    # Loading args
    parser = argparse.ArgumentParser(description="SimSiam_BP in PyTorch Lightning")
    config = yaml_config_hook("config/ssbp_swint.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    
    
    # Init settings
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    find_unused_parameters = True

    # WandB logger
    wandb_logger = WandbLogger(
        project=args.wandb_project_name,
        name=args.wandb_name,
        notes=args.wandb_notes,
        config=vars(args),  # Store args
    ) if args.log_wandb else None
    
    
    # GPU Accelerator Settings
    if str(-1) in args.gpu:
        devices = -1
        strategy = DDPStrategy(find_unused_parameters=find_unused_parameters)
    else:
        devices = [int(i) for i in args.gpu]
        if len(devices) == 1:
            strategy = "auto"
        else:
            strategy = DDPStrategy(find_unused_parameters=find_unused_parameters)
    
    
    # Other settings
    cb = [TQDMProgressBar(refresh_rate=args.print_freq)]
    
    
    # Initialize Lightning Model and DataModule
    model = SimSiamLightning(args)
    data_module = MixedBPDataModule(args)
    
    
    # Trainer with multi-GPU support
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        logger=wandb_logger,
        precision=16,  # Mixed precision training
        callbacks=cb,
    )
    
    trainer.fit(model, datamodule=data_module)

    
    
if __name__ == "__main__":
    main()