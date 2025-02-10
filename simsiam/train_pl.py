import os
import wandb
import argparse
import atexit

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.strategies import DeepSpeedStrategy

from utils import yaml_config_hook
from model import SimSiamPL
from dataset import BPDataModule

torch.set_float32_matmul_precision('medium')


def main():
    # Loading args
    parser = argparse.ArgumentParser(description="SimSiam")

    config = yaml_config_hook("ssbp_pl_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    # Loading Wandb logger
    if args.log_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project_name, 
            name=args.wandb_name,
            save_dir='/data/buffett' if os.path.exists('/data/buffett') else '.', 
            log_model=False,  # Avoid logging full model files to WandB
        )
    else:
        wandb_logger = None

    # Initialize DataModule
    dm = BPDataModule(
        args=args,
        data_dir=args.data_dir, 
    )

    # Initialize Model
    args.lr = args.lr * args.batch_size / 256 # init lr
    model = SimSiamPL(
        args=args,
        dim=args.dim,
        pred_dim=args.pred_dim,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_dict_save_dir,  # Directory to save checkpoints
        filename=args.ckpt_name,  # Filename for the best model
        save_top_k=1,  # Only keep the best model
        save_last=True,
        verbose=True,
        monitor="val_loss",  # Metric to monitor
        mode="min",  # Save model with the minimum validation loss
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=args.early_stop_patience,
        verbose=True,
        mode="min",
    )

    # Trainer
    strategy = DeepSpeedStrategy(logging_batch_size_per_gpu=args.batch_size)
    trainer = Trainer(
        max_epochs=args.epochs,
        devices=args.gpu_ids, #gpus=args.number_of_gpus,
        strategy=strategy, #'ddp',  # Distributed Data Parallel
        accelerator="gpu",
        sync_batchnorm=True,
        check_val_every_n_epoch=args.check_val_every_n_epoch,  # Perform validation every 2 epochs
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=10), 
            checkpoint_callback, 
            early_stop_callback,
        ],
    )
    # Train
    print("-------Start Training-------")
    trainer.fit(model, dm)
    print("-------Finish Training-------")
    
    # # Test best model
    # best_model_path = checkpoint_callback.best_model_path
    # print(f"Best Model Path: {best_model_path}")
    # model = SimSiamPL.load_from_checkpoint(best_model_path, args=args)
    # model.eval()
    # trainer.test(model, dm)

    # Finish WandB
    wandb.finish()

    def cleanup():
        if dist.is_initialized():
            dist.destroy_process_group()
            print("🔥 Process group destroyed successfully.")

    atexit.register(cleanup)

if __name__ == "__main__":
    main()
