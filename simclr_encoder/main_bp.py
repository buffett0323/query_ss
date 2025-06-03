import os
import argparse
import torch
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, GradientAccumulationScheduler
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from utils import yaml_config_hook
from model import SimCLR_pl
from dataset import BPDataModule #NSynthDataModule

torch.set_float32_matmul_precision('high')
warnings.filterwarnings(
    "ignore",
    message="`training_step` returned `None`. If this was on purpose, ignore this warning...",
    category=UserWarning,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR Encoder")

    config = yaml_config_hook("bp_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    if args.log_wandb:
        project = "SimCLR_BP"
        name = args.wandb_name
        save_dir = '/data/buffett' if os.path.exists('/data/buffett') else '.'
        wandb_logger = WandbLogger(
            project=project,
            name=name,
            save_dir=save_dir,
            log_model=False,  # Avoid logging full model files to WandB
        )
    else:
        wandb_logger = None


    dm = BPDataModule(
        args=args,
        data_dir=args.data_dir,
    )
    model = SimCLR_pl(args, device)


    # Callbacks
    accumulator = GradientAccumulationScheduler(scheduling={0: args.gradient_accumulation_steps})

    model_ckpt = ModelCheckpoint(
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

    trainer = Trainer(
        max_epochs=args.max_epochs,
        devices=args.gpu_ids,
        accelerator="gpu",
        sync_batchnorm=True,
        check_val_every_n_epoch=args.check_val_every_n_epoch,  # Perform validation every 2 epochs
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=10),
            accumulator,
            model_ckpt,
            early_stop_callback,
        ],
    )

    print("-------Start Training-------")
    trainer.fit(model, dm)

    print("-------Start Testing-------")
    best_model = SimCLR_pl.from_config(model_ckpt.best_model_path, args, device)
    trainer.test(best_model, datamodule=dm)
    # trainer.test(model.load_from_checkpoint(model_ckpt.best_model_path), datamodule=dm)

    # # if load best model
    # model.load_model(checkpoint_name="best_model.ckpt")
