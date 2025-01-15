import os 
import argparse
import torch
import warnings
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from utils import yaml_config_hook
from model import ContrastiveLearning
from dataset import NSynthDataModule

torch.set_float32_matmul_precision('high')
warnings.filterwarnings(
    "ignore",
    message="`training_step` returned `None`. If this was on purpose, ignore this warning...",
    category=UserWarning,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR Encoder")

    config = yaml_config_hook("config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    
    if args.log_wandb: 
        project = "SimCLR_0109"
        name = "SimCLR_NSynth"
        save_dir = '/data/buffett' if os.path.exists('/data/buffett') else '.'
        wandb_logger = WandbLogger(
            project=project, 
            name=name, 
            save_dir=save_dir, 
            log_model=False,  # Avoid logging full model files to WandB
        )
    else:
        wandb_logger = None

    
    dm = NSynthDataModule(args=args)
    model = ContrastiveLearning(args, device)
    

    # Callbacks
    cb = [TQDMProgressBar(refresh_rate=10)]
    model_ckpt = ModelCheckpoint(
        dirpath=args.model_dict_save_dir,  # Directory to save checkpoints
        filename="best_model",  # Filename for the best model
        save_top_k=1,  # Only keep the best model
        verbose=True,
        monitor="val_loss",  # Metric to monitor
        mode="min",  # Save model with the minimum validation loss
    )
    cb.append(model_ckpt)
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=args.early_stop_patience,
        verbose=True,
        mode="min",
    )
    cb.append(early_stop_callback)

    trainer = Trainer(
        max_epochs=args.epoch_num,
        devices=args.gpu_ids,
        accelerator="gpu",
        sync_batchnorm=True,
        check_val_every_n_epoch=args.check_val_every_n_epoch,  # Perform validation every 2 epochs
        callbacks=cb,
        logger=wandb_logger,
    )

    print("-------Start Training-------")
    trainer.fit(model, dm)
    
    print("-------Start Testing-------")
    trainer.test(model.load_from_checkpoint(model_ckpt.best_model_path), datamodule=dm)
    
    # # if load best model    
    # cl.load_model(checkpoint_name="best_model.ckpt")
