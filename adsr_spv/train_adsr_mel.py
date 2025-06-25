import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path

from module import ADSRLightningModule
from dataset import Mel_ADSRDataModule


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config):
    torch.set_float32_matmul_precision('high')

    # Create main experiment directory
    experiment_dir = Path(config['wandb_dir']) / "experiments" / config['wandb_name']
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for different types of logs
    logs_dir = experiment_dir / "lightning_logs"
    checkpoints_dir = experiment_dir / "checkpoints"
    tensorboard_dir = experiment_dir / "tensorboard"

    logs_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    tensorboard_dir.mkdir(exist_ok=True)

    # Initialize wandb if enabled
    if config.get('wandb_use', False):
        wandb_logger = WandbLogger(
            project=config['wandb_project'],
            name=config['wandb_name'],
            save_dir=str(experiment_dir),  # Save wandb logs in experiment directory
            log_model=True
        )
        # Log config to wandb
        wandb_logger.experiment.config.update(config)
    else:
        wandb_logger = None

    # Initialize Lightning module and data module
    model = ADSRLightningModule(config)
    data_module = Mel_ADSRDataModule(config)

    # Setup callbacks
    callbacks = []

    # Model checkpoint callback - save every save_interval epochs
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename='checkpoint_epoch_{epoch:03d}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        every_n_epochs=config['save_interval'],
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Initialize trainer with custom logging directory
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        accelerator=config.get('accelerator', 'gpu'),
        devices=config.get('devices', [3]),
        # precision=config.get('precision', '16-mixed'),
        gradient_clip_val=config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        log_every_n_steps=config.get('log_every_n_steps', 50),
        callbacks=callbacks,
        logger=wandb_logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
        reload_dataloaders_every_n_epochs=0,
        sync_batchnorm=False,  # Disable for single GPU
        enable_checkpointing=True,
        strategy='auto',
        # Logging directory control
        default_root_dir=str(logs_dir),  # Lightning logs go here
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)

    print(f"Training completed!")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Lightning logs: {logs_dir}")
    print(f"Checkpoints: {checkpoints_dir}")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    # Load configuration
    config = load_config('configs/config_mel.yaml')
    main(config)
