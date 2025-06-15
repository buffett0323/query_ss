import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from query_ss.qbss.htdemucs_qss import Query_HTDemucs

from query_ss.qbss.loss import L1SNR_Recons_Loss
from utils import _load_config
from query_ss.qbss.metrics import MetricHandler
from models.types import InputType, OperationMode, SimpleishNamespace
from data.moisesdb.datamodule import (
    MoisesTestDataModule,
    MoisesValidationDataModule,
    MoisesDataModule,
    MoisesBalancedTrainDataModule,
    MoisesVDBODataModule,
)

"""
    Dataset Structure:
        - estimates (predicted)
            - target
                - audio V
        - mixture
            - audio V
            - spectrogram X (Not stored)
        - sources
            - target
                - audio V
                - spectrogram V
        - query
            - audio V
        - masks
            - pred V
            - ground_truth V
        - metadata
"""

class Q_HTD_MODEL(LightningModule):
    def __init__(self, model, config, datamodule, stems, criterion, batch_size, lr=1e-3):
        super().__init__()
        self.model = model
        self.config = config
        self.datamodule = datamodule
        self.stems = stems
        self.criterion = criterion
        self.batch_size = batch_size
        self.lr = lr

        self.val_metric_handler = MetricHandler(stems)
        self.test_metric_handler = MetricHandler(stems)
        self.min_val_loss = 1e10

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        batch = InputType.from_dict(batch)
        batch = self._to_device(batch)

        batch = self(batch)
        loss = self.criterion(batch)

        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = InputType.from_dict(batch)
        batch = self._to_device(batch)

        batch = self(batch)
        val_loss = self.criterion(batch)

        self.val_metric_handler.calculate_snr(
            batch.estimates.target.audio,
            batch.sources.target.audio,
            batch.metadata.stem
        )
        self.log("val_loss", val_loss.item(), sync_dist=True, batch_size=self.batch_size)
        return val_loss


    def on_validation_epoch_end(self):
        val_snr = self.val_metric_handler.get_mean_median()
        self.log_dict(val_snr, prog_bar=True, sync_dist=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def test_step(self, batch, batch_idx):
        batch = InputType.from_dict(batch)
        batch = self._to_device(batch)

        batch = self(batch)
        test_loss = self.criterion(batch)

        self.test_metric_handler.calculate_snr(
            batch.estimates.target.audioc,
            batch.sources.target.audio,
            batch.metadata.stem
        )
        self.log("test_loss", test_loss.item(), sync_dist=True, batch_size=self.batch_size)
        return test_loss

    def on_test_epoch_end(self):
        test_snr = self.test_metric_handler.get_mean_median()
        self.log_dict(test_snr, prog_bar=True, sync_dist=True, batch_size=self.batch_size)

    def _to_device(self, batch):
        batch.mixture.audio = batch.mixture.audio.to(self.device)
        batch.sources.target.audio = batch.sources.target.audio.to(self.device)
        batch.query.audio = batch.query.audio.to(self.device)
        return batch




if __name__ == "__main__":
    # Load configuration
    config = _load_config("config/train.yml")
    stems = config.data.train_kwargs.allowed_stems
    print("Training with stems: ", stems)

    devices_id = [0] #[0, 1, 2, 3]
    wandb_use = True # False
    batch_size = 4
    lr = 1e-3
    num_epochs = 500

    if wandb_use:
        wandb_logger = WandbLogger(
            project="Query_ss",
            log_model="all",
            name="MultiGPU HTDemucs"
        )


    datamodule = MoisesDataModule(
        data_root=config.data.data_root,
        batch_size=batch_size,
        num_workers=config.data.num_workers,
        train_kwargs=config.data.get("train_kwargs", None),
        val_kwargs=config.data.get("val_kwargs", None),
        test_kwargs=config.data.get("test_kwargs", None),
        datamodule_kwargs=config.data.get("datamodule_kwargs", None),
    )

    # Instantiate the enrollment model
    query_model = Query_HTDemucs(
        num_sources=1
    )


    # Lightning model wrapper
    lightning_model = Q_HTD_MODEL(
        model=query_model,
        config=config,
        datamodule=datamodule,
        stems=stems,
        criterion=L1SNR_Recons_Loss(), #F.l1_loss,
        batch_size=batch_size,
        lr=1e-3,
    )


    # Callbacks
    call_backs = [
        TQDMProgressBar(refresh_rate=1),
        EarlyStopping(monitor="val_loss", patience=4, mode="min"),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename="q_htd-best-ckp-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
        ),
    ]


    # Trainer
    trainer = Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=devices_id,
        logger=wandb_logger if wandb_use else None,
        callbacks=call_backs,
        strategy="ddp",
    )

    # Train
    trainer.fit(lightning_model, datamodule=datamodule)

    # Test
    trainer.test(lightning_model, datamodule=datamodule)

    if wandb_use: wandb.finish()
