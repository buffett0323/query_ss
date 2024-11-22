import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from typing import Optional
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from mir_eval.separation import bss_eval_sources
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger


from enrollment_model import MyModel
from loss import L1SNR_Recons_Loss, L1SNRDecibelMatchLoss
from utils import _load_config
from metrics import (
    AverageMeter, cal_metrics, safe_signal_noise_ratio, MetricHandler
)

from models.types import InputType, OperationMode, SimpleishNamespace
from data.moisesdb.datamodule import (
    MoisesTestDataModule,
    MoisesValidationDataModule,
    MoisesDataModule,
    MoisesBalancedTrainDataModule,
    MoisesVDBODataModule,
)


class QuerySSModel(LightningModule):
    def __init__(self, config, model, criterion, stems, lr=1e-3, gamma=0.98):
        super().__init__()
        self.config = config
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.gamma = gamma
        self.stems = stems
        self.train_metric_handler = MetricHandler(stems)
        self.val_metric_handler = MetricHandler(stems)
        self.test_metric_handler = MetricHandler(stems)

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        batch = InputType.from_dict(batch)
        batch = self._to_device(batch)
        batch = self.model(batch)
        loss = self.criterion(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = InputType.from_dict(batch)
        batch = self._to_device(batch)
        batch = self.model(batch)
        loss = self.criterion(batch)
        self.val_metric_handler.calculate_snr(batch.estimates["target"].audio, batch.sources["target"].audio, batch.metadata.stem)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        val_snr = self.val_metric_handler.get_mean_median()
        self.log_dict(val_snr, prog_bar=True)

    def test_step(self, batch, batch_idx):
        batch = InputType.from_dict(batch)
        batch = self._to_device(batch)
        batch = self.model(batch)
        loss = self.criterion(batch)
        self.test_metric_handler.calculate_snr(batch.estimates["target"].audio, batch.sources["target"].audio, batch.metadata.stem)
        return loss

    def on_test_epoch_end(self):
        test_snr = self.test_metric_handler.get_mean_median()
        self.log_dict(test_snr, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)
        return [optimizer], [scheduler]

    def _to_device(self, batch):
        batch.mixture.audio = batch.mixture.audio.to(self.device)
        batch.sources.target.audio = batch.sources.target.audio.to(self.device)
        batch.query.audio = batch.query.audio.to(self.device)
        return batch



if __name__ == "__main__":
    # Initialize components
    config_path = "config/train.yml"
    config = _load_config(config_path)
    stems = config.data.train_kwargs.allowed_stems
    devices_id = [0, 1, 2, 3]
    wandb_use = False # False
    batch_size = 2

    datamodule = MoisesDataModule(
        data_root=config.data.data_root,
        batch_size=batch_size,
        num_workers=config.data.num_workers,
        train_kwargs=config.data.get("train_kwargs", None),
        val_kwargs=config.data.get("val_kwargs", None),
        test_kwargs=config.data.get("test_kwargs", None),
        datamodule_kwargs=config.data.get("datamodule_kwargs", None),
    )

    model = MyModel(
        embedding_size=768,
        query_size=512,
        n_masks=1,
        mix_query_mode="Hyper_FiLM",
        q_enc="Passt",
    )

    criterion = L1SNR_Recons_Loss(mask_type="L1")
    query_ss_model = QuerySSModel(
        config=config, 
        model=model, 
        criterion=criterion, 
        stems=stems, 
        lr=1e-3,
    )


    # Logging and Callbacks
    if wandb_use:
        wandb_logger = WandbLogger(project="Query_ss")
    else:
        wandb_logger = None
        


    call_backs = [
        TQDMProgressBar(refresh_rate=1),
        EarlyStopping(monitor="val_loss", patience=4, mode="min"),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename="query_ss-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
        ),
    ]

    # Trainer
    trainer = Trainer(
        max_epochs=500,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        devices=devices_id,
        logger=wandb_logger,
        callbacks=call_backs,
    )

    # Train and Test
    trainer.fit(query_ss_model, datamodule=datamodule)
    trainer.test(query_ss_model, datamodule=datamodule)

    if wandb_use: wandb.finish()
