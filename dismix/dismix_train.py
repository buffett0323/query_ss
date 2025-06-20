import os
import wandb
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from functools import partial

from dismix_model import DisMixModel
from dataset import MusicalObjectDataModule, spec_crop # CocoChoralesTinyDataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
torchvision.disable_beta_transforms_warning()
torch.set_float32_matmul_precision('high')


if __name__ == "__main__":
    # Initial settings
    log_wandb = True # False
    wanbd_proj_name = "P+T Enc (ELBO+BCE)" # + Pitch prior"
    find_unused_parameters = True # False if train all params
    device_id = [0, 1, 2, 3] #[0, 1, 2, 3 , 4, 5]
    batch_size = 64 #32
    num_frames = 10 #32
    dropout_rate = 0 #0.3
    lr = 4e-4
    early_stop_patience = 100 #260000
    best_val_loss = float('inf')
    max_steps = 10000000
    comp_path = "/home/buffett/NAS_NTU"
    root = f"{comp_path}/MusicSlots/data/jsb_multi"
    os.environ["WANDB_MODE"] = "online"

    # Initialize data module
    dm = MusicalObjectDataModule(
        root=root,
        batch_size=batch_size,
        num_workers=24,
    )


    img_transforms = [transforms.Lambda(partial(spec_crop, height=128, width=num_frames))]

    train_transforms = transforms.Compose(img_transforms)
    test_transforms = transforms.Compose(img_transforms)

    dm.train_transforms = train_transforms
    dm.test_transforms = test_transforms
    dm.val_transforms = test_transforms


    # Models and other settings
    model = DisMixModel(
        batch_size=batch_size,
        dropout_rate=dropout_rate,
        num_frames=num_frames,
        input_dim=128,
        latent_dim=64,
        hidden_dim=256,
        gru_hidden_dim=256,
        pitch_classes=52,
        output_dim=128,
        learning_rate=lr,
        num_layers=2,
        clip_value=0.5,
    )


    # Log model and hyperparameters in wandb
    if log_wandb:
        project = "Dismix_jsb_multi"
        name = wanbd_proj_name
        save_dir = '/data/buffett' if os.path.exists('/data/buffett') else '.'
        wandb_logger = WandbLogger(
            project=project,
            name=name,
            save_dir=save_dir,
            log_model=False,  # Avoid logging full model files to WandB
        )
    else:
        wandb_logger = None


    # GPU Accelerator Settings
    accelerator = "gpu"
    if str(-1) in device_id:
        devices = -1
        strategy = DDPStrategy(find_unused_parameters=find_unused_parameters)
    else:
        devices = [int(i) for i in device_id]
        if len(devices) == 1:
            strategy = "auto"
        else:
            strategy = DDPStrategy(find_unused_parameters=find_unused_parameters)


    # Other settings
    cb = [TQDMProgressBar(refresh_rate=10)]
    model_ckpt = ModelCheckpoint(monitor="val_loss", mode="min")
    cb.append(model_ckpt)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=early_stop_patience,
        verbose=True,
        mode="min",
    )
    cb.append(early_stop_callback)


    # Trainer settings
    trainer = Trainer(
        max_steps=max_steps,
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        strategy=strategy,
        callbacks=cb,
        precision=32,#'16-mixed',
        gradient_clip_val=model.clip_value,
    )

    trainer.fit(model, dm)
    # trainer.test(model.load_from_checkpoint(model_ckpt.best_model_path), datamodule=dm)
