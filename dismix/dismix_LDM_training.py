import os
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

from dismix_LDM import DisMix_LDM, DisMix_LDM_Model
from dataset import CocoChoraleDataModule
from dismix_loss import ELBOLoss, BarlowTwinsLoss
from diffusers import AudioLDM2Pipeline

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
torch.set_float32_matmul_precision('high') 

# Initial settings
log_wandb = True # False
use_gpu = True
find_unused_parameters = True # False if train all params
device_id = [2] #[0, 1, 2, 3]
batch_size = 4
N_s = 4
lr = 1e-4
early_stop_patience = 100 #260000
max_steps = 10000000
comp_path = "/mnt/gestalt/home/ddmanddman"
root = f"{comp_path}/cocochorales_output/main_dataset"
os.environ["WANDB_MODE"] = "online"

# Initialize data module
dm = CocoChoraleDataModule(
    root=root,
    batch_size=batch_size,
    num_workers=8,
)

# Models and other settings
pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=torch.float16)#.to(device)
vae = pipe.vae
model = DisMix_LDM_Model(
    vae=vae,
    learning_rate=lr,
    N_s=N_s,
    batch_size=batch_size,
)


# Log model and hyperparameters in wandb
if log_wandb: 
    project = "Dismix_ldm"
    name = "Dismix_LDM_Training"
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
if use_gpu:
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
else:
    accelerator = "cpu"
    devices = 1
    strategy = "auto"


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
    precision='16-mixed' if use_gpu else 32,
    # gradient_clip_val=0.5,
)

# Sanity Check float 16
for name, param in model.named_parameters():
    if param.requires_grad and param.dtype == torch.float16:
        print(f"Layer: {name}, Grad: {param.requires_grad}, Dtype: {param.dtype}")
        assert param.dtype == torch.float32, f"Layer {name} parameters must be float32!"

trainer.fit(model, dm)
# trainer.test(model.load_from_checkpoint(model_ckpt.best_model_path), datamodule=dm)
