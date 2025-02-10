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
from model import SimSiamPL
from dataset import BPDataModule


torch.set_float32_matmul_precision('high')
warnings.filterwarnings(
    "ignore",
    message="`training_step` returned `None`. If this was on purpose, ignore this warning...",
    category=UserWarning,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimSiam")

    config = yaml_config_hook("ssbp_pl_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    
    print("Using device:", device)
    print("-" * 100)
    
    # Get dataset and data module
    dm = BPDataModule(
        args=args,
        data_dir=args.data_dir, 
    )
    dm.setup(stage="test") #dm.setup()


    
    def strip_prefix_from_state_dict(state_dict, prefix="_forward_module."):
        stripped_state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        return stripped_state_dict

    # Load checkpoint
    checkpoint_path = "./model_dict/BP_WGL128_CNN14.ckpt/consolidated_model.pt/pytorch_model.bin"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    stripped_state_dict = strip_prefix_from_state_dict(state_dict)

    # Load into model
    model = SimSiamPL(args=args).to(device)
    model.load_state_dict(stripped_state_dict)
    model.eval()

    
    # Example inference
    with torch.no_grad():
        input_tensor = torch.randn(1, 32000).to(device)
        output = model.encoder(input_tensor)
        print(output)