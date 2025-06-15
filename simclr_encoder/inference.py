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
from dataset import NSynthDataModule, NSynthDataset

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
    print("-" * 100)

    # Get dataset and data module
    dm = NSynthDataModule(args=args)
    dm.setup(stage="test") #dm.setup()

    # Load best model
    model = SimCLR_pl.load_from_checkpoint(
        "model_dict/best_model-v6.ckpt",
        args=args, device=device,
    )
    model.eval()
    model.to(device)




    # Test whether the model can read the data and inference
    all_outputs = []
    with torch.no_grad():
        for batch in dm.test_dataloader():
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(device)
            outputs = model(inputs, return_embedding=True)
            all_outputs.append(outputs)
