import os 
import argparse
import torch
import warnings
import torchvision.models as models
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm


# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, GradientAccumulationScheduler
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from utils import yaml_config_hook
from dataset import BPDataset
from transforms import CLARTransform
import simsiam.builder

torch.set_float32_matmul_precision('high')
warnings.filterwarnings(
    "ignore",
    message="`training_step` returned `None`. If this was on purpose, ignore this warning...",
    category=UserWarning,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimSiam")

    config = yaml_config_hook("config/ssbp_resnet50.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    

    
    # Get dataset and data module
    memory_dataset = BPDataset(
        sample_rate=args.sample_rate, 
        segment_second=args.segment_second, 
        piece_second=args.piece_second,
        data_dir=args.data_dir,
        augment_func=CLARTransform(
            sample_rate=args.sample_rate,
            duration=int(args.piece_second),
        ),
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        split="train",
        melspec_transform=args.melspec_transform,
        data_augmentation=args.data_augmentation,
        random_slice=args.random_slice,
        stems=['other'],
    )
    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = simsiam.builder.SimSiam(
        models.__dict__[args.arch], args,
        args.dim, args.pred_dim)

    # Load the saved checkpoint dictionary
    checkpoint = torch.load('model_dict/checkpoint_0100.pth.tar')  # Replace with the actual filename

    # Create a new state_dict without 'module.' prefix
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        new_key = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = v

    # Load the modified state_dict into model
    model.load_state_dict(new_state_dict)

    model.eval()
    feature_bank = []
    with torch.no_grad():
        for x1, x2 in tqdm(memory_loader, desc='Feature extracting'):
            feature = model(x1.to(device))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            
        feature_bank = torch.cat(feature_bank, dim=0)
        feature_bank_list = [torch.zeros_like(feature_bank) for _ in range(args.world_size)]
        
        torch.distributed.all_gather(feature_bank_list, feature_bank)
        feature_bank = torch.cat(feature_bank_list, dim=0)  # [N, D]