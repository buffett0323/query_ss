import argparse
import builtins
import random
import warnings
import wandb


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.models as models

from tqdm import tqdm
from model import SimSiam

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, GradientAccumulationScheduler
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from utils import yaml_config_hook
from dataset import BPDataset, MixedBPDataset
from transforms import CLARTransform
from train_swint_pl import SimSiamLightning

torch.set_float32_matmul_precision('high')
warnings.filterwarnings(
    "ignore",
    message="`training_step` returned `None`. If this was on purpose, ignore this warning...",
    category=UserWarning,
)

# knn monitor as in InstDisc http://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, knn_k=3, knn_t=1.0):
    """
    Perform KNN search to find the top 3 nearest neighbors.

    Args:
        feature (torch.Tensor): Query feature tensor of shape [B, D].
        feature_bank (torch.Tensor): Feature bank tensor of shape [D, N].
        knn_k (int): Number of nearest neighbors to retrieve (default: 3).
        knn_t (float): Temperature parameter for scaling similarity (default: 1.0).

    Returns:
        top_k_indices (torch.Tensor): Indices of the top-K nearest neighbors [B, knn_k].
        top_k_similarities (torch.Tensor): Cosine similarity scores of top-K neighbors [B, knn_k].
    """
    # Compute cosine similarity between feature and feature bank -> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)  # [B, N]

    # Retrieve top-K nearest neighbors
    top_k_similarities, top_k_indices = sim_matrix.topk(k=knn_k, dim=-1)

    # Apply temperature scaling
    top_k_similarities = (top_k_similarities / knn_t).exp()

    return top_k_indices, top_k_similarities

def main():
    parser = argparse.ArgumentParser(description="SimSiam Inference")

    config = yaml_config_hook("config/ssbp_swint.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # args.gpu = int(args.gpu[0]) if isinstance(args.gpu, list) else int(args.gpu)
    print("Using single GPU:", args.gpu)
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = torch.device(f"cuda:{gpu}")

    # Load trained LightningModule
    print(f"=> Loading model from: {args.resume_training_path}")
    model = SimSiamLightning.load_from_checkpoint(
        args.resume_training_path,
        args=args,
    )
    model = model.to(gpu)
    model.eval()

    # Datasets
    memory_dataset = MixedBPDataset(
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
        fmax=args.fmax,
        img_size=args.img_size,
        img_mean=args.img_mean,
        img_std=args.img_std,
    )

    test_dataset = MixedBPDataset(
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
        split="test",
        melspec_transform=args.melspec_transform,
        data_augmentation=args.data_augmentation,
        random_slice=args.random_slice,
        stems=['other'],
        fmax=args.fmax,
        img_size=args.img_size,
        img_mean=args.img_mean,
        img_std=args.img_std,
    )

    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=32, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    validate(memory_loader, test_loader, model, args)


def validate(memory_loader, test_loader, model, args):
    # Evaluation
    model.eval()
    feature_bank, feature_paths = [], []

    with torch.no_grad():
        # Extract features from memory_loader
        for x_i, x_j, path in tqdm(memory_loader, desc='Feature extracting'):
            if args.gpu is not None:
                x_i = x_i.cuda(args.gpu, non_blocking=True)
                x_j = x_j.cuda(args.gpu, non_blocking=True)

            _, _, feature, _ = model(x1=x_i, x2=x_j)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_paths.extend(path)

        feature_bank = torch.cat(feature_bank, dim=0)  # [N, D]

        # Gather feature_bank across all distributed processes if using DDP
        feature_bank_list = [torch.zeros_like(feature_bank) for _ in range(args.world_size)]
        # torch.distributed.all_gather(feature_bank_list, feature_bank)
        feature_bank = torch.cat(feature_bank_list, dim=0)  # [N, D]

        # Transpose to make feature_bank [D, N]
        feature_bank = feature_bank.t().contiguous()  # [D, N]
        print(f"Feature bank shape: {feature_bank.shape}")

        # Ensure feature paths match feature_bank after DDP
        if args.world_size > 1:
            feature_paths_list = [None] * args.world_size
            # torch.distributed.all_gather_object(feature_paths_list, feature_paths)
            feature_paths = sum(feature_paths_list, [])  # Flatten the list


        # Loop through test data to find top-3 nearest neighbors
        test_bar = tqdm(test_loader, desc='KNN Evaluation')
        with open('info/test_matches_swint_vox_others.txt', 'w') as f:
            for x_i, x_j, test_path in test_bar:
                if args.gpu is not None:
                    x_i = x_i.cuda(args.gpu, non_blocking=True)
                    x_j = x_j.cuda(args.gpu, non_blocking=True)

                _, _, feature, _ = model(x1=x_i, x2=x_j)
                feature = F.normalize(feature, dim=1)

                # Get top-3 nearest neighbors
                top_k_indices, _ = knn_predict(
                    feature, feature_bank, knn_k=args.knn_k, knn_t=args.knn_t
                )

                # Retrieve the paths of the top-3 nearest neighbors
                for i in range(len(test_path)):  # Iterate over batch
                    test_sample_path = test_path[i]
                    nearest_paths = [feature_paths[idx] for idx in top_k_indices[i].tolist()]

                    # Write to file
                    f.write("python npy2mp3.py --output_mp3 target.mp3 \\")
                    f.write(f"    {test_sample_path}\n")
                    for rank, neighbor_path in enumerate(nearest_paths, 1):
                        f.write(f"python npy2mp3.py --output_mp3 top{rank}.mp3 \\")
                        f.write(f"    {neighbor_path}\n")
                    f.write("\n")

                print(f"Test sample: {test_sample_path}")
                print(f"Top-3 Nearest Paths: {nearest_paths}")



if __name__ == "__main__":
    main()
