import argparse
import builtins
import os
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

import laion_clap
import torchaudio.functional as TF

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
    parser = argparse.ArgumentParser(description="SimSiam")

    config = yaml_config_hook("config/ssbp_resnet50.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Initial settings
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count() #len(args.gpu.split(',')) #
    print("ngpus:", ngpus_per_node)


    # Multiprocess
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    else:
        # Simply call main_worker function
        print("No Multiprocessing")
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args, **kwargs):  # Allow any extra keyword arguments
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
                                # group_name="my_ddp_group") # Added group name
        torch.distributed.barrier()


    # Loading model
    print("=> creating model '{}'".format(args.arch))
    model = simsiam.builder.SimSiam(
        models.__dict__[args.arch], args,
        args.dim, args.pred_dim)
    checkpoint = torch.load('/mnt/gestalt/home/buffett/simsiam_model_dict/resnet_model_dict/checkpoint_0199.pth.tar')  # Replace with the actual filename

    # Create a new state_dict without 'module.' prefix
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        new_key = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)

    cudnn.benchmark = True

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                              find_unused_parameters=args.find_unused_parameters)

        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")




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
        fmax=args.fmax,
        img_size=args.img_size,
        img_mean=args.img_mean,
        img_std=args.img_std,
    )
    test_dataset = BPDataset(
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

    if args.distributed:
        memory_sampler = torch.utils.data.distributed.DistributedSampler(memory_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        memory_sampler = None
        test_sampler = None

    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=64, shuffle=(memory_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=memory_sampler, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=True)


    # CLAP Model
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, device=args.gpu)
    clap_model.load_ckpt('/mnt/gestalt/home/buffett/clap_ckpt/630k-audioset-best.pt') # download the default pretrained checkpoint.


    validate(memory_loader, test_loader, model, args, clap_model)



def validate(memory_loader, test_loader, model, args, clap_model):
    # Evaluation
    model.eval()
    feature_bank, feature_paths = [], []
    feature_bank_clap, feature_paths_clap = [], []

    with torch.no_grad():
        # Extract features from memory_loader
        for x_i, x_j, x_i_audio, _, path in tqdm(memory_loader, desc='Feature extracting'):
            if args.gpu is not None:
                x_i = x_i.cuda(args.gpu, non_blocking=True)
                x_j = x_j.cuda(args.gpu, non_blocking=True)

            _, _, feature, _ = model(x1=x_i, x2=x_j)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_paths.extend(path)

            x_i_audio = TF.resample(x_i_audio, 16000, 48000)
            print("x_i_audio.shape:", x_i_audio.shape)
            audio_embed = clap_model.get_audio_embedding_from_data(x=x_i_audio, use_tensor=True)
            audio_embed = F.normalize(audio_embed, dim=1)
            print("audio_embed.shape:", audio_embed.shape)
            feature_bank_clap.append(audio_embed)
            feature_paths_clap.extend(path)

        feature_bank = torch.cat(feature_bank, dim=0)  # [N, D]
        feature_bank_clap = torch.cat(feature_bank_clap, dim=0)  # [N, D]

        # Gather feature_bank across all distributed processes if using DDP
        feature_bank_list = [torch.zeros_like(feature_bank) for _ in range(args.world_size)]
        torch.distributed.all_gather(feature_bank_list, feature_bank)
        feature_bank = torch.cat(feature_bank_list, dim=0)  # [N, D]

        feature_bank_clap_list = [torch.zeros_like(feature_bank_clap) for _ in range(args.world_size)]
        torch.distributed.all_gather(feature_bank_clap_list, feature_bank_clap)
        feature_bank_clap = torch.cat(feature_bank_clap_list, dim=0)  # [N, D]

        # Transpose to make feature_bank [D, N]
        feature_bank = feature_bank.t().contiguous()  # [D, N]
        feature_bank_clap = feature_bank_clap.t().contiguous()  # [D, N]
        print(f"Feature bank shape: {feature_bank.shape}")
        print(f"Feature bank clap shape: {feature_bank_clap.shape}")


        # Ensure feature paths match feature_bank after DDP
        if args.world_size > 1:
            feature_paths_list = [None] * args.world_size
            torch.distributed.all_gather_object(feature_paths_list, feature_paths)
            feature_paths = sum(feature_paths_list, [])  # Flatten the list

            feature_paths_clap_list = [None] * args.world_size
            torch.distributed.all_gather_object(feature_paths_clap_list, feature_paths_clap)
            feature_paths_clap = sum(feature_paths_clap_list, [])  # Flatten the list


        # Loop through test data to find top-3 nearest neighbors
        test_bar = tqdm(test_loader, desc='KNN Evaluation')
        with open('info/test_matches_resnet50_clap.txt', 'w') as f:
            for x_i, x_j, x_i_audio, _, test_path in test_bar:
                if args.gpu is not None:
                    x_i = x_i.cuda(args.gpu, non_blocking=True)
                    x_j = x_j.cuda(args.gpu, non_blocking=True)

                _, _, feature, _ = model(x1=x_i, x2=x_j)
                feature = F.normalize(feature, dim=1)

                audio_embed = clap_model.get_audio_embedding_from_data(x=x_i_audio, use_tensor=True)
                audio_embed = F.normalize(audio_embed, dim=1)

                # Get top-3 nearest neighbors
                top_k_indices, _ = knn_predict(
                    feature, feature_bank, knn_k=args.knn_k, knn_t=args.knn_t
                )
                top_k_indices_clap, _ = knn_predict(
                    audio_embed, feature_bank_clap, knn_k=args.knn_k, knn_t=args.knn_t
                )

                # Retrieve the paths of the top-3 nearest neighbors
                for i in range(len(test_path)):  # Iterate over batch
                    test_sample_path = test_path[i]
                    nearest_paths = [feature_paths[idx] for idx in top_k_indices[i].tolist()]
                    nearest_paths_clap = [feature_paths_clap[idx] for idx in top_k_indices_clap[i].tolist()]

                    # Write to file
                    f.write("python npy2mp3.py --output_mp3 target.mp3 \\")
                    f.write(f"    {test_sample_path}\n")
                    for rank, neighbor_path in enumerate(nearest_paths, 1):
                        f.write(f"python npy2mp3.py --output_mp3 top{rank}.mp3 \\")
                        f.write(f"    {neighbor_path}\n")
                    for rank, neighbor_path_clap in enumerate(nearest_paths_clap, 1):
                        f.write(f"python npy2mp3.py --output_mp3 top{rank}.mp3 \\")
                        f.write(f"    {neighbor_path_clap}\n")
                    f.write("\n")

                print(f"Test sample: {test_sample_path}")
                print(f"Top-3 Nearest Paths: {nearest_paths}")



if __name__ == "__main__":
    main()
