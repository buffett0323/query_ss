#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import yaml
import wandb
import nnAudio
import numpy as np

# import deeplearning.cross_image_ssl.moco.builder
# import deeplearning.cross_image_ssl.moco.loader
# import moco.builder
# import moco.loader
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from model import MoCo
from utils import yaml_config_hook, AverageMeter, ProgressMeter
from dataset import SegmentBPDataset
from augmentation import PrecomputedNorm, NormalizeBatch


def main() -> None:
    parser = argparse.ArgumentParser(description="MoCoV2_BP")
    
    config = yaml_config_hook("config/moco_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    
    args = parser.parse_args()
    
    # Initial settings
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )


    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print("ngpus:", ngpus_per_node)
    
    # Distributed training
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        print("No Multiprocessing")
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)

    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ.get("RANK", 0))

    if args.multiprocessing_distributed:
        args.rank = args.rank * ngpus_per_node + gpu

    if args.multiprocessing_distributed and args.gpu != 0:
        builtins.print = lambda *args, **kwargs: None

    print(f"[rank{args.rank}] on GPU {args.gpu} — device = {torch.cuda.current_device()}")

    # Init process group
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier(device_ids=[args.gpu])  # ✅ Prevent NCCL unknown device warning
        
    
    # Only initialize WandB on the master process (rank 0)
    if args.rank == 0 and args.log_wandb:
        wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_name,
            notes=args.wandb_notes,
            config=vars(args),  # Store args
        )
    
    
    # create model
    print("=> Creating model '{}'".format(args.arch))
    model = MoCo(
        args, 
        dim=args.moco_dim, 
        K=args.moco_K, 
        m=args.moco_m, 
        T=args.moco_T, 
        mlp=args.moco_mlp
    )
    # print(model)

    if args.distributed:
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
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    train_dataset = SegmentBPDataset(
        data_dir=args.seg_dir,
        split="train",
        stem="bass_other",
        eval_mode=False,
        num_seq_segments=args.num_seq_segments,
        fixed_second=args.fixed_second,
        sp_method=args.sp_method,
        p_ts=args.p_ts,
        p_ps=args.p_ps,
        p_tm=args.p_tm,
        p_tstr=args.p_tstr,
        semitone_range=args.semitone_range,
        tm_min_band_part=args.tm_min_band_part,
        tm_max_band_part=args.tm_max_band_part,
        tm_fade=args.tm_fade,
        amp_name=args.amp_name,
    )
    

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    
    # MelSpectrogram
    to_spec = nnAudio.features.MelSpectrogram(
        sr=args.sample_rate,
        n_fft=args.n_fft,
        win_length=args.window_size,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        center=True,
        power=2,
        verbose=False,
    ).to(args.gpu)
    
    # Normalization: PrecomputedNorm
    pre_norm = PrecomputedNorm(np.array(args.norm_stats)).to(args.gpu)
    post_norm = NormalizeBatch().to(args.gpu)


    # Training loop
    os.makedirs(args.model_dict_save_dir, exist_ok=True)
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, args,\
            to_spec, pre_norm, post_norm)
        
        # Log only from master process
        if args.gpu == 0 and args.log_wandb:
            wandb.log({"train_loss_epoch": train_loss, "epoch": epoch})

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            if (epoch+1) % 10 == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=False,
                    filename="checkpoint_{:04d}.pth.tar".format(epoch),
                    save_dir=args.model_dict_save_dir,
                )
                
    if args.distributed:
        dist.destroy_process_group()


def train(train_loader, model, criterion, optimizer, epoch, args, to_spec, pre_norm, post_norm) -> None:
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    # top1 = AverageMeter("Acc@1", ":6.2f")
    # top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses], #], top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x_i, x_j, _, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            x_i = x_i.cuda(args.gpu, non_blocking=True)
            x_j = x_j.cuda(args.gpu, non_blocking=True)

        # Mel-spec transform and normalize
        x_i = (to_spec(x_i) + torch.finfo().eps).log()
        x_j = (to_spec(x_j) + torch.finfo().eps).log()

        x_i = pre_norm(x_i).unsqueeze(1)
        x_j = pre_norm(x_j).unsqueeze(1)
        
        # Form a batch and post-normalize it.
        bs = x_i.shape[0]
        paired_inputs = torch.cat([x_i, x_j], dim=0)
        paired_inputs = post_norm(paired_inputs)
        
        # compute output
        output, target = model(im_q=paired_inputs[:bs], im_k=paired_inputs[bs:])
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), x_i.size(0))
        # top1.update(acc1[0], images[0].size(0))
        # top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            
            # Log training loss per step only from GPU 0
            if args.gpu == 0 and args.log_wandb:
                wandb.log({"train_loss_step": loss.item(), "step": epoch * len(train_loader) + i})

    return losses.avg



def save_checkpoint(
    state, 
    is_best, 
    filename: str = "checkpoint.pth.tar",
    save_dir: str = "/mnt/gestalt/home/buffett/moco_model_dict/bass_other_new_amp08"
) -> None:
    
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))
        
        




def adjust_learning_rate(optimizer, epoch, args) -> None:
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.moco_cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr



if __name__ == "__main__":
    main()
