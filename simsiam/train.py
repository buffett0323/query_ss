"""
This is the script revised from the official code -- main_simsiam_knn.py
But it fails to visualize the training figures.
"""

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import wandb

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data

from utils import yaml_config_hook, AverageMeter, ProgressMeter
from model import SimSiam
from dataset import BPDataModule, BPDataset


torch.set_float32_matmul_precision('high')

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message="`training_step` returned `None`. If this was on purpose, ignore this warning...",
    category=UserWarning,
)


def main():
    # Loading args
    parser = argparse.ArgumentParser(description="SimSiam")

    config = yaml_config_hook("ssbp_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Loading Wandb logger
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_name,
            config=args,
        )
        
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
    
    # Multiprocess
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
        
    wandb.finish()
    
    
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
                                world_size=args.world_size, rank=args.rank, 
                                group_name="my_ddp_group") # Added group name
        torch.distributed.barrier()
    
    
    # create model
    print("=> Creating model with backbone encoder: '{}'".format(args.encoder_name))
    model = SimSiam(
        args=args,
        dim=args.dim,
        pred_dim=args.pred_dim,
    )

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

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
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=args.find_unused_parameters)

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

    
    # Loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    args.resume = os.path.join(args.model_dict_save_dir, args.resume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    # Data Module for loading dataset
    dm = BPDataModule(args=args, data_dir=args.data_dir)
    dm.setup()
    
    
    # Validation
    best_val_loss = 999999.0
    patience = 0
    
    # Training loops
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            dm.train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # Train for one epoch
        train_loss = train(dm.train_dataloader(), model, criterion, optimizer, epoch, args)

        if args.log_wandb:
            wandb.log({"train_loss_epoch": train_loss, "epoch": epoch})

        # Validation
        is_best = False
        if epoch % args.check_val_every_n_epoch == 0:
            val_loss = evaluate(dm.val_dataloader(), model, criterion, epoch, args)
            if args.log_wandb:
                wandb.log({"val_loss_epoch": val_loss, "epoch": epoch})
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                is_best = True
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    break
            
        # Save checkpoints
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_dir = args.model_dict_save_dir
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, 
                is_best=is_best, 
                filename=f'{save_dir}/checkpoint_{epoch}.pth.tar',
                save_dir=save_dir,
            )
        
    

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x_i, x_j) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            x_i = x_i.cuda(args.gpu, non_blocking=True)
            x_j = x_j.cuda(args.gpu, non_blocking=True)

        # compute output and loss
        p1, p2, z1, z2 = model(x1=x_i, x2=x_j)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        losses.update(loss.item(), x_i.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            progress.display(i)
            
            if args.log_wandb:
                wandb.log({"train_loss_step": loss.item(), "step": epoch * len(train_loader) + i})


    return losses.avg



def evaluate(valid_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(valid_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}] (Validation)".format(epoch))

    # switch to evaluation mode
    model.eval()

    end = time.time()
    with torch.no_grad():  # Disable gradient computation for validation
        for i, (x_i, x_j) in enumerate(valid_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                x_i = x_i.cuda(args.gpu, non_blocking=True)
                x_j = x_j.cuda(args.gpu, non_blocking=True)

            # compute output and loss
            p1, p2, z1, z2 = model(x1=x_i, x2=x_j)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

            losses.update(loss.item(), x_i.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    print(f"Validation Loss: {losses.avg:.4f}")

    return losses.avg  # Return the average validation loss


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def save_checkpoint(state, is_best, filename, save_dir):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))
    
    
if __name__ == "__main__":
    main()