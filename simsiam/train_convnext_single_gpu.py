import argparse
import math
import os
import random
import shutil
import time
import warnings
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models
import nnAudio.features
import numpy as np
from utils import yaml_config_hook, AverageMeter, ProgressMeter
from dataset import SegmentBPDataset, SimpleBPDataset
from model import SimSiam
from augmentation import SequencePerturbation, PrecomputedNorm, NormalizeBatch, Time_Freq_Masking



torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    parser = argparse.ArgumentParser(description="SimSiam Single GPU")

    config = yaml_config_hook("config/ssbp_convnext.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    if args.log_wandb:
        wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_name,
            notes=args.wandb_notes,
            config=vars(args),
        )

    # build model
    print("=> Creating SimSiam model with backbone encoder: '{}'".format(args.encoder_name))
    model = SimSiam(
        args=args,
        dim=args.dim,
        pred_dim=args.pred_dim,
    ).cuda()

    # criterion and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda()
    if args.fix_pred_lr:
        optim_params = [
            {'params': model.encoder.parameters(), 'fix_lr': False},
            {'params': model.predictor.parameters(), 'fix_lr': True}
        ]
    else:
        optim_params = model.parameters()

    init_lr = args.lr * args.batch_size / 256
    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    args.resume = os.path.join(args.model_dict_save_path, args.resume)
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    # dataset
    train_dataset = SegmentBPDataset(
        data_dir=args.seg_dir,
        split="train",
        stem="other",
        eval_mode=False,
        train_mode=args.train_mode,
        sample_rate=args.sample_rate,
        sp_method=args.sp_method,
        num_seq_segments=args.num_seq_segments,
        fixed_second=args.fixed_second,
        p_ts=args.p_ts,
        p_ps=args.p_ps,
        p_tm=args.p_tm,
        p_tstr=args.p_tstr,
        semitone_range=args.semitone_range,
        tm_min_band_part=args.tm_min_band_part,
        tm_max_band_part=args.tm_max_band_part,
        tm_fade=args.tm_fade,
        tstr_min_rate=args.tstr_min_rate,
        tstr_max_rate=args.tstr_max_rate,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
        persistent_workers=args.persistent_workers,
        prefetch_factor=8, #4,
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
    ).cuda()
    
    # Normalization: PrecomputedNorm
    pre_norm = PrecomputedNorm(np.array(args.norm_stats)).cuda()
    post_norm = NormalizeBatch().cuda()


    # training loop
    os.makedirs(args.model_dict_save_path, exist_ok=True)
    print(f"Training for {args.epochs} epochs in '{args.train_mode}' mode")
    
    for epoch in range(args.start_epoch, args.epochs):
        # Adjust learning rate
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        
        # Training
        train_loss, no_nan_exists = train(train_loader, model, criterion, optimizer, epoch, 
                           args, to_spec, pre_norm, post_norm)

        if args.log_wandb:
            wandb.log({"train_loss_epoch": train_loss, "epoch": epoch})

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, 
                filename=f'checkpoint_{epoch:04d}.pth.tar', 
                save_dir=args.model_dict_save_path
            )
        
        if not no_nan_exists:
            print(f"Found NaN in the model at epoch {epoch}")
            break

def train(train_loader, model, criterion, optimizer, epoch, args, to_spec, pre_norm, post_norm):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    no_nan_exists = True
    
    if args.train_mode == "augmentation":
        losses = AverageMeter('Loss', ':.4f')
        progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses], prefix=f"Epoch: [{epoch}]")
    
        model.train()
        end = time.time()
    
        for i, (x_i, x_j, _, _) in enumerate(train_loader):
            data_time.update(time.time() - end)
    
            x_i = x_i.cuda(non_blocking=True)
            x_j = x_j.cuda(non_blocking=True)
    
            # Mel-spec transform and normalize
            x_i = (to_spec(x_i) + torch.finfo().eps).log()
            x_j = (to_spec(x_j) + torch.finfo().eps).log()

            x_i = pre_norm(x_i).unsqueeze(1)
            x_j = pre_norm(x_j).unsqueeze(1)
    
            # Form a batch and post-normalize it.
            bs = x_i.shape[0]
            paired_inputs = torch.cat([x_i, x_j], dim=0)
            paired_inputs = post_norm(paired_inputs)
            
            # Forward pass
            p1, p2, z1, z2 = model(x1=paired_inputs[:bs], x2=paired_inputs[bs:])
            
            # Check for NaN values in z1 and z2
            if i % 100 == 0:
                print("z1:", z1)
                print("z2:", z2)
            
            if torch.isnan(z1).any() or torch.isnan(z2).any():
                no_nan_exists = False
                print(f"NaN detected in batch {i}")
                print("z1 contains NaN:", torch.isnan(z1).any().item())
                print("z2 contains NaN:", torch.isnan(z2).any().item())
                # Optional: print the full tensors to see where NaNs occur
                print("z1:", z1)
                print("z2:", z2)
                break
            
            # Calculate Loss
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
    
            # Calculate Avg_std_train
            z1_std = F.normalize(z1, dim=1).std(dim=0).mean()
            z2_std = F.normalize(z2, dim=1).std(dim=0).mean()
            avg_std = (z1_std + z2_std) / 2
    
            losses.update(loss.item(), x_i.size(0))
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            batch_time.update(time.time() - end)
            end = time.time()
    
            if i % args.print_freq == 0:
                progress.display(i)
                if args.log_wandb:
                    step = epoch * len(train_loader) + i
                    wandb.log({"train_loss_step": loss.item(), "avg_std_train": avg_std, "step": step})
    
        return losses.avg, no_nan_exists



    elif args.train_mode == "aug+sel":
        losses_pair1 = AverageMeter('Loss1', ':.4f')
        losses_pair2 = AverageMeter('Loss2', ':.4f')
        progress = ProgressMeter(
            len(train_loader), 
            [batch_time, data_time, losses_pair1, losses_pair2], 
            prefix=f"Epoch: [{epoch}]",
        )

        model.train()
        end = time.time()

        for i, (x_1, x_2, x_i, x_j, _) in enumerate(train_loader):
            data_time.update(time.time() - end)

            x_1 = x_1.cuda(non_blocking=True)
            x_2 = x_2.cuda(non_blocking=True)
            x_i = x_i.cuda(non_blocking=True)
            x_j = x_j.cuda(non_blocking=True)
            
            # Mel-spec transform and normalize
            x_1 = (to_spec(x_1) + torch.finfo().eps).log()
            x_2 = (to_spec(x_2) + torch.finfo().eps).log()
            x_i = (to_spec(x_i) + torch.finfo().eps).log()
            x_j = (to_spec(x_j) + torch.finfo().eps).log()

            x_1 = pre_norm(x_1).unsqueeze(1)
            x_2 = pre_norm(x_2).unsqueeze(1)
            x_i = pre_norm(x_i).unsqueeze(1)
            x_j = pre_norm(x_j).unsqueeze(1)
            
            # Form a batch and post-normalize it.
            bs = x_1.shape[0]
            paired_inputs = torch.cat([x_1, x_2, x_i, x_j], dim=0)
            paired_inputs = post_norm(paired_inputs)
            
            # Split the batch into 4 parts
            x_1, x_2 = paired_inputs[:bs], paired_inputs[bs:2*bs]
            x_i, x_j = paired_inputs[2*bs:3*bs], paired_inputs[3*bs:]
            
            # Augmentation
            # x_i = seq_pert(x_i).unsqueeze(1)
            # x_j = seq_pert(x_j).unsqueeze(1)


            # Pair 1: Different segments
            p1, p2, z1, z2 = model(x1=x_1, x2=x_2)  
            loss_pair1 = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            
            # Pair 2: Same segment but different augmentation
            p1, p2, z1, z2 = model(x1=x_i, x2=x_j)
            loss_pair2 = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

            # Calculate Avg_std_train
            z1_std = F.normalize(z1, dim=1).std(dim=0).mean()
            z2_std = F.normalize(z2, dim=1).std(dim=0).mean()
            avg_std = (z1_std + z2_std) / 2

            losses_pair1.update(loss_pair1.item(), x_1.size(0))
            losses_pair2.update(loss_pair2.item(), x_i.size(0))

            optimizer.zero_grad()
            loss_pair1.backward()
            loss_pair2.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            total_loss = loss_pair1.item() + loss_pair2.item()

            if i % args.print_freq == 0:
                progress.display(i)
                if args.log_wandb:
                    step = epoch * len(train_loader) + i
                    wandb.log({"train_loss_step": total_loss, "loss_1_segment": loss_pair1.item(), 
                            "loss_2_augmentation": loss_pair2.item(), "avg_std_train": avg_std, "step": step})

        return (losses_pair1.avg + losses_pair2.avg) / 2
    
    
    else:
        return 0
    


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Adjust the learning rate with warm-up for the first 20 epochs."""
    # Warm-up phase: Linearly increase learning rate from 0 to the base learning rate
    if epoch < args.warmup_epochs:
        lr = init_lr * (epoch + 1) / args.warmup_epochs  # Linear warm-up
    else:
        # After warm-up, use cosine decay
        lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    # Update learning rate for optimizer
    for param_group in optimizer.param_groups:
        if param_group.get('fix_lr', False):
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = lr
    
    # Log learning rate to WandB
    if args.log_wandb:
        wandb.log({"learning_rate": lr, "epoch": epoch})


def save_checkpoint(state, filename, save_dir):
    torch.save(state, os.path.join(save_dir, filename))



if __name__ == "__main__":
    main()