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
    print("=> Creating model with backbone encoder: '{}'".format(args.encoder_name))
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


    """ Augmentations """
    # Aug1. Time_Freq_Masking
    # tf_mask = Time_Freq_Masking(
    #     p_mask=args.p_mask,
    # ).cuda()
    
    # Aug2. SequencePerturbation
    seq_pert = SequencePerturbation(
        method=args.sp_method,
        sample_rate=args.sample_rate,
    ).cuda()
    
    
    # Normalization: PrecomputedNorm
    pre_norm = PrecomputedNorm(np.array(args.norm_stats)).cuda()
    post_norm = NormalizeBatch().cuda()


    # training loop
    os.makedirs(args.model_dict_save_path, exist_ok=True)
    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        train_loss = train(train_loader, model, criterion, optimizer, epoch, 
                           args, to_spec, seq_pert, pre_norm, post_norm)

        if args.log_wandb:
            wandb.log({"train_loss_epoch": train_loss, "epoch": epoch})

        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 
                filename=f'checkpoint_{epoch:04d}.pth.tar', 
                save_dir=args.model_dict_save_path
            )
            

def train(train_loader, model, criterion, optimizer, epoch, args, to_spec, seq_pert, pre_norm, post_norm):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
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
        print("Shape Check:", x_1.shape, x_2.shape, x_i.shape, x_j.shape)
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

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
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