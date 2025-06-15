import argparse
import math
import os
import random
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
from nsynth_dataset import NsynthDataset
from model import SimSiam
from augmentation import SequencePerturbation, PrecomputedNorm



torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    parser = argparse.ArgumentParser(description="SimSiam Single GPU")

    config = yaml_config_hook("config/nsynth_convnext.yaml")
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
    train_dataset = NsynthDataset(
        sample_rate=args.sample_rate,
        data_dir=args.data_dir,
        piece_second=args.piece_second,
        segment_second=args.segment_second,
        window_size=args.window_size,
        hop_length=args.hop_length,
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

    # Augmentation: SequencePerturbation
    tfms = SequencePerturbation(
        method=args.sp_method,
        sample_rate=args.sample_rate,
    ).cuda()

    # Normalization: PrecomputedNorm
    pre_norm = PrecomputedNorm(np.array(args.norm_stats)).cuda()


    # training loop
    os.makedirs(args.model_dict_save_path, exist_ok=True)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        train_loss = train(train_loader, model, criterion, optimizer, epoch, args, to_spec, tfms, pre_norm)

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


def train(train_loader, model, criterion, optimizer, epoch, args, to_spec, tfms, pre_norm):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses], prefix=f"Epoch: [{epoch}]")

    model.train()
    end = time.time()

    for i, (x, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # Mel-spec transform and normalize
        x = x.cuda(non_blocking=True)
        x = (to_spec(x) + torch.finfo().eps).log()
        x = pre_norm(x)

        # Augmentation
        x_i = tfms(x).unsqueeze(1)
        x_j = tfms(x).unsqueeze(1)

        p1, p2, z1, z2 = model(x1=x_i, x2=x_j)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

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
                wandb.log({
                    "train_loss_step": loss.item(),
                    "avg_std_train": avg_std,
                    "step": step
                })

    return losses.avg


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if param_group.get('fix_lr', False):
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = lr

def save_checkpoint(state, filename, save_dir):
    torch.save(state, os.path.join(save_dir, filename))

if __name__ == "__main__":
    main()
