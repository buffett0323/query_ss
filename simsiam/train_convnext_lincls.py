# Modified version of your SimSiam training script for single-GPU only

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from collections import OrderedDict
from utils import yaml_config_hook, AverageMeter
from model import SimSiam
from augmentation import PrecomputedNorm
from dataset import SegmentBPDataset
from conv_next import ConvNeXt


best_acc1 = 0

def main():
    parser = argparse.ArgumentParser(description="SimSiam Inference - Single GPU")
    config = yaml_config_hook("config/ssbp_convnext.yaml")
    for k, v in config.items(): 
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    """ Config settings """
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    args.pretrained = '/mnt/gestalt/home/buffett/simsiam_model_dict/convnext_model_dict_0422/checkpoint_0499.pth.tar'
    train_batch_size = 64 # 4096
    
    
    # build model
    print("=> Creating model with backbone encoder: '{}'".format(args.encoder_name))
    model = ConvNeXt(
        in_chans=args.channels,
        num_classes=args.dim,
        depths=[3, 3, 27, 3], #args.depths,
        dims=[128, 256, 512, 1024], #args.dims,
    )
    
    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['head.weight', 'head.bias']:
            param.requires_grad = False

            
    # init the fc layer
    model.head.weight.data.normal_(mean=0.0, std=0.01)
    model.head.bias.data.zero_()
    

    # load from pre-trained
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.encoder') and not k.startswith('module.encoder.head'):
                    # remove prefix
                    state_dict[k[len("module.encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            model.load_state_dict(state_dict, strict=False)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    model = model.to(device)

    # Print requires_grad status for each layer
    print("\nLayer-wise requires_grad status:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.requires_grad}")
    

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.lars:
        print("=> use LARS optimizer.")
        from larc import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)


    cudnn.benchmark = True


    # Data loading code
    train_dataset = SegmentBPDataset(
        data_dir=args.seg_dir,
        split="train",
        stem="other",
        eval_mode=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        SegmentBPDataset(
            data_dir=args.seg_dir,
            split="valid",
            stem="other",
            eval_mode=False,
        ),
        batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    
    os.makedirs(args.model_lincls_save_path, exist_ok=True)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, args.lr, epoch, args)
        train(train_loader, model, criterion, optimizer, epoch, device, args)
        acc1 = validate(val_loader, model, criterion, device, args)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 
                filename=f'checkpoint_{epoch:04d}.pth.tar', 
                save_dir=args.model_lincls_save_path
            )

def train(train_loader, model, criterion, optimizer, epoch, device, args):
    model.eval()
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(images)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(val_loader, model, criterion, device, args):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    with torch.no_grad():
        for images, target in val_loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(images)
            acc1, _ = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
    print(f"Validation Accuracy: {top1.avg:.2f}%")
    return top1.avg

def save_checkpoint(state, filename, save_dir):
    torch.save(state, os.path.join(save_dir, filename))

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()