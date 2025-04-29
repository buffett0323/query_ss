# Modified version of your SimSiam training script for single-GPU only

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import nnAudio.features
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import wandb

from collections import OrderedDict
from utils import yaml_config_hook, AverageMeter
from model import SimSiam
from augmentation import PrecomputedNorm, NormalizeBatch
from dataset import SegmentBPDataset
from tqdm import tqdm


class LinearClassifier(nn.Module):
    def __init__(self, in_features=2048, num_classes=1000):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)  # 1000 classes for ImageNet

    def forward(self, x):
        return self.fc(x)



# Training loop for the linear classifier
def train_linear_classifier(lin_cls, train_loader, device, to_spec, pre_norm, post_norm, model, criterion, optimizer):
    lin_cls.train()  # Set classifier to training mode
    total = 0
    correct = 0
    losses = []
    
    for x, labels, _ in train_loader: #tqdm(train_loader, desc="Train Loader"):
        optimizer.zero_grad()
        
        x = x.to(device)
        labels = labels.to(device)
        
        x = (to_spec(x) + torch.finfo().eps).log()
        x = pre_norm(x).unsqueeze(1)
        x = post_norm(x)
        
        features = model.encoder(x)  # Extract features from the frozen backbone
        features = F.normalize(features, dim=1)
        
        outputs = lin_cls(features)  # Get predictions from the linear classifier
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()
        optimizer.step()  # Update classifier weights
        
        # Calculate accuracy and store loss
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        losses.append(loss.item())
        
    acc = 100. * correct / total
    return acc, losses
        

def evaluate(lin_cls, eval_loader, device, to_spec, pre_norm, post_norm, model, criterion):
    lin_cls.eval()  # Set classifier to evaluation mode
    total = 0
    correct = 0
    losses = []
    
    with torch.no_grad():
        for x, labels, _ in eval_loader:
            x = x.to(device)
            labels = labels.to(device)
            
            x = (to_spec(x) + torch.finfo().eps).log()
            x = pre_norm(x).unsqueeze(1)
            x = post_norm(x)
            
            features = model.encoder(x)
            features = F.normalize(features, dim=1)
            
            outputs = lin_cls(features)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100. * correct / total
    avg_loss = np.mean(losses)
    return acc, avg_loss



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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.pretrained = '/mnt/gestalt/home/buffett/simsiam_model_dict/convnext_model_dict_0423/checkpoint_0199.pth.tar'
    train_batch_size = 64 # 4096
    epochs = 1000 #1000
    wandb_log = True
    split = "test"
    # wandb
    if wandb_log:
        wandb.init(
            project="simsiam_lincls",
            name=f"convnext_lincls_{split}_0423_ckpt_0199",
            config=vars(args),
        )
    
    # build model
    print("=> Creating model with backbone encoder: '{}'".format(args.encoder_name))
    model = SimSiam(
        args=args,
        dim=args.dim,
        pred_dim=args.pred_dim,
    ).to(device)
    
    checkpoint = torch.load(args.pretrained)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()  # Set model to evaluation mode
    model.requires_grad_(False) 
    cudnn.benchmark = True

    
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
    ).to(device)


    # Normalization: PrecomputedNorm
    pre_norm = PrecomputedNorm(np.array(args.norm_stats)).to(device)
    post_norm = NormalizeBatch().to(device)

    # Dataset settings
    train_dataset = SegmentBPDataset(
        data_dir=args.seg_dir,
        split=split,
        stem="other",
        eval_mode=True,
        train_mode=args.train_mode,
    )
    
    eval_dataset = SegmentBPDataset(
        data_dir=args.seg_dir,
        split=split,
        stem="other",
        eval_mode=True,
        train_mode=args.train_mode,
        eval_id=1,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=train_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # Get linear classifier
    print(f"In Linear Classifier, num_classes: {len(train_dataset.label_dict)}")
    lin_cls = LinearClassifier(
        in_features=1024, 
        num_classes=len(train_dataset.label_dict)
    ).to(device)
    
    
    # Define optimizer and loss function
    optimizer = torch.optim.SGD(lin_cls.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Train the classifier
    for epoch in range(epochs): #, desc="Epochs"):
        acc, losses = train_linear_classifier(
            lin_cls, train_loader, device, to_spec, 
            pre_norm, post_norm, model, criterion, optimizer
        )
        avg_loss = np.mean(losses)
        print(f"TRAIN -- Epoch {epoch} -- Accuracy: {acc}% -- Loss: {avg_loss}")
        
        if epoch % 10 == 0:
            # evaluate
            acc, avg_loss = evaluate(
                lin_cls, eval_loader, device, to_spec, 
                pre_norm, post_norm, model, criterion
            )
            print(f"EVAL -- Epoch {epoch} -- Accuracy: {acc}% -- Loss: {avg_loss}")
            
            if wandb_log:   
                wandb.log({
                    "epoch": epoch,
                    "eval_accuracy": acc,
                    "eval_loss": avg_loss
                })
        
        # Log metrics to wandb
        if wandb_log:   
            wandb.log({
                "epoch": epoch,
                "train_accuracy": acc,
                "train_loss": avg_loss
            })
    
    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    main()