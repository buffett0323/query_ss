"""For Banquet Experiments"""
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from typing import Optional
from torch.utils.data import DataLoader
from mir_eval.separation import bss_eval_sources
from sklearn.model_selection import train_test_split

from bandit_model import MyBandSplit
from enrollment_model import MyModel
from load_data import BEATS_path, ORIG_mixture, ORIG_target, stems
from dataset import MusicDataset
from loss import L1SNR_Recons_Loss, L1SNRDecibelMatchLoss
from metrics import (
    AverageMeter, cal_metrics, safe_signal_noise_ratio, MetricHandler
)

from models.types import InputType, OperationMode, SimpleishNamespace


# Init settings
wandb_use = True # False
lr = 1e-3 #1e-4
num_epochs = 200 # 200
batch_size = 2 # 32
n_srcs = 2
n_sqm_modules = 8


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training on device:", device)


if wandb_use:
    wandb.init(
        project="Query_ss",
        config={
        "learning_rate": lr,
        "architecture": "Self-Bandquet",
        "dataset": "MoisesDB",
        "epochs": num_epochs,
        }
    )


# Get dataset
train_data, val_data, train_labels, val_labels = train_test_split(
    ORIG_mixture, ORIG_target, test_size=0.2, random_state=42)
val_data, test_data, val_labels, test_labels = train_test_split(
    val_data, val_labels, test_size=0.5, random_state=42)

# Create datasets
train_dataset = MusicDataset(train_data, BEATS_path, train_labels)
val_dataset = MusicDataset(val_data, BEATS_path, val_labels)
test_dataset = MusicDataset(test_data, BEATS_path, test_labels)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Instantiate the enrollment model
model = MyBandSplit(
    in_channel=n_srcs,
    stems=stems,
    n_sqm_modules=n_sqm_modules,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98) # optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)
criterion = L1SNRDecibelMatchLoss() #L1SNR_Recons_Loss()


early_stop_counter, early_stop_thres = 0, 4
min_val_loss = 1e10

# Training loop
for epoch in tqdm(range(num_epochs)):
    
    model.train()
    train_loss = 0.0
    for mixture, query_emb, target, stem in tqdm(train_loader):
        mixture = mixture.to(device)
        query_emb = query_emb.mean(axis=1).to(device)
        target = target.to(device)

        batch = InputType(
            mixture= SimpleishNamespace(
                audio=mixture, spectrogram=None
            ),
            sources={
                f"{stem}": SimpleishNamespace(
                    audio=target, spectrogram=None
                ),
            },
            query=SimpleishNamespace(
                audio=target, spectrogram=None
            ),
            estimates={
                "target": SimpleishNamespace(
                    audio=mixture, spectrogram=None
                ),
            },
        )
        

        optimizer.zero_grad()
        
        # Forward pass
        batch = model(batch)
    
        # Compute the loss
        loss = criterion(batch.estimates["target"].audio, batch.sources[f"{stem}"].audio) # Y_Pred, Y_True
        train_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    # Validation step
    if epoch % 5 == 0:
        model.eval()
        val_loss = 0.0
        val_metric_handler = MetricHandler(stems)
        with torch.no_grad():
            for mixture, query_emb, target, stem in val_loader:
                mixture = mixture.to(device)
                # query_emb = query_emb.mean(axis=1).to(device)
                target = target.to(device)

                batch = InputType(
                    mixture= SimpleishNamespace(
                        audio=mixture, spectrogram=None
                    ),
                    sources={
                        f"{stem}": SimpleishNamespace(
                            audio=target, spectrogram=None
                        ),
                    },
                    query=SimpleishNamespace(
                        audio=target, spectrogram=None
                    ),
                    estimates={
                        "target": SimpleishNamespace(
                            audio=mixture, spectrogram=None
                        ),
                    },
                    # metadata={"sample_rate": fs}
                )

                # Forward pass
                batch = model(batch)

                # Compute the loss
                loss = criterion(batch.estimates["target"].audio, batch.sources[f"{stem}"].audio) # Y_Pred, Y_True
                val_loss += loss.item()

                # Calculate metrics
                val_metric_handler.calculate_snr(batch.estimates["target"].audio, batch.sources[f"{stem}"].audio, stem)
                
            val_snr = val_metric_handler.get_mean_median()

        
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val SNR: {val_snr}")
        if wandb_use:
            wandb.log({"val_loss": val_loss})
            wandb.log(val_snr)
            # wandb.log({"val_sdr": sdr, "val_sir": sir, "val_sar": sar})
            
        # Early stop
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_thres:
                break
            
    else:
        if wandb_use:
            wandb.log({"train_loss": train_loss})

    
    
    
# Test step after all epochs
model.eval()
test_loss = 0.0
test_metric_handler = MetricHandler(stems)

with torch.no_grad():
    for mixture, query_emb, target, stem in test_loader:
        mixture = mixture.to(device)
        query_emb = query_emb.mean(axis=1).to(device)
        target = target.to(device)

        batch = InputType(
            mixture= SimpleishNamespace(
                audio=mixture, spectrogram=None
            ),
            sources={
                f"{stem}": SimpleishNamespace(
                    audio=target, spectrogram=None
                ),
            },
            query=SimpleishNamespace(
                audio=target, spectrogram=None
            ),
            estimates={
                "target": SimpleishNamespace(
                    audio=mixture, spectrogram=None
                ),
            },
            # metadata={"sample_rate": fs}
        )

        # Forward pass
        batch = model(batch)

        # Compute the loss
        loss = criterion(batch.estimates["target"].audio, batch.sources[f"{stem}"].audio) # Y_Pred, Y_True
        test_loss += loss.item()

        # Calculate metrics
        test_metric_handler.calculate_snr(batch.estimates["target"].audio, batch.sources[f"{stem}"].audio, stem)
    
    # Get the final result of test SNR
    test_snr = test_metric_handler.get_mean_median()
    print("Test snr:", test_snr)
        

print(f"Final Test Loss: {test_loss}")

if wandb_use: wandb.finish()