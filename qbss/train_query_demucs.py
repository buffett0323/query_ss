import os
import json
import wandb
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from mir_eval.separation import bss_eval_sources
from sklearn.model_selection import train_test_split



from query_ss.qbss.htdemucs_qss import Query_HTDemucs
from query_ss.qbss.loss import L1SNR_Recons_Loss, Banquet_L1SNRLoss
from utils import _load_config
from query_ss.qbss.metrics import (
    AverageMeter, cal_metrics, safe_signal_noise_ratio, MetricHandler
)

from models.types import InputType, OperationMode, SimpleishNamespace
from data.moisesdb.datamodule import (
    MoisesTestDataModule,
    MoisesValidationDataModule,
    MoisesDataModule,
    MoisesBalancedTrainDataModule,
    MoisesVDBODataModule,
)


"""
Dataset Structure:
- estimates (predicted)
    - target
        - audio V
- mixtures
    - audio V
    - spectrogram V
- sources
    - target
        - audio V
        - spectrogram X
- query
    - audio V
- masks
    - pred V
    - ground_truth V
- metadata
"""

# Init settings
wandb_use = True # False
lr = 1e-3 # 1e-4
num_epochs = 500
batch_size = 8 # 8
n_srcs = 1
emb_dim = 768 # For BEATs
query_size = 512 # 512
mix_query_mode = "Hyper_FiLM" # "Transformer"
q_enc = "Passt"
config_path = "config/train.yml"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0] #[0, 1, 2, 3] # [0]

def to_device(batch, device=device):
    batch.mixture.audio = batch.mixture.audio.to(device) # torch.Size([BS, 2, 294400])
    batch.sources.target.audio = batch.sources.target.audio.to(device) # torch.Size([BS, 2, 294400])
    batch.query.audio = batch.query.audio.to(device) # torch.Size([BS, 2, 441000])
    return batch


if wandb_use:
    wandb.init(
        project="Query_ss",
        config={
            "learning_rate": lr,
            "architecture": "Query_HTDemucs Using 9 stems",
            "dataset": "MoisesDB",
            "epochs": num_epochs,
        },
        notes=f"HTDemucs Query Version",
    )


config = _load_config(config_path)
stems = config.data.train_kwargs.allowed_stems
print("Training with stems: ", stems)

datamodule = MoisesDataModule(
    data_root=config.data.data_root,
    batch_size=batch_size, #config.data.batch_size,
    num_workers=config.data.num_workers,
    train_kwargs=config.data.get("train_kwargs", None),
    val_kwargs=config.data.get("val_kwargs", None),
    test_kwargs=config.data.get("test_kwargs", None), # Cannot use now
    datamodule_kwargs=config.data.get("datamodule_kwargs", None),
)



# Instantiate the enrollment model
model = Query_HTDemucs(num_sources=1).to(device)

if len(device_ids) > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    model = nn.DataParallel(model)

# Optimizer & Scheduler setup
criterion = Banquet_L1SNRLoss() #L1SNR_Recons_Loss(mask_type=mask_type)
optimizer = optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0)
scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

early_stop_counter, early_stop_thres = 0, 4
min_val_loss = 1e10

val_folder = f"{os.getenv('DATA_ROOT')}/qss_output/val"
test_folder = f"{os.getenv('DATA_ROOT')}/qss_output/test"
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Training loop
for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):

    model.train()
    train_loss = 0.0

    # Second loop with tqdm for batch progress
    for batch_idx, batch in enumerate(tqdm(datamodule.train_dataloader(), desc=f"Epoch {epoch+1} Batch Progress", leave=False)):
        batch = InputType.from_dict(batch)
        batch = to_device(batch)

        optimizer.zero_grad()

        # Forward pass
        batch = model(batch)

        # Compute the loss
        loss, D_ss, D_real, D_imag = criterion(batch)
        train_loss += loss.item()
        if wandb_use:
            wandb.log({
                "Batch Total Loss": loss.item(),
                "Batch SS Loss": D_ss.item(),
                "Batch Real Loss": D_real.item(),
                "Batch Img Loss": D_imag.item(),
                "Batch Index": batch_idx + 1,
                "Epoch": epoch + 1
            })

        # Update tqdm description with loss
        if not wandb_use:
            tqdm.write(f"Batch {batch_idx+1}, Loss: {train_loss:.4f}")

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    scheduler.step()

    # Log epoch loss
    print(f"Epoch {epoch+1}, Training Loss: {train_loss / len(datamodule.train_dataloader()):.4f}")


    # Validation step
    if epoch % 5 == 0:
        model.eval()
        val_loss = 0.0
        val_metric_handler = MetricHandler(stems)
        with torch.no_grad():
            for batch in tqdm(datamodule.val_dataloader()):
                batch = InputType.from_dict(batch)
                batch = to_device(batch)

                # Forward pass
                batch = model(batch)

                # Compute the loss
                loss, D_ss, D_real, D_imag = criterion(batch)
                val_loss += loss.item()

                # Calculate metrics
                val_metric_handler.calculate_snr(
                    batch.estimates.target.audio,
                    batch.sources.target.audio,
                    batch.metadata.stem
                )
                os.makedirs(f"{val_folder}/Epoch_{epoch}", exist_ok=True)
                json_metrics = {
                    key: [tensor.cpu().tolist() for tensor in value] if isinstance(value, list) else value
                    for key, value in val_metric_handler.metrics.items()
                }

                with open(f"{val_folder}/Epoch_{epoch}/metrics.json", "w") as json_file:
                    json.dump(json_metrics, json_file, indent=4)

                for i, (pred, gt, stem) in enumerate(zip(batch.estimates.target.audio, batch.sources.target.audio, batch.metadata.stem)):
                    snr = safe_signal_noise_ratio(pred.cpu(), gt.cpu())
                    snr_mean = (snr[0] + snr[1]) / 2
                    snr = round(snr_mean.item(), 1)

                    pred_path = os.path.join(f"{val_folder}/Epoch_{epoch}", f"{stem}_{i}_pred_{snr}.wav")
                    gt_path = os.path.join(f"{val_folder}/Epoch_{epoch}", f"{stem}_{i}_gt_{snr}.wav")
                    torchaudio.save(pred_path, pred.cpu(), sample_rate=44100)
                    torchaudio.save(gt_path, gt.cpu(), sample_rate=44100)

            # Record the validation SNR
            val_snr = val_metric_handler.get_mean_median()


        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val SNR: {val_snr}")

        if wandb_use:
            wandb.log({"val_loss": val_loss})
            wandb.log(val_snr)

        # Early stop
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_thres:
                break

    if wandb_use:
        wandb.log({"train_loss": train_loss})



# Test step after all epochs
model.eval()
test_loss = 0.0
test_metric_handler = MetricHandler(stems)


with torch.no_grad():
    for batch in tqdm(datamodule.test_dataloader()):
        batch = InputType.from_dict(batch)
        batch = to_device(batch)

        # Forward pass
        batch = model(batch)

        # Compute the loss
        loss, D_ss, D_real, D_imag = criterion(batch)
        test_loss += loss.item()

        # Calculate metrics
        test_metric_handler.calculate_snr(
            batch.estimates.target.audio,
            batch.sources.target.audio,
            batch.metadata.stem
        )

        json_metrics = {
            key: [tensor.cpu().tolist() for tensor in value] if isinstance(value, list) else value
            for key, value in test_metric_handler.metrics.items()
        }

        with open(f"{test_folder}/metrics.json", "w") as json_file:
            json.dump(json_metrics, json_file, indent=4)


        for i, (pred, gt, stem) in enumerate(zip(batch.estimates.target.audio, batch.sources.target.audio, batch.metadata.stem)):
            snr = safe_signal_noise_ratio(pred.cpu(), gt.cpu())
            snr_mean = (snr[0] + snr[1]) / 2
            snr = round(snr_mean.item(), 1)

            pred_path = os.path.join(test_folder, f"{stem}_{i}_pred_{snr}.wav")
            gt_path = os.path.join(test_folder, f"{stem}_{i}_gt_{snr}.wav")
            torchaudio.save(pred_path, pred.cpu(), sample_rate=44100)
            torchaudio.save(gt_path, gt.cpu(), sample_rate=44100)

    # Get the final result of test SNR
    test_snr = test_metric_handler.get_mean_median()
    print("Test snr:", test_snr)


print(f"Final Test Loss: {test_loss}")
if wandb_use:
    wandb.log({"test_loss": test_loss})
    wandb.log(test_snr)


if wandb_use: wandb.finish()
