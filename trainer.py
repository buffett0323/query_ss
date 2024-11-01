import os
import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from typing import Optional
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from mir_eval.separation import bss_eval_sources
from sklearn.model_selection import train_test_split



from enrollment_model import MyModel
# from load_data import BEATS_path, ORIG_mixture, ORIG_target #, stems
# from dataset import MusicDataset
from loss import L1SNR_Recons_Loss, L1SNRDecibelMatchLoss
from utils import _load_config
from metrics import (
    AverageMeter, cal_metrics, safe_signal_noise_ratio, MetricHandler
)

from models.types import InputType, OperationMode, SimpleishNamespace
from data.moisesdb.datamodule import (
    get_datasets,
    MoisesTestDataModule,
    MoisesValidationDataModule,
    MoisesDataModule,
    MoisesBalancedTrainDataModule,
    MoisesVDBODataModule,
)


import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

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


    

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        stems: list,
        gpu_id: int,
        num_epochs: int,
        save_every: int,
        early_stop_thres: int = 4,
        wandb_use: bool = False, # True
    ) -> None:
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.stems = stems
        self.gpu_id = gpu_id
        self.num_epochs = num_epochs
        self.early_stop_thres = early_stop_thres
        self.save_every = save_every
        self.wandb_use = wandb_use
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.min_val_loss = 1e10
        self.early_stop_counter = 0
        
        # Wrapper for DDP
        self.model = model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        
        
    def _run_batch(self, batch):
        batch = InputType.from_dict(batch)
        batch = self.to_device(batch)
        
        # Forward pass
        outputs = self.model(batch)

        # Compute loss
        loss = self.criterion(outputs.estimates["target"].audio, batch.sources["target"].audio)

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def to_device(self, batch):
        batch.mixture.audio = batch.mixture.audio.to(self.device)  # Move to GPU
        batch.sources.target.audio = batch.sources.target.audio.to(self.device)
        batch.query.audio = batch.query.audio.to(self.device)
        return batch


    def _run_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        for batch in tqdm(self.train_data):
            train_loss += self._run_batch(batch)

        self.scheduler.step()
        print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss}")
        return train_loss
    

    def _validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        val_metric_handler = MetricHandler(self.stems)
        with torch.no_grad():
            for batch in tqdm(self.val_data):
                batch = InputType.from_dict(batch)
                batch = self.to_device(batch)

                # Forward pass
                outputs = self.model(batch)

                # Compute the loss
                loss = self.criterion(outputs.estimates["target"].audio, batch.sources["target"].audio)
                val_loss += loss.item()

                # Calculate metrics
                val_metric_handler.calculate_snr(outputs.estimates["target"].audio, batch.sources["target"].audio, batch.metadata.stem)

            # Record the validation SNR
            val_snr = val_metric_handler.get_mean_median()

        print(f"Epoch {epoch+1}/{self.num_epochs}, Val Loss: {val_loss}, Val SNR: {val_snr}")
        if self.wandb_use:
            wandb.log({"val_loss": val_loss})
            wandb.log(val_snr)

        return val_loss


    def _save_model(self, epoch):
        ckp = self.model.module.state_dict()
        torch.save(ckp, "checkpoint.pt")
        print(f"Epoch {epoch} | Training checkpoint saved at checkpoint.pt")
        

    def train(self):
        for epoch in tqdm(range(self.num_epochs)):
            train_loss = self._run_epoch(epoch)
            
            # if epoch % 5 == 0:
            #     val_loss = self._validate(epoch)
                
            #     # Early stopping logic
            #     if val_loss < self.min_val_loss:
            #         self.min_val_loss = val_loss
            #         self.early_stop_counter = 0
            #     else:
            #         self.early_stop_counter += 1
            #         if self.early_stop_counter >= self.early_stop_thres:
            #             print(f"Early stopping at epoch {epoch+1}")
            #             break

            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_model(epoch)

            if self.wandb_use:
                wandb.log({"train_loss": train_loss})


def load_train_objs(
    wandb_use = False, # False
    lr = 1e-3, # 1e-4
    num_epochs = 500,
    batch_size = 4, #8
    n_srcs = 2,
    emb_dim = 768, # For BEATs
    mix_query_mode = "FiLM",
    q_enc = "Passt",
    config_path = "config/train.yml",
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    if wandb_use:
        wandb.init(
            project="Query_ss",
            config={
            "learning_rate": lr,
            "architecture": "FiLM_UNet Using Other's dataset",
            "dataset": "MoisesDB",
            "epochs": num_epochs,
            }
        )
    
    config = _load_config(config_path)
    stems = config.data.train_kwargs.allowed_stems
    print("Training with stems: ", stems)
    
    train_dataset, val_dataset, test_dataset = get_datasets(
        data_root=config.data.data_root,
        batch_size=batch_size, #config.data.batch_size,
        num_workers=config.data.num_workers,
        train_kwargs=config.data.get("train_kwargs", None),
        val_kwargs=config.data.get("val_kwargs", None),
        test_kwargs=config.data.get("test_kwargs", None), # Cannot use now
        datamodule_kwargs=config.data.get("datamodule_kwargs", None),
    )

    # Instantiate the enrollment model
    model = MyModel(
        batch_size=batch_size,
        in_channels=n_srcs, 
        embedding_size=emb_dim, 
        out_channels=emb_dim, 
        kernel_size=(3, 3), 
        stride=(1, 1),
        n_masks=n_srcs,
        mix_query_mode=mix_query_mode,
        q_enc=q_enc,
    )
    
    # Optimizer & Scheduler setup
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
    criterion = L1SNRDecibelMatchLoss() # criterion = L1SNR_Recons_Loss()
    
    return stems, train_dataset, val_dataset, test_dataset, model, optimizer, scheduler, criterion


def prepare_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        # shuffle=True,
        sampler=DistributedSampler(dataset),
    )
    
    
def main(
    rank: int,
    world_size: int, 
    save_every: int, 
    total_epochs: int, 
    batch_size: int,
):
    ddp_setup(rank, world_size)
    stems, train_dataset, val_dataset, test_dataset, model, optimizer, scheduler, criterion = load_train_objs(
        num_epochs=total_epochs,
        batch_size=batch_size,
    )
    
    train_data = prepare_dataloader(train_dataset, batch_size)
    val_data = prepare_dataloader(val_dataset, batch_size)
    
    trainer = Trainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        stems=stems,
        gpu_id=rank,
        num_epochs=total_epochs,
        save_every=save_every,
    )
    
    trainer.train()
    destroy_process_group()
    
    


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=500, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=10, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=2, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)