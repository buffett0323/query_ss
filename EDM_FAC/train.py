import shutil
import warnings
import argparse
import torch
import os
import os.path as osp
import yaml
import random
import time
import glob
import torchaudio
import torch
warnings.simplefilter('ignore')



from modules.commons import *
from losses import *
from optimizers import build_optimizer

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import DistributedDataParallelKwargs

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

import logging
from accelerate.logging import get_logger

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss, GANLoss, L1Loss
from audiotools import AudioSignal
from dataset import EDM_Render_Dataset

logger = get_logger(__name__, log_level="INFO")
# torch.autograd.set_detect_anomaly(True)


def build_dataloader(
    dataset,
    batch_size=32,
    num_workers=0,
    prefetch_factor=16,
    split="train",
):
    collate_fn = dataset.collate
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        shuffle=True if split == "train" else False,
    )

    return data_loader



def main(args):
    config_path = args.config_path
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    
    # Accelerate setup
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=True)
    accelerator = Accelerator(project_dir=log_dir, split_batches=True, kwargs_handlers=[ddp_kwargs])
    
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.logger.addHandler(file_handler)

    # load model and processor
    batch_size = config.get('batch_size', 10)
    batch_length = config.get('batch_length', 120)
    device = accelerator.device #if accelerator.num_processes > 1 else torch.device('cpu')

    epochs = config.get('epochs', 200)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)
    save_interval = config.get('save_interval', 1000)

    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    root_path = data_params['root_path']
    midi_path = data_params['midi_path']
    
    num_workers = config.get('num_workers', 4)
    prefetch_factor = config.get('prefetch_factor', 8)
    
    duration = config.get('duration', 0.38)
    sample_rate = config.get('sample_rate', 44100)
    min_note = config.get('min_note', 21)
    max_note = config.get('max_note', 108)
    stems = config.get('stems', ["lead", "pad", "bass", "keys", "pluck"])
    
    # max_frame_len = config.get('max_len', 80)
    # discriminator_iter_start = config['loss_params'].get('discriminator_iter_start', 0)
    # loss_params = config.get('loss_params', {})
    # hop_length = config['preprocess_params']['spect_params'].get('hop_length', 300)
    # win_length = config['preprocess_params']['spect_params'].get('win_length', 1200)
    # n_fft = config['preprocess_params']['spect_params'].get('n_fft', 2048)
    # norm_f0 = config['model_params'].get('norm_f0', True)
    # frame_rate = sr // hop_length


    # Build datasets and dataloaders
    train_data = EDM_Render_Dataset(
        root_path=root_path,
        midi_path=os.path.join(midi_path, "train", "midi"),
        duration=duration,
        sample_rate=sample_rate,
        min_note=min_note,
        max_note=max_note,
        stems=stems,
    )
    
    val_data = EDM_Render_Dataset(
        root_path=root_path,
        midi_path=os.path.join(midi_path, "evaluation", "midi"),
        duration=duration,
        sample_rate=sample_rate,
        min_note=min_note,
        max_note=max_note,
        stems=stems,
    )

    train_loader = build_dataloader(
        dataset=train_data, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        prefetch_factor=prefetch_factor,
        split="train",
    )
    val_loader = build_dataloader(
        dataset=val_data, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        prefetch_factor=prefetch_factor,
        split="val",
    )

    scheduler_params = {
        "warmup_steps": 200,
        "base_lr": 0.0001,
    }

    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params)

    for k in model:
        model[k] = accelerator.prepare(model[k])

    _ = [model[key].to(device) for key in model]

    # initialize optimizers after preparing models for compatibility with FSDP
    optimizer = build_optimizer(
        {key: model[key] for key in model},
        scheduler_params_dict={key: scheduler_params.copy() for key in model},
        lr=float(scheduler_params['base_lr'])
    )

    for k, v in optimizer.optimizers.items():
        optimizer.optimizers[k] = accelerator.prepare(optimizer.optimizers[k])
        optimizer.schedulers[k] = accelerator.prepare(optimizer.schedulers[k])

    # find latest checkpoint with name pattern of 'T2V_epoch_*_step_*.pth'
    available_checkpoints = glob.glob(osp.join(log_dir, "FAcodec_epoch_*_step_*.pth"))
    if len(available_checkpoints) > 0:
        # find the checkpoint that has the highest step number
        latest_checkpoint = max(
            available_checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        earliest_checkpoint = min(
            available_checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        # delete the earliest checkpoint
        if (
            earliest_checkpoint != latest_checkpoint
            and accelerator.is_main_process
            and len(available_checkpoints) > 4
        ):
            os.remove(earliest_checkpoint)
            print(f"Removed {earliest_checkpoint}")
    else:
        latest_checkpoint = config.get("pretrained_model", "")


    with accelerator.main_process_first():
        if latest_checkpoint != '':
            model, optimizer, start_epoch, iters = load_checkpoint(model, optimizer, latest_checkpoint,
                  load_only_params=config.get('load_only_params', True), ignore_modules=[], is_distributed=accelerator.num_processes > 1)
        else:
            start_epoch = 0
            iters = 0

    content_criterion = FocalLoss(gamma=2).to(device)
    stft_criterion = MultiScaleSTFTLoss().to(device)
    mel_criterion = MelSpectrogramLoss(
        n_mels=[5, 10, 20, 40, 80, 160, 320],
        window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
        mel_fmin=[0, 0, 0, 0, 0, 0, 0],
        mel_fmax=[None, None, None, None, None, None, None],
        pow=1.0,
        mag_weight=0.0,
        clamp_eps=1e-5,
    ).to(device)
    l1_criterion = L1Loss().to(device)


    # accelerate prepare
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config.yml')
    args = parser.parse_args()
    main(args)