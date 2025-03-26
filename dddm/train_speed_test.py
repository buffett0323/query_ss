import logging
import warnings
logging.getLogger("numba").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler
from tqdm import tqdm

import commons
import utils
import wandb

from augmentation.aug import Augment
from model.new_dddm_mixup import DDDM
from data_loader import BP_DDDM_Dataset, MelSpectrogramFixed
from vocoder.hifigan import HiFi

torch.backends.cudnn.benchmark = True
global_step = 0

def get_param_num(model):
    return sum(param.numel() for param in model.parameters())

def main():
    global global_step
    assert torch.cuda.is_available(), "CUDA GPU is required for training."

    hps = utils.get_hparams()
    device = torch.device(f"cuda:{hps.train.device}")

    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    # utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    ).to(device)

    train_dataset = BP_DDDM_Dataset(hps, split="train", training=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        num_workers=hps.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )

    eval_dataset = BP_DDDM_Dataset(hps, split="test", training=False)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        num_workers=hps.train.num_workers,
        pin_memory=True
    )

    net_v = HiFi(
        hps.data.n_mel_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)
    utils.load_checkpoint(hps.data.voc_ckpt_path, net_v, None)
    net_v.eval()
    net_v.dec.remove_weight_norm()

    model = DDDM(
        hps.data.n_mel_channels, hps.diffusion.spk_dim,
        hps.diffusion.dec_dim, hps.diffusion.beta_min,
        hps.diffusion.beta_max, hps
    ).to(device)

    if hps.wandb.log_wandb:
        wandb.init(
            project=hps.wandb.project_name,
            name=hps.wandb.run_name,
            config=hps,
        )
        wandb.watch(model, log="all")

    print('[Encoder] number of Parameters:', get_param_num(model.encoder))
    print('[Decoder] number of Parameters:', get_param_num(model.decoder))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )

    if hps.train.resume:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.data.model_save_dir, "G_*.pth"),
            model, optimizer
        )
        global_step = (epoch_str - 1) * len(train_loader)
    else:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scaler = GradScaler(f"cuda:{hps.train.device}", enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_one_epoch(model, train_loader, eval_loader, mel_fn, net_v, optimizer, 
                        scheduler_g, scaler, writer, writer_eval, 
                        hps, epoch, device, logger)
        scheduler_g.step()

def train_one_epoch(model, train_loader, eval_loader, mel_fn, net_v, optimizer, scheduler, scaler, writer, writer_eval, hps, epoch, device, logger):
    global global_step
    model.train()
    for batch_idx, (x, mel_x, length) in enumerate(tqdm(train_loader)):
        x = x.to(device)
        mel_x = mel_x.to(device)
        length = length.to(device).squeeze()
        mel_fn_x = mel_fn(x)#.to(device)
        continue
    

if __name__ == "__main__":
    main()