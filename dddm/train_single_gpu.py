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
from model.te_dddm_mixup import DDDM
from data_loader import BP_DDDM_Dataset, MelSpectrogramFixed
from vocoder.hifigan import HiFi

torch.backends.cudnn.benchmark = True
global_step = 0

def get_param_num(model):
    return sum(param.numel() for param in model.parameters())

def main():
    global global_step
    assert torch.cuda.is_available(), "CUDA GPU is required for training."

    hps = utils.get_hparams(
        config_path="./ckpt/config.json",
        model_dir="/home/buffett/nas_home/buffett/timbre_transfer_logs/",
    )
    device = torch.device(f"cuda:{hps.train.device}")

    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)

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

        optimizer.zero_grad()
        loss_diff, loss_mel = model.compute_loss(x, mel_x, mel_fn_x, length)
        loss_gen_all = loss_diff + loss_mel * hps.train.c_mel

        if hps.train.fp16_run:
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optimizer)
            grad_norm = commons.clip_grad_value_(model.parameters(), None)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_gen_all.backward()
            grad_norm = commons.clip_grad_value_(model.parameters(), None)
            optimizer.step()

        if global_step % hps.train.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch}, Step {global_step}: loss_diff={loss_diff.item():.4f}, loss_mel={loss_mel.item():.4f}, total={loss_gen_all.item():.4f}")

            if hps.wandb.log_wandb:
                wandb.log({
                    "loss_diff": loss_diff.item(),
                    "loss_mel": loss_mel.item(),
                    "total_loss": loss_gen_all.item(),
                    "learning_rate": lr,
                    "step": global_step,
                })

            utils.summarize(
                writer=writer,
                global_step=global_step,
                scalars={
                "loss/g/total": loss_gen_all,
                "learning_rate": lr,
                "grad_norm_g": grad_norm,
                "loss/g/diff": loss_diff,
                "loss/g/mel": loss_mel,
                },
            )

        if global_step % hps.train.eval_interval == 0:
            torch.cuda.empty_cache()
            evaluate(hps, model, mel_fn, net_v, eval_loader, writer_eval, device)

            if global_step % hps.train.save_interval == 0:
                utils.save_checkpoint(
                    model, optimizer, hps.train.learning_rate, epoch,
                    os.path.join(hps.data.model_save_dir, f"G_{global_step}.pth")
                )

        global_step += 1


def evaluate(hps, model, mel_fn, net_v, eval_loader, writer_eval, device):
    model.eval()
    mel_loss, enc_loss = 0, 0
    audio_dict = {}
    with torch.no_grad():
        for batch_idx, (y, mel_y) in enumerate(tqdm(eval_loader)):
            y = y.to(device)
            mel_y = mel_y.to(device)
            mel_fn_y = mel_fn(y)
            length = torch.LongTensor([mel_y.size(2)]).to(device)

            enc_output, mel_rec = model(y, mel_y, mel_fn_y, length)#, n_timesteps=6, mode='ml')
            mel_loss += F.l1_loss(mel_fn_y, mel_rec).item()
            enc_loss += F.l1_loss(mel_fn_y, enc_output).item()

            if batch_idx <= hps.train.save_audio_num:
                y_hat = net_v(mel_rec)
                enc_hat = net_v(enc_output)
                audio_dict.update({
                    f"gen/audio_{batch_idx}": y_hat.squeeze(),
                    f"gen/enc_audio_{batch_idx}": enc_hat.squeeze(),
                })
                if global_step == 0:
                    audio_dict.update({f"gt/audio_{batch_idx}": y.squeeze()})

            if batch_idx > 100:
                break

    mel_loss /= 100
    enc_loss /= 100

    if hps.wandb.log_wandb:
        wandb.log({
            "val_mel_loss": mel_loss,
            "val_enc_loss": enc_loss,
            "step": global_step,
        })

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        audio_sampling_rate=hps.data.sampling_rate,
        audios=audio_dict,
        scalars={
            "val/mel": mel_loss,
            "val/enc_mel": enc_loss,
        },
    )
    model.train()

if __name__ == "__main__":
    main()
