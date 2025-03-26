import os
import random
import torch
import numba
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.amp import GradScaler

import commons
import utils
import wandb

from accelerate import Accelerator
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
    accelerator = Accelerator()
    # device = accelerator.device

    if accelerator.is_main_process:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    else:
        logger = None
        writer = None
        writer_eval = None

    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    )#.to(device)

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

    test_dataset = BP_DDDM_Dataset(hps, split="test", training=False)
    eval_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=hps.train.num_workers,
        pin_memory=True
    )

    net_v = HiFi(
        hps.data.n_mel_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    )#.to(device)
    utils.load_checkpoint(hps.data.voc_ckpt_path, net_v, None)
    net_v.eval()
    net_v.dec.remove_weight_norm()

    model = DDDM(
        hps.data.n_mel_channels, hps.diffusion.spk_dim,
        hps.diffusion.dec_dim, hps.diffusion.beta_min,
        hps.diffusion.beta_max, hps
    )#.to(device)

    if accelerator.is_main_process and hps.wandb.log_wandb:
        wandb.init(
            project=hps.wandb.project_name,
            name=hps.wandb.run_name,
            config=hps,
        )
        wandb.watch(model, log="all")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )

    model, optimizer, train_loader, eval_loader, net_v = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, net_v
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
    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(accelerator, epoch, hps, model, mel_fn, net_v,
                           optimizer, scheduler_g, scaler, train_loader, eval_loader,
                           logger, writer, writer_eval)
        scheduler_g.step()

def train_and_evaluate(accelerator, epoch, hps, model, mel_fn, net_v, optimizer, scheduler, scaler,
                       train_loader, eval_loader, logger, writer, writer_eval):
    global global_step
    model.train()

    for batch_idx, (x, mel_x, length) in enumerate(tqdm(train_loader)):
        # x = x.to(accelerator.device, non_blocking=True)
        # mel_x = mel_x.to(accelerator.device, non_blocking=True)
        # length = length.to(accelerator.device, non_blocking=True).squeeze()
        length = length.squeeze()
        mel_fn_x = mel_fn(x)

        optimizer.zero_grad()
        loss_diff, loss_mel = model.compute_loss(x, mel_x, mel_fn_x, length)
        loss_gen_all = loss_diff + loss_mel * hps.train.c_mel

        accelerator.backward(loss_gen_all)
        grad_norm = commons.clip_grad_value_(model.parameters(), None)
        optimizer.step()

        if accelerator.is_main_process and global_step % hps.train.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']

            if hps.wandb.log_wandb:
                wandb.log({
                    "loss_diff": loss_diff.item(),
                    "loss_mel": loss_mel.item(),
                    "total_loss": loss_gen_all.item(),
                    "learning_rate": lr,
                    "step": global_step,
                })

            logger.info(f"Epoch {epoch}, Step {global_step}: loss_diff={loss_diff.item():.4f}, loss_mel={loss_mel.item():.4f}, total={loss_gen_all.item():.4f}")
            utils.summarize(writer, global_step, {
                "loss/g/total": loss_gen_all,
                "learning_rate": lr,
                "grad_norm_g": grad_norm,
                "loss/g/diff": loss_diff,
                "loss/g/mel": loss_mel,
            })

        if accelerator.is_main_process and global_step % hps.train.eval_interval == 0:
            torch.cuda.empty_cache()
            evaluate(accelerator, hps, model, mel_fn, net_v, eval_loader, writer_eval)

            if global_step % hps.train.save_interval == 0:
                accelerator.save({
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch
                }, os.path.join(hps.data.model_save_dir, f"G_{global_step}.pth"))

        global_step += 1

def evaluate(accelerator, hps, model, mel_fn, net_v, eval_loader, writer_eval):
    global global_step
    model.eval()
    audio_dict = {}
    mel_loss = 0
    enc_loss = 0

    with torch.no_grad():
        for batch_idx, (y, mel_y) in enumerate(tqdm(eval_loader)):
            # y = y.to(accelerator.device)
            mel_fn_y = mel_fn(y)
            length = torch.LongTensor([mel_fn_y.size(2)])#.to(accelerator.device)

            enc_output, mel_rec = model(y, mel_y, mel_fn_y, length, n_timesteps=6, mode='ml')

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

    if hps.wandb.log_wandb and accelerator.is_main_process:
        wandb.log({
            "val_mel_loss": mel_loss,
            "val_enc_loss": enc_loss,
            "step": global_step,
        })

    if accelerator.is_main_process:
        utils.summarize(
            writer=writer_eval,
            global_step=global_step,
            audios=audio_dict,
            audio_sampling_rate=hps.data.sampling_rate,
            scalars={
                "val/mel": mel_loss,
                "val/enc_mel": enc_loss
            }
        )

    model.train()

if __name__ == "__main__":
    main()
