import sys
import warnings
import argparse
import os
import torch
import numpy as np
import torch
import soundfile as sf
import pretty_midi
import argparse
import librosa
import torch.nn.functional as F
import matplotlib.pyplot as plt
warnings.simplefilter('ignore')

from modules.commons import *
from losses import *
from typing import List, Dict
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss, GANLoss, L1Loss
from audiotools import AudioSignal
from audiotools import ml
from audiotools.ml.decorators import Tracker, timer, when
from audiotools.core import util
from evaluate_metrics import LogRMSEnvelopeLoss


from utils import (
    yaml_config_hook, get_infinite_loader, save_checkpoint, load_checkpoint, log_rms
)
import dac


class EDM_MN_Val_Total_Dataset(Dataset):
    def __init__(
        self,
        root_path: str,
        midi_path: str,
        duration: float = 1.0,
        sample_rate: int = 44100,
        split: str = "train",
        reconstruction: bool = False,
    ):
        self.root_path = Path(os.path.join(root_path, split))
        self.midi_path = Path(os.path.join(midi_path, "evaluation", "midi"))
        self.duration = duration
        self.sample_rate = sample_rate
        self.split = split
        self.reconstruction = reconstruction

        # Storing names
        self.paired_data = []
        self._build_paired_index()



    def _build_paired_index(self):
        if self.reconstruction:
            with open("info/test_cases.txt", "r") as f:
                for line in f:
                    if line.startswith("Original:") or line.startswith("Reference:") or line.startswith("Target:"):
                        self.paired_data.append(
                            os.path.join(self.root_path, line.strip().split()[-1] + ".wav")
                        )
                print(f"Paired data: {len(self.paired_data)}")

        else:
            with open("info/test_cases.txt", "r") as f:
                origs, refs, targets = [], [], []
                for line in f:
                    if line.startswith("Original:"):
                        origs.append(
                            os.path.join(self.root_path, line.strip().split()[-1] + ".wav")
                        )
                    elif line.startswith("Reference:"):
                        refs.append(
                            os.path.join(self.root_path, line.strip().split()[-1] + ".wav")
                        )
                    elif line.startswith("Target:"):
                        targets.append(
                            os.path.join(self.root_path, line.strip().split()[-1] + ".wav")
                        )

                for org, rf, tg in zip(origs, refs, targets):
                    self.paired_data.append((org, rf, tg))
                print(f"Paired data: {len(self.paired_data)}")


    def _load_audio(self, file_path: Path, offset: float = 0.0) -> AudioSignal:
        signal, _ = sf.read(
            file_path,
            start=int(offset*self.sample_rate),
            frames=int(self.duration*self.sample_rate)
        )
        # signal = signal.mean(axis=1, keepdims=False)
        return AudioSignal(signal, self.sample_rate)


    def __len__(self):
        return len(self.paired_data)


    def __getitem__(self, idx):
        if self.reconstruction:
            wav_path = self.paired_data[idx]
            orig_audio = self._load_audio(wav_path, 0.0)

            return {
                'orig_audio': orig_audio,
                'ref_audio': orig_audio,
                'target_audio': orig_audio
            }

        else:
            org, rf, tg = self.paired_data[idx]
            orig_audio = self._load_audio(org, 0.0)
            ref_audio = self._load_audio(rf, 0.0)
            target_audio = self._load_audio(tg, 0.0)

            return {
                'orig_audio': orig_audio,
                'ref_audio': ref_audio,
                'target_audio': target_audio
            }

    @staticmethod
    def collate(batch: List[Dict]) -> Dict:
        """Custom collate function for batching"""
        return {
            'orig_audio': AudioSignal.batch([item['orig_audio'] for item in batch]),
            'ref_audio': AudioSignal.batch([item['ref_audio'] for item in batch]),
            'target_audio': AudioSignal.batch([item['target_audio'] for item in batch]),
        }


class Wrapper:
    def __init__(
        self,
        args,
        accelerator,
        val_paired_data,
    ):
        self.disentanglement = args.disentanglement # training
        self.convert_type = args.convert_type # validation

        self.generator = dac.model.MyDAC(
            encoder_dim=args.encoder_dim,
            encoder_rates=args.encoder_rates,
            latent_dim=args.latent_dim,
            decoder_dim=args.decoder_dim,
            decoder_rates=args.decoder_rates,
            adsr_enc_dim=args.adsr_enc_dim,
            adsr_enc_ver=args.adsr_enc_ver,
            sample_rate=args.sample_rate,
            timbre_classes=args.timbre_classes,
            adsr_classes=args.adsr_classes,
            pitch_nums=args.max_note - args.min_note + 1, # 88
            use_gr_content=args.use_gr_content,
            use_gr_adsr=args.use_gr_adsr,
            use_gr_timbre=args.use_gr_timbre,
            use_FiLM=args.use_FiLM,
            rule_based_adsr_folding=args.rule_based_adsr_folding,
            use_cross_attn=args.use_cross_attn,
        ).to(accelerator.device)

        self.optimizer_g = torch.optim.AdamW(self.generator.parameters(), lr=args.base_lr)
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_g, gamma=1.0)

        self.discriminator = dac.model.Discriminator().to(accelerator.device)
        self.optimizer_d = torch.optim.AdamW(self.discriminator.parameters(), lr=args.base_lr)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_d, gamma=1.0)

        # Losses
        self.stft_loss = MultiScaleSTFTLoss().to(accelerator.device)
        self.envelope_loss = LogRMSEnvelopeLoss().to(accelerator.device)

        # Val dataset
        self.val_paired_data = val_paired_data


def main(args, accelerator):
    device = accelerator.device
    util.seed(args.seed)
    print(f"Using device: {device}")

    convert_type = args.conv_type
    print(f"Convert type: {convert_type}")

    val_paired_data = EDM_MN_Val_Total_Dataset(
        root_path=args.root_path,
        midi_path=args.midi_path,
        duration=3, #args.duration,
        sample_rate=args.sample_rate,
        split=args.split,
        reconstruction=True if convert_type == "reconstruction" else False,
    )

    val_paired_loader = accelerator.prepare_dataloader(
        val_paired_data,
        start_idx=0,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=val_paired_data.collate,
    )
    wrapper = Wrapper(args, accelerator, val_paired_data)
    load_checkpoint(args, device, args.iter, wrapper)

    # Start iteration
    counter = 1
    for i, paired_batch in tqdm(enumerate(val_paired_loader), desc="Evaluating", total=len(val_paired_loader)):
        batch = util.prepare_batch(paired_batch, accelerator.device)

        if convert_type == "reconstruction":
            target_audio = batch['orig_audio']
            with torch.no_grad():
                out = wrapper.generator.conversion(
                    orig_audio=batch['orig_audio'].audio_data,
                    ref_audio=None,
                    convert_type=convert_type,
                )

            orig_audio = AudioSignal(batch['orig_audio'].audio_data.cpu(), args.sample_rate)
            recons = AudioSignal(out["audio"].cpu(), args.sample_rate)
            recons_gt = AudioSignal(target_audio.audio_data.cpu(), args.sample_rate)


            for j in range(args.batch_size):
                recon_path = os.path.join(args.audio_save_path, f'{counter:02d}_recon_{args.model_name}.wav')
                recon_gt_path = os.path.join(args.audio_save_path, f'{counter:02d}_gt.wav')
                orig_path = os.path.join(args.audio_save_path, f'{counter:02d}_orig.wav')

                recons[j].write(recon_path)
                recons_gt[j].write(recon_gt_path)
                orig_audio[j].write(orig_path)
                counter += 1

        else:
            target_audio = batch['target_audio']
            with torch.no_grad():
                out = wrapper.generator.conversion(
                    orig_audio=batch['orig_audio'].audio_data,
                    ref_audio=batch['ref_audio'].audio_data,
                    convert_type=convert_type,
                )

            orig_audio = AudioSignal(batch['orig_audio'].audio_data.cpu(), args.sample_rate)
            ref_audio = AudioSignal(batch['ref_audio'].audio_data.cpu(), args.sample_rate)
            recons = AudioSignal(out["audio"].cpu(), args.sample_rate)
            recons_gt = AudioSignal(target_audio.audio_data.cpu(), args.sample_rate)

            for j in range(args.batch_size):
                single_recon = AudioSignal(recons.audio_data[j], args.sample_rate)
                single_recon_gt = AudioSignal(recons_gt.audio_data[j], args.sample_rate)
                single_orig = AudioSignal(orig_audio.audio_data[j], args.sample_rate)
                single_ref = AudioSignal(ref_audio.audio_data[j], args.sample_rate)


                recon_path = os.path.join(args.audio_save_path, f'{counter:02d}_recon_{args.model_name}.wav')
                recon_gt_path = os.path.join(args.audio_save_path, f'{counter:02d}_gt.wav')
                orig_path = os.path.join(args.audio_save_path, f'{counter:02d}_orig.wav')
                ref_path = os.path.join(args.audio_save_path, f'{counter:02d}_ref.wav')

                single_recon.write(recon_path)
                single_recon_gt.write(recon_gt_path)
                single_orig.write(orig_path)
                single_ref.write(ref_path)
                counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDM-FAC")
    parser.add_argument("--conv_type", default="both")
    parser.add_argument("--iter", default=-1, type=int)
    parser.add_argument("--split", default="eval_seen_extreme_adsr") # eval_seen_normal_adsr
    parser.add_argument("--model_name", default="proposed")
    parser.add_argument("--audio_save_path",
                        default="/home/buffett/nas_data/EDM_FAC_LOG/demo_website/sample_audio")

    # config = yaml_config_hook("configs/config_proposed_no_mask.yaml")
    config = yaml_config_hook("configs/config_proposed_no_ca.yaml")

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.ckpt_path, exist_ok=True)

    # Initialize accelerator
    accelerator = ml.Accelerator()
    if accelerator.local_rank != 0:
        sys.tracebacklimit = 0
    main(args, accelerator)
