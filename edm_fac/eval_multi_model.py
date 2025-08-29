import sys
import warnings
import argparse
import os
import torch
import json
import random
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
        hop_length: int = 512,
        min_note: int = 21,
        max_note: int = 108,
        mask_delay_frames: int = 3, # About 3 frames
        split: str = "train",
        perturb_content: bool = True,
        perturb_adsr: bool = True,
        perturb_timbre: bool = True,
        get_midi_only_from_onset: bool = False,
        disentanglement_mode: List[str] = ["reconstruction", "conv_both", "conv_adsr", "conv_timbre"],
        pair_cnts: int = 1,
        reconstruction: bool = False,
    ):
        self.root_path = Path(os.path.join(root_path, split))
        self.midi_path = Path(os.path.join(midi_path, "evaluation", "midi"))
        self.duration = duration
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.min_note = min_note
        self.max_note = max_note
        self.mask_delay_frames = mask_delay_frames
        self.mask_delay = round(self.mask_delay_frames * hop_length / sample_rate, 4)
        self.n_notes = max_note - min_note + 1
        self.split = split
        self.perturb_content = perturb_content
        self.perturb_adsr = perturb_adsr
        self.perturb_timbre = perturb_timbre
        self.get_midi_only_from_onset = get_midi_only_from_onset
        self.disentanglement_mode = disentanglement_mode
        self.pair_cnts = pair_cnts
        self.reconstruction = reconstruction

        if self.get_midi_only_from_onset:
            print(f"Mask onset after: {self.mask_delay} seconds")



        # Pre-load metadata
        with open(f'{self.root_path}/metadata.json', 'r') as f:
            self.metadata = json.load(f)
            # Shuffle metadata
            random.shuffle(self.metadata)

        # Create ID mappings for all three levels
        self.ids_to_item_idx = {}

        # Storing names
        self.paired_data = []
        self.single_data = []
        self._build_paired_index()



    def _build_paired_index(self):
        if self.reconstruction:
            self.paired_data = []
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
                self.paired_data = list(zip(origs, refs, targets))

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
            timbre_id, midi_id, adsr_id, wav_path = self.paired_data[idx]
            orig_audio = self._load_audio(wav_path, 0.0)
            return {
                'orig_audio': orig_audio,
                'ref_audio': orig_audio,
                'target_timbre': orig_audio,
                'target_adsr': orig_audio,
                'target_both': orig_audio,

                'metadata': {
                    'orig_audio': {
                        'timbre_id': timbre_id,
                        'content_id': midi_id,
                        'adsr_id': adsr_id,
                        'path': str(wav_path)
                    },
                }
            }

        else:
            timbre_id, midi_id, adsr_id, wav_path, ref_timbre_id, ref_midi_id, ref_adsr_id, ref_wav_path = self.paired_data[idx]
            orig_audio = self._load_audio(wav_path, 0.0)
            ref_audio = self._load_audio(ref_wav_path, 0.0)


            target_timbre_idx = self.ids_to_item_idx[f"T{ref_timbre_id:03d}_ADSR{adsr_id:03d}_C{midi_id:03d}"]
            target_adsr_idx = self.ids_to_item_idx[f"T{timbre_id:03d}_ADSR{ref_adsr_id:03d}_C{midi_id:03d}"]
            target_both_idx = self.ids_to_item_idx[f"T{ref_timbre_id:03d}_ADSR{ref_adsr_id:03d}_C{midi_id:03d}"]

            # target_content = self._load_audio(self.paired_data[target_content_idx][3], ref_offset_pick)
            target_timbre = self._load_audio(self.paired_data[target_timbre_idx][3], 0.0)
            target_adsr = self._load_audio(self.paired_data[target_adsr_idx][3], 0.0)
            target_both = self._load_audio(self.paired_data[target_both_idx][3], 0.0)

            return {
                'orig_audio': orig_audio,
                'ref_audio': ref_audio,
                'target_timbre': target_timbre,
                'target_adsr': target_adsr,
                'target_both': target_both,

                'metadata': {
                    'orig_audio': {
                        'timbre_id': timbre_id,
                        'content_id': midi_id,
                        'adsr_id': adsr_id,
                        'path': str(wav_path)
                    },
                    'ref_audio': {
                        'timbre_id': ref_timbre_id,
                        'content_id': ref_midi_id,
                        'adsr_id': ref_adsr_id,
                        'path': str(ref_wav_path),
                    },
                    'target_timbre': {
                        'timbre_id': self.paired_data[target_timbre_idx][0],
                        'content_id': self.paired_data[target_timbre_idx][1],
                        'adsr_id': self.paired_data[target_timbre_idx][2],
                        'path': str(self.paired_data[target_timbre_idx][3]),
                    },
                    'target_adsr': {
                        'timbre_id': self.paired_data[target_adsr_idx][0],
                        'content_id': self.paired_data[target_adsr_idx][1],
                        'adsr_id': self.paired_data[target_adsr_idx][2],
                        'path': str(self.paired_data[target_adsr_idx][3]),
                    },
                    'target_both': {
                        'timbre_id': self.paired_data[target_both_idx][0],
                        'content_id': self.paired_data[target_both_idx][1],
                        'adsr_id': self.paired_data[target_both_idx][2],
                        'path': str(self.paired_data[target_both_idx][3]),
                    },
                }
            }

    @staticmethod
    def collate(batch: List[Dict]) -> Dict:
        """Custom collate function for batching"""
        return {
            'orig_audio': AudioSignal.batch([item['orig_audio'] for item in batch]),
            'ref_audio': AudioSignal.batch([item['ref_audio'] for item in batch]),
            'target_timbre': AudioSignal.batch([item['target_timbre'] for item in batch]),
            'target_adsr': AudioSignal.batch([item['target_adsr'] for item in batch]),
            'target_both': AudioSignal.batch([item['target_both'] for item in batch]),
            'metadata': [item['metadata'] for item in batch]
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
        hop_length=args.hop_length,
        split=args.split,
        perturb_content=args.perturb_content,
        perturb_adsr=args.perturb_adsr,
        perturb_timbre=args.perturb_timbre,
        get_midi_only_from_onset=args.get_midi_only_from_onset,
        mask_delay_frames=args.mask_delay_frames,
        disentanglement_mode=args.disentanglement,
        pair_cnts=args.pair_cnts,
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
    total_stft_loss = []
    total_envelope_loss = []
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

            recons = AudioSignal(out["audio"], args.sample_rate)
            stft_loss = wrapper.stft_loss(recons, target_audio)
            envelope_loss = wrapper.envelope_loss(recons, target_audio)
            total_stft_loss.append(stft_loss)
            total_envelope_loss.append(envelope_loss)
            print(f"Batch {i}: STFT Loss: {stft_loss:.4f}, Envelope Loss: {envelope_loss:.4f}")

        else:
            target_audio = batch[f'target_{convert_type}']
            with torch.no_grad():
                out = wrapper.generator.conversion(
                    orig_audio=batch['orig_audio'].audio_data,
                    ref_audio=batch['ref_audio'].audio_data,
                    convert_type=convert_type,
                )

            recons = AudioSignal(out["audio"], args.sample_rate)
            stft_loss = wrapper.stft_loss(recons, target_audio)
            envelope_loss = wrapper.envelope_loss(recons, target_audio)
            total_stft_loss.append(stft_loss)
            total_envelope_loss.append(envelope_loss)
            print(f"Batch {i}: STFT Loss: {stft_loss:.4f}, Envelope Loss: {envelope_loss:.4f}")

    # Summary
    avg_stft_loss = sum(total_stft_loss) / len(total_stft_loss)
    avg_envelope_loss = sum(total_envelope_loss) / len(total_envelope_loss)
    print(f"Average STFT Loss: {avg_stft_loss:.4f}")
    print(f"Average Envelope Loss: {avg_envelope_loss:.4f}")

    with open(f"/home/buffett/nas_data/EDM_FAC_LOG/final_eval/eval_0826_no_ca/eval_results_{convert_type}.txt", "a") as f:
        f.write(f"Average STFT Loss: {avg_stft_loss:.4f}\n")
        f.write(f"Average Envelope Loss: {avg_envelope_loss:.4f}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDM-FAC")
    parser.add_argument("--conv_type", default="both")
    parser.add_argument("--pair_cnts", default=10, type=int)
    parser.add_argument("--iter", default=-1, type=int)
    parser.add_argument("--split", default="eval_seen_extreme_adsr") # eval_seen_normal_adsr
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
