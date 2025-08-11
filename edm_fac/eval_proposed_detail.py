import argparse
import json
import torch
import torch.nn as nn
import os
import librosa
import dac
import warnings
import mir_eval
import numpy as np
import multiprocessing as mp
from functools import partial
import time

from tqdm import tqdm
from torch.utils.data import DataLoader
from audiotools import AudioSignal
from utils import yaml_config_hook
from evaluate_metrics import MultiScaleSTFTLoss, LogRMSEnvelopeLoss, F0Metrics, eval_f0_from_files, eval_f0_batch, aggregate_f0_metrics
from dataset import EDM_MN_Test_Dataset
from dac.nn.loss import MelSpectrogramLoss, L1Loss
from typing import List, Tuple, Dict

# Filter out specific warnings
warnings.filterwarnings("ignore", message="stft_data changed shape")
warnings.filterwarnings("ignore", message="Audio amplitude > 1 clipped when saving")

# Configuration options for F0 multiprocessing:
# - use_f0_multiprocessing: Enable/disable multiprocessing for F0 computation (default: True)
# - max_f0_workers: Maximum number of worker processes (default: 8)
# - min_batch_for_multiprocessing: Minimum batch size to use multiprocessing (default: 4)

LENGTH = 44100*3


def extract_f0(y: np.ndarray, sr: int, fmin: float, fmax: float, frame_length: int, hop_length: int) -> np.ndarray:
    """Extract monophonic F0 using librosa.pyin; returns Hz with NaN for unvoiced."""
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    f0, _, _ = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )
    return np.asarray(f0, dtype=float)



def compute_mir_melody_scores(
    pred_y: np.ndarray,
    ref_y: np.ndarray,
    sr: int,
    hop_length: int,
    frame_length: int,
    fmin: float,
    fmax: float,
    cent_tolerance: float,
) -> Dict[str, float]:
    """Compute MIR melody metrics using pYIN F0 extraction and mir_eval.

    Returns a dict with keys like 'Overall Accuracy', 'Voicing Recall', etc.
    """
    ref_f0 = extract_f0(ref_y, sr, fmin, fmax, frame_length, hop_length)
    pred_f0 = extract_f0(pred_y, sr, fmin, fmax, frame_length, hop_length)

    n = min(len(ref_f0), len(pred_f0))
    if n == 0:
        return {
            "Voicing Recall": 0.0,
            "Voicing False Alarm": 0.0,
            "Raw Pitch Accuracy": 0.0,
            "Raw Chroma Accuracy": 0.0,
            "Overall Accuracy": 0.0,
        }

    ref_f0 = ref_f0[:n]
    pred_f0 = pred_f0[:n]
    frames = np.arange(n)
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    ref_freq = np.nan_to_num(ref_f0, nan=0.0)
    est_freq = np.nan_to_num(pred_f0, nan=0.0)

    scores = mir_eval.melody.evaluate(
        ref_time=times,
        ref_freq=ref_freq,
        est_time=times,
        est_freq=est_freq,
        cent_tolerance=cent_tolerance,
    )
    return scores


def compute_f0_worker(args_tuple):
    """Worker function for multiprocessing F0 computation.
    
    Args:
        args_tuple: Tuple containing (pred_y, ref_y, sr, hop_length, frame_length, fmin, fmax, cent_tolerance)
    
    Returns:
        Dictionary with F0 metrics
    """
    pred_y, ref_y, sr, hop_length, frame_length, fmin, fmax, cent_tolerance = args_tuple
    
    try:
        # Input validation
        if pred_y is None or ref_y is None:
            raise ValueError("Audio data is None")
        
        if len(pred_y) == 0 or len(ref_y) == 0:
            raise ValueError("Audio data is empty")
        
        # Ensure audio data is 1D
        if pred_y.ndim > 1:
            pred_y = pred_y.squeeze()
        if ref_y.ndim > 1:
            ref_y = ref_y.squeeze()
        
        scores = compute_mir_melody_scores(
            pred_y=pred_y,
            ref_y=ref_y,
            sr=sr,
            hop_length=hop_length,
            frame_length=frame_length,
            fmin=fmin,
            fmax=fmax,
            cent_tolerance=cent_tolerance,
        )
        return scores
    except Exception as e:
        # Return default scores if computation fails
        print(f"Warning: F0 computation failed for one sample: {e}")
        return {
            "Voicing Recall": 0.0,
            "Voicing False Alarm": 0.0,
            "Raw Pitch Accuracy": 0.0,
            "Raw Chroma Accuracy": 0.0,
            "Overall Accuracy": 0.0,
        }


class EDMFACInference:
    def __init__(
        self,
        checkpoint_path,
        config_path="configs/config.yaml",
        device="cuda",
    ):
        """
        Initialize the EDM-FAC inference model

        Args:
            checkpoint_path: Path to the trained model checkpoint
            config_path: Path to the configuration file
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load configuration
        self.config = yaml_config_hook(config_path)
        self.args = argparse.Namespace(**self.config)

        # Get parameters
        self.sample_rate = self.args.sample_rate
        self.hop_length = self.args.hop_length

        # Initialize model
        self.generator = dac.model.MyDAC(
            encoder_dim=self.args.encoder_dim,
            encoder_rates=self.args.encoder_rates,
            latent_dim=self.args.latent_dim,
            decoder_dim=self.args.decoder_dim,
            decoder_rates=self.args.decoder_rates,
            adsr_enc_dim=self.args.adsr_enc_dim,
            adsr_enc_ver=self.args.adsr_enc_ver,
            sample_rate=self.args.sample_rate,
            timbre_classes=self.args.timbre_classes,
            adsr_classes=self.args.adsr_classes,
            pitch_nums=self.args.max_note - self.args.min_note + 1, # 88
            use_gr_content=self.args.use_gr_content,
            use_gr_adsr=self.args.use_gr_adsr,
            use_gr_timbre=self.args.use_gr_timbre,
            use_FiLM=self.args.use_FiLM,
            rule_based_adsr_folding=self.args.rule_based_adsr_folding,
            use_cross_attn=self.args.use_cross_attn,
        ).to(self.device)

        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        self.generator.eval()

        # Load losses for evaluation
        # 1) Multi-scale STFT Loss
        self.stft_loss = MultiScaleSTFTLoss().to(self.device)

        # 2) Envelope L1 Loss (Log-RMS envelope)
        self.envelope_loss = LogRMSEnvelopeLoss().to(self.device)

        # 3) Mel-Spectrogram Loss (match training settings)
        self.mel_loss = MelSpectrogramLoss(
            n_mels=[5, 10, 20, 40, 80, 160, 320],
            window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
            mel_fmin=[0, 0, 0, 0, 0, 0, 0],
            mel_fmax=[None, None, None, None, None, None, None],
            pow=1.0,
            mag_weight=0.0,
        ).to(self.device)

        # 4) L1 waveform loss
        self.l1_eval_loss = L1Loss().to(self.device)

        print(f"EDM-FAC model loaded on {self.device}")


    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'generator_state_dict' in checkpoint:
            # Training checkpoint format with separate state dicts
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            print(f"Loaded checkpoint from step {checkpoint.get('iter', 'unknown')}")
        elif 'generator' in checkpoint:
            # Alternative format
            self.generator.load_state_dict(checkpoint['generator'])
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        elif 'model_state_dict' in checkpoint:
            # Simple format
            self.generator.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the checkpoint contains the model state dict directly
            self.generator.load_state_dict(checkpoint)

        print("Model weights loaded successfully")

    def load_audio(self, audio_path):
        """
        Load and preprocess audio file

        Args:
            audio_path: Path to audio file

        Returns:
            AudioSignal object
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        audio, _ = librosa.load(audio_path, sr=self.args.sample_rate, mono=True)
        audio = audio[:LENGTH]

        # Convert to AudioSignal
        audio_signal = AudioSignal(torch.tensor(audio).unsqueeze(0).unsqueeze(0), self.args.sample_rate)
        return audio_signal


    @torch.no_grad()
    def evaluate_loader(self, data_loader: DataLoader, output_dir: str, recon: bool = False):
        os.makedirs(output_dir, exist_ok=True)

        # Aggregators
        overall = {"stft": 0.0, "l1": 0.0, "mel": 0.0, "env": 0.0, "num": 0}

        # Aggregators for MIR melody F0 metrics
        f0_overall = {"va": 0.0, "vfa": 0.0, "rpa": 0.0, "rca": 0.0, "oa": 0.0, "num": 0}

        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move to device
            orig_audio = batch['orig_audio'].to(self.device)
            ref_audio = batch['ref_audio'].to(self.device)
            target_audio = batch['gt_audio'].to(self.device)

            out = self.generator.conversion(
                orig_audio=orig_audio.audio_data,
                ref_audio=None if recon else ref_audio.audio_data,
                convert_type="reconstruction" if recon else "both",
            )

            recons = AudioSignal(out["audio"], self.args.sample_rate)

            # Losses
            stft_val = self.stft_loss(recons, target_audio)
            l1_val = self.l1_eval_loss(recons, target_audio)
            mel_val = self.mel_loss(recons, target_audio)
            env_val = self.envelope_loss(recons, target_audio)

            bs = int(out["audio"].shape[0])

            overall["stft"] += float(stft_val.item()) * bs
            overall["l1"] += float(l1_val.item()) * bs
            overall["mel"] += float(mel_val.item()) * bs
            overall["env"] += float(env_val.item()) * bs
            overall["num"] += bs


            # # F0 Metrics (mir_eval-based, like eval_mir_test.py) - Using multiprocessing
            # # Configurable parameters
            # fmin = getattr(self.args, 'fmin', 50.0)
            # fmax = getattr(self.args, 'fmax', 2000.0)
            # frame_length = getattr(self.args, 'frame_length', 2048)
            # hop_length = self.args.hop_length
            # cent_tol = getattr(self.args, 'cents_threshold', 50.0)

            # # Compute per-item F0 metrics in the batch using multiprocessing
            # recon_np = recons.audio_data.detach().cpu().numpy()  # (B, 1, T)
            # tgt_np = target_audio.audio_data.detach().cpu().numpy()  # (B, 1, T)

            # batch_size = recon_np.shape[0]
            
            # # Prepare arguments for multiprocessing
            # f0_args_list = []
            # for i in range(batch_size):
            #     pred_y = recon_np[i].squeeze()
            #     ref_y = tgt_np[i].squeeze()
            #     args_tuple = (pred_y, ref_y, self.args.sample_rate, hop_length, frame_length, fmin, fmax, cent_tol)
            #     f0_args_list.append(args_tuple)
            
            # # Use multiprocessing to compute F0 metrics
            # use_multiprocessing = getattr(self.args, 'use_f0_multiprocessing', True)
            # min_batch_for_multiprocessing = getattr(self.args, 'min_batch_for_multiprocessing', 4)  # Only use MP for batches >= 4
            
            # start_time = time.time()
            
            # if use_multiprocessing and batch_size >= min_batch_for_multiprocessing:
            #     num_workers = min(mp.cpu_count(), batch_size, getattr(self.args, 'max_f0_workers', 8))
            #     if num_workers > 1:
            #         print(f"Computing F0 metrics using {num_workers} workers...")
            #         with mp.Pool(processes=num_workers) as pool:
            #             f0_scores_list = list(tqdm(
            #                 pool.imap(compute_f0_worker, f0_args_list),
            #                 total=len(f0_args_list),
            #                 desc="F0 computation",
            #                 leave=False
            #             ))
            #     else:
            #         # Fallback to sequential processing for small batches
            #         f0_scores_list = [compute_f0_worker(args) for args in f0_args_list]
            # else:
            #     # Sequential processing for small batches or when MP is disabled
            #     f0_scores_list = [compute_f0_worker(args) for args in f0_args_list]
            
            # elapsed_time = time.time() - start_time
            # if batch_size >= min_batch_for_multiprocessing and use_multiprocessing:
            #     print(f"F0 computation completed in {elapsed_time:.2f}s for {batch_size} samples")
            
            # # Aggregate F0 metrics from multiprocessing results
            # for scores in f0_scores_list:
            #     va = float(scores.get("Voicing Recall", 0.0))
            #     vfa = float(scores.get("Voicing False Alarm", 0.0))
            #     rpa = float(scores.get("Raw Pitch Accuracy", 0.0))
            #     rca = float(scores.get("Raw Chroma Accuracy", 0.0))
            #     oa = float(scores.get("Overall Accuracy", 0.0))

            #     f0_overall["va"] += va
            #     f0_overall["vfa"] += vfa
            #     f0_overall["rpa"] += rpa
            #     f0_overall["rca"] += rca
            #     f0_overall["oa"] += oa
            #     f0_overall["num"] += 1




        n_all = max(1, overall["num"])
        results = {
            "num_total_samples": overall["num"],
            "overall": {
                "stft_loss": overall["stft"] / n_all,
                "l1_loss": overall["l1"] / n_all,
                "mel_loss": overall["mel"] / n_all,
                "envelope_loss": overall["env"] / n_all,
                # "f0_mir": {
                #     "va": f0_overall["va"] / max(1, f0_overall["num"]),
                #     "vfa": f0_overall["vfa"] / max(1, f0_overall["num"]),
                #     "rpa": f0_overall["rpa"] / max(1, f0_overall["num"]),
                #     "rca": f0_overall["rca"] / max(1, f0_overall["num"]),
                #     "oa": f0_overall["oa"] / max(1, f0_overall["num"]),
                #     "num_samples": f0_overall["num"],
                # },
            },
        }

        # Save metadata immediately here as well
        metadata_path = os.path.join(output_dir, f"metadata_{'recon' if recon else 'conv'}.json")
        with open(metadata_path, "w") as f:
            json.dump(results, f, indent=4)

        return results


    def evaluate_f0_batch(
        self,
        file_pairs: List[Tuple[str, str]],
        output_dir: str,
        save_plots: bool = False,
        **kwargs
    ) -> Dict:

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Run batch F0 evaluation
        results = eval_f0_batch(
            file_pairs=file_pairs,
            output_dir=output_dir,
            sr=self.args.sample_rate,
            fmin=getattr(self.args, 'fmin', 50.0),
            fmax=getattr(self.args, 'fmax', 2000.0),
            hop_length=self.args.hop_length,
            frame_length=self.args.frame_length,
            cents_threshold=getattr(self.args, 'cents_threshold', 50.0),
            assume_all_voiced=getattr(self.args, 'assume_all_voiced', True),
            silence_db=getattr(self.args, 'silence_db', -45),
            save_plots=save_plots,
            **kwargs
        )

        return results


def main():
    parser = argparse.ArgumentParser(description="EDM-FAC Evaluation on Validation/Test Loader")

    # Arguments
    parser.add_argument("--device", default="cuda:0", help="Device to use for inference")
    parser.add_argument("--bs", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--checkpoint", type=str, default="/home/buffett/nas_data/EDM_FAC_LOG/0804_proposed/ckpt/checkpoint_latest.pt", help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config_proposed.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="/home/buffett/nas_data/EDM_FAC_LOG/final_eval/0804_proposed/detail", help="Output directory for results/metadata")
    
    # F0 Multiprocessing options
    parser.add_argument("--use_f0_multiprocessing", action="store_true", default=True, help="Enable multiprocessing for F0 computation")
    parser.add_argument("--max_f0_workers", type=int, default=8, help="Maximum number of F0 computation workers")
    parser.add_argument("--min_batch_for_multiprocessing", type=int, default=4, help="Minimum batch size to use multiprocessing")

    # Parse initial arguments to get config path
    initial_args, _ = parser.parse_known_args()

    # Load config and add config parameters as arguments
    config = yaml_config_hook(initial_args.config)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    # Parse all arguments including config parameters
    args = parser.parse_args()

    # Ibnference Model
    model = EDMFACInference(args.checkpoint, args.config, args.device)

    # Build Evaluation Dataset/Loader from Model Config
    test_dataset_recon = EDM_MN_Test_Dataset(
        root_path=args.root_path,
        duration=args.duration,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        split="evaluation",
        recon=True,
    )

    test_loader_recon = DataLoader(
        test_dataset_recon,
        shuffle=False,
        batch_size=args.bs, # args.batch_size
        num_workers=16, # args.num_workers
        collate_fn=test_dataset_recon.collate,
        pin_memory=True,
        drop_last=False,
    )
    
    test_dataset = EDM_MN_Test_Dataset(
        root_path=args.root_path,
        duration=args.duration,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        split="evaluation",
        recon=False,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.bs, # args.batch_size
        num_workers=16, # args.num_workers
        collate_fn=test_dataset.collate,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Evaluation batch size: {args.bs}")
    print(f"F0 Multiprocessing: {'Enabled' if args.use_f0_multiprocessing else 'Disabled'}")
    if args.use_f0_multiprocessing:
        print(f"Max F0 workers: {args.max_f0_workers}")
        print(f"Min batch size for multiprocessing: {args.min_batch_for_multiprocessing}")
    print(f"Available CPU cores: {mp.cpu_count()}")

    # Perform evaluation over loader and save metadata
    # results_recon = model.evaluate_loader(test_loader_recon, args.output_dir, recon=True)
    results = model.evaluate_loader(test_loader, args.output_dir, recon=False)
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
