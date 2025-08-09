"""
Final metrics for evaluating the model.

- Multi-scale STFT Loss
- F0 Frame Error Rate
- LogRMS Envelope Loss (L1)
"""

import typing
import torch
import argparse
import json
import librosa

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F


from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List
from audiotools import AudioSignal
from audiotools import STFTParams
from torch import nn


# 1. MSTFT Loss
class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        for s in self.stft_params:
            x.stft(s.window_length, s.hop_length, s.window_type)
            y.stft(s.window_length, s.hop_length, s.window_type)
            loss += self.log_weight * self.loss_fn(
                x.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
                y.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x.magnitude, y.magnitude)
        return loss


# 2. LogRMS Envelope Loss (L1)
class LogRMSEnvelopeLoss(nn.Module):
    """Computes the L1 loss between log RMS envelopes of audio signals.

    This loss measures the difference between the RMS energy envelopes
    of predicted and target audio signals in the log domain.

    Parameters
    ----------
    frame_length : int, optional
        Length of each frame for RMS calculation, by default 2048
    hop_length : int, optional
        Number of samples between consecutive frames, by default 512
    weight : float, optional
        Weight of this loss, by default 1.0
    """

    def __init__(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
        weight: float = 1.0,
    ):
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.weight = weight

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes L1 loss between log RMS envelopes.

        Parameters
        ----------
        x : AudioSignal
            Predicted signal
        y : AudioSignal
            Target signal

        Returns
        -------
        torch.Tensor
            LogRMS envelope L1 loss
        """
        # Extract audio tensors
        x_audio = x.audio_data.squeeze()  # Remove batch dimension if present
        y_audio = y.audio_data.squeeze()

        # Ensure we have 1D tensors
        if x_audio.dim() > 1:
            x_audio = x_audio.mean(dim=0)  # Average across channels if stereo
        if y_audio.dim() > 1:
            y_audio = y_audio.mean(dim=0)

        # Pad signals to ensure they have the same length
        max_length = max(x_audio.shape[-1], y_audio.shape[-1])
        x_audio = F.pad(x_audio, (0, max_length - x_audio.shape[-1]))
        y_audio = F.pad(y_audio, (0, max_length - y_audio.shape[-1]))

        # Compute RMS envelopes
        x_rms = self._compute_rms_envelope(x_audio)
        y_rms = self._compute_rms_envelope(y_audio)

        # Apply log transformation with small epsilon to avoid log(0)
        eps = 1e-8
        x_log_rms = torch.log(x_rms + eps)
        y_log_rms = torch.log(y_rms + eps)

        # Compute L1 loss
        loss = F.l1_loss(x_log_rms, y_log_rms)

        return self.weight * loss

    def _compute_rms_envelope(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute RMS envelope of audio signal using sliding windows.

        Parameters
        ----------
        audio : torch.Tensor
            Input audio signal (1D tensor)

        Returns
        -------
        torch.Tensor
            RMS envelope values
        """
        # Use unfold to create overlapping frames
        frames = audio.unfold(0, self.frame_length, self.hop_length)

        # Compute RMS for each frame
        rms_values = torch.sqrt(torch.mean(frames ** 2, dim=1))

        return rms_values


@dataclass
class F0Metrics:
    total_frames: int
    ref_voiced_frames: int
    ref_unvoiced_frames: int
    pred_voiced_frames: int
    pred_unvoiced_frames: int

    va: float
    vr: float
    vfa: float

    rpa_50c: float
    rca_50c: float
    oa_50c: float
    octave_error_rate: float  # RCA-correct but RPA-incorrect (wrong octave) / voiced ref frames

    mae_cents: Optional[float]
    rmse_cents: Optional[float]
    mean_cents: Optional[float]
    median_cents: Optional[float]
    std_cents: Optional[float]
    median_abs_cents: Optional[float]
    mae_hz: Optional[float]
    rmse_hz: Optional[float]
    pearson_r_logf0: Optional[float]

    gpe_50c_rate: Optional[float]
    gpe_20pct_rate: Optional[float]
    fpe_mean_cents: Optional[float]
    fpe_std_cents: Optional[float]


def hz_to_cents(hz: np.ndarray) -> np.ndarray:
    hz = np.asarray(hz, dtype=float)
    cents = np.full_like(hz, np.nan, dtype=float)
    valid = hz > 0
    cents[valid] = 1200.0 * np.log2(hz[valid])
    return cents


def cents_diff(pred_hz: np.ndarray, ref_hz: np.ndarray) -> np.ndarray:
    return hz_to_cents(pred_hz) - hz_to_cents(ref_hz)


def cents_mod_interval(diff_cents: np.ndarray, period: float = 1200.0) -> np.ndarray:
    x = (diff_cents + period / 2.0) % period - period / 2.0
    return np.abs(x)


def extract_f0(
    y: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    frame_length: int,
    hop_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )
        return np.asarray(f0), np.asarray(voiced_flag, dtype=bool)
    except Exception:
        # Fallback: YIN + simple RMS thresholding for voicing
        f0 = librosa.yin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True).squeeze()
        thr = np.percentile(rms, 10.0)
        voiced_flag = rms > thr
        f0 = np.asarray(f0, dtype=float)
        f0[~voiced_flag] = np.nan
        return f0, voiced_flag


def align_lengths(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def apply_silence_mask(y: np.ndarray, sr: int, hop_length: int, frame_length: int, silence_db: Optional[float]) -> np.ndarray:
    if silence_db is None:
        return None
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True).squeeze()
    # Convert to dBFS-like scale relative to max RMS
    eps = 1e-12
    rms_db = 20.0 * np.log10(np.maximum(rms, eps) / (np.max(rms) + eps))
    return rms_db < silence_db  # True where silence (mark unvoiced)


def compute_metrics(
    ref_f0: np.ndarray,
    ref_voiced: np.ndarray,
    pred_f0: np.ndarray,
    pred_voiced: np.ndarray,
    cents_threshold: float = 50.0,
) -> F0Metrics:
    ref_f0, pred_f0 = align_lengths(ref_f0, pred_f0)
    ref_voiced, pred_voiced = align_lengths(ref_voiced.astype(bool), pred_voiced.astype(bool))

    N = len(ref_f0)
    voiced_ref = ref_voiced
    unvoiced_ref = ~ref_voiced
    voiced_pred = pred_voiced
    unvoiced_pred = ~pred_voiced

    va = np.mean(ref_voiced == pred_voiced) if N > 0 else np.nan
    vr = np.sum(voiced_ref & voiced_pred) / max(1, np.sum(voiced_ref))
    vfa = np.sum(unvoiced_ref & voiced_pred) / max(1, np.sum(unvoiced_ref))

    both_voiced = voiced_ref & voiced_pred
    idx_both = np.where(both_voiced)[0]
    mae_cents = rmse_cents = mean_cents = median_cents = std_cents = median_abs_cents = mae_hz = rmse_hz = pearson_r = np.nan
    gpe_50c_rate = gpe_20pct_rate = fpe_mean_cents = fpe_std_cents = np.nan

    if idx_both.size > 0:
        diff_c = cents_diff(pred_f0[idx_both], ref_f0[idx_both])
        abs_diff_c = np.abs(diff_c)
        mae_cents = float(np.mean(abs_diff_c))
        rmse_cents = float(np.sqrt(np.mean(diff_c ** 2)))
        mean_cents = float(np.mean(diff_c))
        median_cents = float(np.median(diff_c))
        std_cents = float(np.std(diff_c))
        median_abs_cents = float(np.median(abs_diff_c))

        diff_hz = pred_f0[idx_both] - ref_f0[idx_both]
        mae_hz = float(np.mean(np.abs(diff_hz)))
        rmse_hz = float(np.sqrt(np.mean(diff_hz ** 2)))

        log_ref = np.log(ref_f0[idx_both])
        log_pred = np.log(pred_f0[idx_both])
        if np.std(log_ref) > 0 and np.std(log_pred) > 0:
            pearson_r = float(np.corrcoef(log_ref, log_pred)[0, 1])
        else:
            pearson_r = np.nan

        gpe_50c_rate = float(np.mean(abs_diff_c > cents_threshold))
        rel_err = np.abs((pred_f0[idx_both] - ref_f0[idx_both]) / ref_f0[idx_both])
        gpe_20pct_rate = float(np.mean(rel_err > 0.2))

        mask_fine = abs_diff_c <= cents_threshold
        if np.any(mask_fine):
            fpe_mean_cents = float(np.mean(diff_c[mask_fine]))
            fpe_std_cents = float(np.std(diff_c[mask_fine]))

    denom_voiced = max(1, np.sum(voiced_ref))
    denom_all = max(1, N)

    cents_all = np.full(N, np.nan, dtype=float)
    cents_all[both_voiced] = cents_diff(pred_f0[both_voiced], ref_f0[both_voiced])

    correct_rpa = (both_voiced) & (np.abs(cents_all) <= cents_threshold)
    rpa_50c = float(np.sum(correct_rpa) / denom_voiced)

    cents_mod = cents_mod_interval(cents_all, period=1200.0)
    correct_rca = (both_voiced) & (cents_mod <= cents_threshold)
    rca_50c = float(np.sum(correct_rca) / denom_voiced)

    # Octave error: right chroma but wrong absolute octave
    octave_err = correct_rca & (~correct_rpa)
    octave_error_rate = float(np.sum(octave_err) / denom_voiced)

    correct_unvoiced = unvoiced_ref & unvoiced_pred
    correct_voiced_pitch = correct_rpa
    oa_50c = float((np.sum(correct_unvoiced) + np.sum(correct_voiced_pitch)) / denom_all)

    return F0Metrics(
        total_frames=N,
        ref_voiced_frames=int(np.sum(voiced_ref)),
        ref_unvoiced_frames=int(np.sum(unvoiced_ref)),
        pred_voiced_frames=int(np.sum(voiced_pred)),
        pred_unvoiced_frames=int(np.sum(unvoiced_pred)),
        va=float(va),
        vr=float(vr),
        vfa=float(vfa),
        rpa_50c=rpa_50c,
        rca_50c=rca_50c,
        oa_50c=oa_50c,
        octave_error_rate=octave_error_rate,
        mae_cents=None if np.isnan(mae_cents) else float(mae_cents),
        rmse_cents=None if np.isnan(rmse_cents) else float(rmse_cents),
        mean_cents=None if np.isnan(mean_cents) else float(mean_cents),
        median_cents=None if np.isnan(median_cents) else float(median_cents),
        std_cents=None if np.isnan(std_cents) else float(std_cents),
        median_abs_cents=None if np.isnan(median_abs_cents) else float(median_abs_cents),
        mae_hz=None if np.isnan(mae_hz) else float(mae_hz),
        rmse_hz=None if np.isnan(rmse_hz) else float(rmse_hz),
        pearson_r_logf0=None if np.isnan(pearson_r) else float(pearson_r),
        gpe_50c_rate=None if np.isnan(gpe_50c_rate) else float(gpe_50c_rate),
        gpe_20pct_rate=None if np.isnan(gpe_20pct_rate) else float(gpe_20pct_rate),
        fpe_mean_cents=None if np.isnan(fpe_mean_cents) else float(fpe_mean_cents),
        fpe_std_cents=None if np.isnan(fpe_std_cents) else float(fpe_std_cents),
    )


def eval_f0_from_files(
    pred_path: str,
    ref_path: str,
    sr: int = 44100,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    hop_length: int = 512,
    frame_length: int = 2048,
    cents_threshold: float = 50.0,
    plot_path: Optional[str] = None,
    assume_all_voiced: bool = False,
    silence_db: Optional[float] = None,
) -> Dict:
    ref_y, _ = librosa.load(ref_path, sr=sr, mono=True)
    pred_y, _ = librosa.load(pred_path, sr=sr, mono=True)

    ref_f0, ref_voiced = extract_f0(ref_y, sr, fmin, fmax, frame_length, hop_length)
    pred_f0, pred_voiced = extract_f0(pred_y, sr, fmin, fmax, frame_length, hop_length)

    # Optional: treat finite-F0 frames as voiced (synths are often fully voiced)
    if assume_all_voiced:
        ref_voiced = np.isfinite(ref_f0)
        pred_voiced = np.isfinite(pred_f0)

    # Optional: override with silence gating (dB relative to max RMS)
    if silence_db is not None:
        ref_sil = apply_silence_mask(ref_y, sr, hop_length, frame_length, silence_db)
        pred_sil = apply_silence_mask(pred_y, sr, hop_length, frame_length, silence_db)
        if ref_sil is not None:
            # align lengths with f0
            n = min(len(ref_sil), len(ref_f0))
            ref_sil = ref_sil[:n]
            ref_f0 = ref_f0[:n]
            ref_voiced = ref_voiced[:n] & (~ref_sil)
        if pred_sil is not None:
            n = min(len(pred_sil), len(pred_f0))
            pred_sil = pred_sil[:n]
            pred_f0 = pred_f0[:n]
            pred_voiced = pred_voiced[:n] & (~pred_sil)

    metrics = compute_metrics(ref_f0, ref_voiced, pred_f0, pred_voiced, cents_threshold=cents_threshold)
    result = asdict(metrics)

    if plot_path is not None:
        n = min(len(ref_f0), len(pred_f0))
        t = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)
        plt.figure()
        plt.plot(t, ref_f0[:n], label="Ref F0 (Hz)")
        plt.plot(t, pred_f0[:n], label="Pred F0 (Hz)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.legend()
        plt.title("F0 contours")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    return result


def eval_f0_batch(
    file_pairs: List[Tuple[str, str]],
    output_dir: Optional[str] = None,
    sr: int = 44100,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    hop_length: int = 512,
    frame_length: int = 2048,
    cents_threshold: float = 50.0,
    assume_all_voiced: bool = False,
    silence_db: Optional[float] = None,
    save_plots: bool = False,
    save_individual_results: bool = True,
    aggregate_results: bool = True,
    progress_bar: bool = True,
) -> Dict:
    """
    Evaluate F0 metrics for multiple file pairs in batch.

    Args:
        file_pairs: List of (pred_path, ref_path) tuples
        output_dir: Directory to save results and plots (optional)
        sr: Sample rate
        fmin: Minimum frequency for F0 extraction
        fmax: Maximum frequency for F0 extraction
        hop_length: Hop length for F0 extraction
        frame_length: Frame length for F0 extraction
        cents_threshold: Threshold for pitch accuracy metrics
        assume_all_voiced: Whether to assume all frames with finite F0 are voiced
        silence_db: Silence threshold in dB (optional)
        save_plots: Whether to save F0 contour plots
        save_individual_results: Whether to save individual file results
        aggregate_results: Whether to compute aggregated statistics
        progress_bar: Whether to show progress bar

    Returns:
        Dictionary containing individual results and aggregated statistics
    """
    import os
    from tqdm import tqdm
    import json

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if save_plots:
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

    results = {
        "individual_results": [],
        "aggregated_results": None,
        "file_pairs": file_pairs,
        "parameters": {
            "sr": sr,
            "fmin": fmin,
            "fmax": fmax,
            "hop_length": hop_length,
            "frame_length": frame_length,
            "cents_threshold": cents_threshold,
            "assume_all_voiced": assume_all_voiced,
            "silence_db": silence_db,
        }
    }

    # Process each file pair
    iterator = tqdm(file_pairs, desc="Evaluating F0") if progress_bar else file_pairs

    for i, (pred_path, ref_path) in enumerate(iterator):
        try:
            # Determine plot path if saving plots
            plot_path = None
            if save_plots and output_dir is not None:
                pred_name = os.path.splitext(os.path.basename(pred_path))[0]
                ref_name = os.path.splitext(os.path.basename(ref_path))[0]
                plot_path = os.path.join(plots_dir, f"{pred_name}_vs_{ref_name}_f0.png")

            # Evaluate F0 metrics
            metrics = eval_f0_from_files(
                pred_path=pred_path,
                ref_path=ref_path,
                sr=sr,
                fmin=fmin,
                fmax=fmax,
                hop_length=hop_length,
                frame_length=frame_length,
                cents_threshold=cents_threshold,
                plot_path=plot_path,
                assume_all_voiced=assume_all_voiced,
                silence_db=silence_db,
            )

            # Add file information
            file_result = {
                "pred_path": pred_path,
                "ref_path": ref_path,
                "metrics": metrics,
                "success": True,
                "error": None
            }

            results["individual_results"].append(file_result)

        except Exception as e:
            # Handle errors gracefully
            file_result = {
                "pred_path": pred_path,
                "ref_path": ref_path,
                "metrics": None,
                "success": False,
                "error": str(e)
            }
            results["individual_results"].append(file_result)

            if progress_bar:
                print(f"Error processing {pred_path} vs {ref_path}: {e}")

    # Save individual results if requested
    if save_individual_results and output_dir is not None:
        individual_results_path = os.path.join(output_dir, "individual_results.json")
        with open(individual_results_path, "w") as f:
            json.dump(results["individual_results"], f, indent=2)

    # Compute aggregated statistics if requested
    if aggregate_results:
        successful_results = [r for r in results["individual_results"] if r["success"]]

        if successful_results:
            aggregated = aggregate_f0_metrics(successful_results)
            results["aggregated_results"] = aggregated

            # Save aggregated results
            if output_dir is not None:
                aggregated_path = os.path.join(output_dir, "aggregated_results.json")
                with open(aggregated_path, "w") as f:
                    json.dump(aggregated, f, indent=2)
        else:
            results["aggregated_results"] = None

    # Save complete results
    if output_dir is not None:
        complete_results_path = os.path.join(output_dir, "complete_results.json")
        with open(complete_results_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def aggregate_f0_metrics(individual_results: List[Dict]) -> Dict:
    """
    Aggregate F0 metrics from individual results.

    Args:
        individual_results: List of individual result dictionaries

    Returns:
        Dictionary containing aggregated statistics
    """
    if not individual_results:
        return None

    # Extract all metrics
    metrics_list = [r["metrics"] for r in individual_results if r["success"]]

    # Initialize aggregators
    numeric_fields = [
        "va", "vr", "vfa", "rpa_50c", "rca_50c", "oa_50c", "octave_error_rate",
        "mae_cents", "rmse_cents", "mean_cents", "median_cents", "std_cents",
        "median_abs_cents", "mae_hz", "rmse_hz", "pearson_r_logf0",
        "gpe_50c_rate", "gpe_20pct_rate", "fpe_mean_cents", "fpe_std_cents"
    ]

    integer_fields = [
        "total_frames", "ref_voiced_frames", "ref_unvoiced_frames",
        "pred_voiced_frames", "pred_unvoiced_frames"
    ]

    aggregated = {
        "num_files": len(metrics_list),
        "summary_stats": {},
        "per_file_results": []
    }

    # Collect valid values for each metric
    valid_values = {field: [] for field in numeric_fields}

    for i, metrics in enumerate(metrics_list):
        file_summary = {"file_index": i}

        # Process numeric fields
        for field in numeric_fields:
            value = metrics.get(field)
            if value is not None and not np.isnan(value):
                valid_values[field].append(value)
                file_summary[field] = value
            else:
                file_summary[field] = None

        # Process integer fields
        for field in integer_fields:
            value = metrics.get(field)
            if value is not None:
                file_summary[field] = value
            else:
                file_summary[field] = None

        aggregated["per_file_results"].append(file_summary)

    # Compute summary statistics
    for field, values in valid_values.items():
        if values:
            values = np.array(values)
            aggregated["summary_stats"][field] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "num_valid": len(values),
                "num_total": len(metrics_list)
            }
        else:
            aggregated["summary_stats"][field] = {
                "mean": None,
                "std": None,
                "median": None,
                "min": None,
                "max": None,
                "num_valid": 0,
                "num_total": len(metrics_list)
            }

    return aggregated


def find_audio_file_pairs(
    pred_dir: str,
    ref_dir: str,
    pred_suffix: str = "",
    ref_suffix: str = "",
    audio_extensions: List[str] = None
) -> List[Tuple[str, str]]:
    """
    Find matching audio file pairs between prediction and reference directories.

    Args:
        pred_dir: Directory containing prediction files
        ref_dir: Directory containing reference files
        pred_suffix: Suffix to add to prediction filenames
        ref_suffix: Suffix to add to reference filenames
        audio_extensions: List of audio file extensions to consider

    Returns:
        List of (pred_path, ref_path) tuples
    """
    import os
    import glob

    if audio_extensions is None:
        audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]

    # Get all audio files in reference directory
    ref_files = []
    for ext in audio_extensions:
        ref_files.extend(glob.glob(os.path.join(ref_dir, f"*{ext}")))
        ref_files.extend(glob.glob(os.path.join(ref_dir, f"*{ext.upper()}")))

    file_pairs = []

    for ref_file in ref_files:
        # Extract base filename
        ref_basename = os.path.basename(ref_file)
        ref_name, ref_ext = os.path.splitext(ref_basename)

        # Remove reference suffix if present
        if ref_suffix and ref_name.endswith(ref_suffix):
            ref_name = ref_name[:-len(ref_suffix)]

        # Add prediction suffix
        pred_name = ref_name + pred_suffix

        # Look for corresponding prediction file
        for ext in audio_extensions:
            pred_file = os.path.join(pred_dir, f"{pred_name}{ext}")
            if os.path.exists(pred_file):
                file_pairs.append((pred_file, ref_file))
                break
            pred_file = os.path.join(pred_dir, f"{pred_name}{ext.upper()}")
            if os.path.exists(pred_file):
                file_pairs.append((pred_file, ref_file))
                break

    return file_pairs


def eval_f0_from_directory(
    pred_dir: str,
    ref_dir: str,
    output_dir: str,
    pred_suffix: str = "",
    ref_suffix: str = "",
    audio_extensions: List[str] = None,
    **kwargs
) -> Dict:
    """
    Evaluate F0 metrics for all matching audio files in two directories.

    Args:
        pred_dir: Directory containing prediction files
        ref_dir: Directory containing reference files
        output_dir: Directory to save results
        pred_suffix: Suffix to add to prediction filenames
        ref_suffix: Suffix to add to reference filenames
        audio_extensions: List of audio file extensions to consider
        **kwargs: Additional arguments passed to eval_f0_batch

    Returns:
        Dictionary containing evaluation results
    """
    # Find matching file pairs
    file_pairs = find_audio_file_pairs(
        pred_dir, ref_dir, pred_suffix, ref_suffix, audio_extensions
    )

    if not file_pairs:
        raise ValueError(f"No matching audio files found between {pred_dir} and {ref_dir}")

    print(f"Found {len(file_pairs)} matching file pairs")

    # Run batch evaluation
    results = eval_f0_batch(
        file_pairs=file_pairs,
        output_dir=output_dir,
        **kwargs
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute F0 evaluation metrics for monophonic audio (music/synth-friendly).")
    parser.add_argument("--pred", required=True, help="Path to predicted audio (wav)")
    parser.add_argument("--ref", required=True, help="Path to reference/ground-truth audio (wav)")
    parser.add_argument("--sr", type=int, default=24000, help="Target sample rate for loading")
    parser.add_argument("--fmin", type=float, default=50.0, help="Minimum F0 in Hz")
    parser.add_argument("--fmax", type=float, default=2000.0, help="Maximum F0 in Hz")
    parser.add_argument("--hop_length", type=int, default=512, help="Hop length for analysis")
    parser.add_argument("--frame_length", type=int, default=2048, help="Frame length for analysis")
    parser.add_argument("--cents_threshold", type=float, default=50.0, help="Threshold (in cents) for accuracy metrics")
    parser.add_argument("--out_prefix", default="f0_eval", help="Prefix for output files (JSON/CSV)")
    parser.add_argument("--plot", action="store_true", help="If set, saves an F0 overlay plot PNG")
    parser.add_argument("--assume_all_voiced", action="store_true", help="Treat finite-F0 frames as voiced (good for sustained synths)")
    parser.add_argument("--silence_db", type=float, default=None, help="Mark frames below this dB (relative to max RMS) as unvoiced, e.g., -40")

    args = parser.parse_args()

    plot_path = f"{args.out_prefix}.png" if args.plot else None
    metrics = eval_f0_from_files(
        pred_path=args.pred,
        ref_path=args.ref,
        sr=args.sr,
        fmin=args.fmin,
        fmax=args.fmax,
        hop_length=args.hop_length,
        frame_length=args.frame_length,
        cents_threshold=args.cents_threshold,
        plot_path=plot_path,
        assume_all_voiced=args.assume_all_voiced,
        silence_db=args.silence_db,
    )

    json_path = f"{args.out_prefix}.json"
    csv_path = f"{args.out_prefix}.csv"

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    df = pd.DataFrame([metrics])
    df.to_csv(csv_path, index=False)

    print(f"Saved metrics to: {json_path} and {csv_path}")
    if plot_path is not None:
        print(f"Saved F0 plot to: {plot_path}")


if __name__ == "__main__":
    main()
