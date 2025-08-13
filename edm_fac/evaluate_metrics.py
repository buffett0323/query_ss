"""
Final metrics for evaluating the model.

- Multi-scale STFT Loss
- F0 Frame Error Rate
- LogRMS Envelope Loss (L1)
"""

import typing
import torch
import librosa
import torchcrepe
import torchaudio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F


from typing import List
from audiotools import AudioSignal
from audiotools import STFTParams
from torch import nn
import multiprocessing as mp
from typing import List


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


# 3. F0 Evaluation Loss
class F0EvalLoss(nn.Module):
    """Computes F0 correlation and RMSE metrics between reference and estimated audio signals.

    This loss measures the pitch accuracy between predicted and target audio signals
    using the CREPE model for F0 extraction and computes correlation and RMSE metrics.

    Parameters
    ----------
    hop_length : int, optional
        Number of samples between consecutive F0 frames, by default 160
    fmin : float, optional
        Minimum frequency for F0 detection, by default 50.0 Hz
    fmax : float, optional
        Maximum frequency for F0 detection, by default 1100.0 Hz
    model_size : str, optional
        CREPE model size ('tiny' or 'full'), by default 'full'
    voicing_thresh : float, optional
        Confidence threshold for voicing detection, by default 0.5
    weight : float, optional
        Weight of this loss, by default 1.0
    """

    def __init__(
        self,
        hop_length: int = 512,
        fmin: float = 50.0,
        fmax: float = 1100.0,
        model_size: str = 'full',
        voicing_thresh: float = 0.5,
        weight: float = 1.0,
        device: torch.device = torch.device('cpu'),
        sr_in: int = 44100,
        sr_out: int = 16000,
        resample_to_16k: bool = True,
    ):
        super().__init__()
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.model_size = model_size
        self.voicing_thresh = voicing_thresh
        self.weight = weight
        self.sr_in = sr_in
        self.sr_out = sr_out
        self.resampler = torchaudio.transforms.Resample(orig_freq=sr_in, new_freq=sr_out).to(device)
        self.resample_to_16k = resample_to_16k

    def _validate_audio_signals(self, x: AudioSignal, y: AudioSignal) -> bool:
        """Validate audio signals before F0 processing.

        Returns
        -------
        bool
            True if signals are valid for F0 processing
        """
        x_audio = x.audio_data.squeeze()
        y_audio = y.audio_data.squeeze()

        # Check if signals are too short
        min_length = self.hop_length * 4  # Need at least 4 frames
        if x_audio.shape[-1] < min_length or y_audio.shape[-1] < min_length:
            return False

        # Check if signals are too quiet (likely silence)
        x_rms = torch.sqrt(torch.mean(x_audio ** 2))
        y_rms = torch.sqrt(torch.mean(y_audio ** 2))

        if x_rms < 1e-6 or y_rms < 1e-6:
            return False

        return True

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes F0 correlation and RMSE between predicted and target signals.

        Parameters
        ----------
        x : AudioSignal
            Predicted signal
        y : AudioSignal
            Target signal

        Returns
        -------
        torch.Tensor
            F0 evaluation loss (negative correlation + RMSE penalty)
        """
        # Validate audio signals first
        if not self._validate_audio_signals(x, y):
            # Return high penalty for invalid signals
            device = x.audio_data.device
            return torch.tensor(1.0, device=device, requires_grad=True)

        # Extract audio tensors and ensure they're on the same device
        device = x.audio_data.device
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

        # Add batch dimension for processing
        x_batch = x_audio.unsqueeze(0)  # (1, T)
        y_batch = y_audio.unsqueeze(0)  # (1, T)

        # Compute F0 metrics
        per_sample, summary = self._compute_f0_metrics(x_batch, y_batch, device)

        # Extract metrics for loss computation
        f0_corr = summary["F0CORR_mean"]
        f0_rmse = summary["F0RMSE_cents_mean"]

        # Convert to tensors and handle NaN values
        if np.isnan(f0_corr) or np.isnan(f0_rmse):
            # Return zero loss if metrics are invalid
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Convert to tensors
        f0_corr_tensor = torch.tensor(f0_corr, device=device, dtype=torch.float32)
        f0_rmse_tensor = torch.tensor(f0_rmse, device=device, dtype=torch.float32)

        # Compute loss: negative correlation (higher correlation = lower loss) + RMSE penalty
        # Normalize RMSE to reasonable range (typical RMSE is 0-100 cents)
        normalized_rmse = f0_rmse_tensor / 100.0

        # Loss = -correlation + RMSE penalty
        # Higher correlation reduces loss, higher RMSE increases loss
        loss = -f0_corr_tensor + normalized_rmse

        return self.weight * loss

    @torch.no_grad()
    def _compute_f0_metrics(
        self,
        ref_wavs: torch.Tensor,
        est_wavs: torch.Tensor,
        device: torch.device,
    ):
        """Compute F0 correlation and RMSE metrics efficiently."""
        assert ref_wavs.shape == est_wavs.shape, "ref/est shape must match (B,T)"
        B, T = ref_wavs.shape

        # Combine both ref and est into single batch for single CREPE processing
        combined_wavs = torch.cat([ref_wavs, est_wavs], dim=0)  # (2B, T)

        # Process combined batch in one pass - much more efficient
        f0_combined = self._extract_f0_batch(
            combined_wavs,
            device=device,
        )

        # Split results back to ref and est
        f0_ref = f0_combined[:B]
        f0_est = f0_combined[B:]

        # Calculate metrics efficiently
        per_sample = []
        corr_vals, rmse_vals = [], []

        for i in range(B):
            # Length alignment to shortest
            L = min(len(f0_ref[i]), len(f0_est[i]))
            r, e = f0corr_rmse_from_hz(f0_ref[i][:L], f0_est[i][:L])
            per_sample.append({"idx": i, "F0CORR": r, "F0RMSE_cents": e})
            if not (np.isnan(r) or np.isnan(e)):
                corr_vals.append(r)
                rmse_vals.append(e)

        # Handle case where no valid metrics were computed
        if not corr_vals:
            # Provide fallback values instead of NaN
            summary = {
                "F0CORR_mean": 0.0,  # No correlation = 0
                "F0RMSE_cents_mean": 100.0,  # High RMSE as penalty
                "hop_ms": 1000 * self.hop_length / self.sr_out
            }
        else:
            summary = {
                "F0CORR_mean": float(np.mean(corr_vals)),
                "F0RMSE_cents_mean": float(np.mean(rmse_vals)),
                "hop_ms": 1000 * self.hop_length / self.sr_out
            }
        return per_sample, summary

    @torch.no_grad()
    def _extract_f0_batch(
        self,
        wav_batch: torch.Tensor,
        device: torch.device,
    ):
        """Extract F0 values from audio batch using CREPE."""
        B, T = wav_batch.shape
        wav_batch = wav_batch.to(device)
        if self.resample_to_16k:
            audio = self.resampler(wav_batch)
        else:
            audio = wav_batch
        audio = audio.unsqueeze(1)

        # Process all samples in batch for better GPU utilization
        f0_list = []

        for i in range(B):
            # Check audio quality before processing
            audio_i = audio[i]
            audio_rms = torch.sqrt(torch.mean(audio_i ** 2))

            # Skip F0 extraction if audio is too quiet (likely silence)
            if audio_rms < 1e-6:
                # Return all NaN for silent audio
                print("silent")
                n_frames = (T + self.hop_length - 1) // self.hop_length
                f0_np = np.full(n_frames, np.nan, dtype=float)
                f0_list.append(f0_np)
                continue

            # predict returns (b, n_frames)
            f0, pd = torchcrepe.predict(
                audio_i,
                sample_rate=self.sr_out,
                hop_length=self.hop_length,
                fmin=self.fmin,
                fmax=self.fmax,
                model=self.model_size,
                batch_size=None,
                device=device,
                pad=True,
                return_periodicity=True
            )

            # Apply voicing mask more carefully
            # Only mask frames with very low confidence to avoid excessive NaN values
            if self.voicing_thresh > 0.0:
                mask = (pd < self.voicing_thresh)
                f0 = f0.masked_fill(mask, float('nan'))

            # Convert to numpy immediately to free GPU memory
            f0_np = f0.detach().cpu().float().numpy()
            f0_list.append(f0_np)

        return f0_list

    def get_metrics(self, x: AudioSignal, y: AudioSignal):
        """Get detailed F0 metrics without computing loss.

        Returns
        -------
        dict
            Dictionary containing F0 correlation, RMSE, and other metrics
        """
        # Validate audio signals first
        if not self._validate_audio_signals(x, y):
            # Return fallback metrics for invalid signals
            return {
                "per_sample": [{"idx": 0, "F0CORR": 0.0, "F0RMSE_cents": 100.0}],
                "summary": {
                    "F0CORR_mean": 0.0,
                    "F0RMSE_cents_mean": 100.0,
                    "hop_ms": 1000 * self.hop_length / self.sr_out
                },
                "f0_corr": 0.0,
                "f0_rmse": 100.0,
                "hop_ms": 1000 * self.hop_length / self.sr_out
            }

        device = x.audio_data.device
        x_audio = x.audio_data.squeeze()
        y_audio = y.audio_data.squeeze()

        # Ensure we have 1D tensors
        if x_audio.dim() > 1:
            x_audio = x_audio.mean(dim=0)
        if y_audio.dim() > 1:
            y_audio = y_audio.mean(dim=0)

        # Pad signals to ensure they have the same length
        max_length = max(x_audio.shape[-1], y_audio.shape[-1])
        x_audio = F.pad(x_audio, (0, max_length - x_audio.shape[-1]))
        y_audio = F.pad(y_audio, (0, max_length - y_audio.shape[-1]))

        # Add batch dimension for processing
        x_batch = x_audio.unsqueeze(0)
        y_batch = y_audio.unsqueeze(0)

        # Compute metrics
        per_sample, summary = self._compute_f0_metrics(x_batch, y_batch, device)

        return {
            "per_sample": per_sample,
            "summary": summary,
            "f0_corr": summary["F0CORR_mean"],
            "f0_rmse": summary["F0RMSE_cents_mean"],
            "hop_ms": summary["hop_ms"]
        }


# ---------- F0 評估器類別 ----------

def _f0_evaluation_worker(args_tuple):
    """Worker function for multiprocessing F0 evaluation.

    Args:
        args_tuple: Tuple containing (ref_audio, est_audio, evaluator_params, use_dtw, idx)

    Returns:
        tuple: (idx, correlation, rmse_cents)
    """
    ref_audio, est_audio, evaluator_params, use_dtw, idx = args_tuple

    # Create evaluator instance in worker process
    evaluator = F0Evaluator(**evaluator_params)

    # Evaluate single pair
    corr, rmse = evaluator.evaluate_single(ref_audio, est_audio, use_dtw=use_dtw)

    return idx, corr, rmse

class F0Evaluator:
    """F0 (fundamental frequency) evaluation class for AudioSignal objects"""

    def __init__(
        self,
        ref_hz=440.0,
        fmin=50,
        fmax=1100,
        frame_length=2048,
        hop_length=512,
    ):
        """
        Initialize F0Evaluator

        Args:
            ref_hz (float): Reference frequency in Hz for cents calculation
            fmin (float): Minimum frequency for F0 extraction
            fmax (float): Maximum frequency for F0 extraction
            frame_length (int): Frame length for F0 extraction
            hop_length (int): Hop length for F0 extraction
        """
        self.ref_hz = ref_hz
        self.fmin = fmin
        self.fmax = fmax
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.sample_rate = 44100

    def hz_to_cents(self, f_hz):
        """Convert Hz to cents relative to reference frequency"""
        f = np.asarray(f_hz, dtype=float)
        cents = np.full_like(f, np.nan, dtype=float)
        m = np.isfinite(f) & (f > 0)
        cents[m] = 1200.0 * np.log2(f[m] / self.ref_hz)
        return cents

    def extract_f0_pyin(self, wav):
        """
        Extract F0 using librosa.pyin from AudioSignal

        Args:
            wav: AudioSignal object, torch.Tensor, or numpy.ndarray

        Returns:
            1D F0 array in Hz, NaN for unvoiced frames
        """
        # Convert AudioSignal to numpy array
        if isinstance(wav, AudioSignal):
            wav = wav.audio_data.squeeze().cpu().numpy()
        elif isinstance(wav, torch.Tensor):
            wav = wav.squeeze().cpu().numpy()
        elif not isinstance(wav, np.ndarray):
            raise TypeError(f"Input wav must be AudioSignal, torch.Tensor or numpy.ndarray, got {type(wav)}")

        f0, vflag, _ = librosa.pyin(
            wav,
            fmin=self.fmin, fmax=self.fmax, sr=self.sample_rate,
            frame_length=self.frame_length, hop_length=self.hop_length
        )
        return f0.astype(np.float32)

    def f0corr_rmse_from_hz(self, f_ref_hz, f_est_hz, use_dtw=False):
        """
        Calculate correlation and RMSE between reference and estimated F0

        Args:
            f_ref_hz: Reference F0 array in Hz
            f_est_hz: Estimated F0 array in Hz
            use_dtw (bool): Whether to use DTW alignment

        Returns:
            tuple: (correlation, rmse_cents)
        """
        x = self.hz_to_cents(f_ref_hz)
        y = self.hz_to_cents(f_est_hz)

        if use_dtw:
            # DTW alignment on cents; interpolate NaN to avoid path collapse
            def _interp_nan(a):
                a = a.copy()
                idx = np.isfinite(a)
                if idx.sum() < 2:  # Almost all NaN
                    return a
                xp = np.flatnonzero(idx)
                fp = a[idx]
                xn = np.arange(len(a))
                a[~idx] = np.interp(xn[~idx], xp, fp)
                return a

            xi = _interp_nan(x)
            yi = _interp_nan(y)
            _, wp = librosa.sequence.dtw(xi, yi, metric=lambda a, b: np.abs(a - b))
            path = list(reversed(wp))
            xr = x[[p[0] for p in path]]
            yr = y[[p[1] for p in path]]
            x, y = xr, yr

        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            return np.nan, np.nan

        xv, yv = x[m], y[m]
        if np.std(xv) < 1e-8 or np.std(yv) < 1e-8:
            r = np.nan
        else:
            r = float(np.corrcoef(xv, yv)[0, 1])
        rmse_cents = float(np.sqrt(np.mean((xv - yv) ** 2)))
        return r, rmse_cents

    def evaluate_single(self, ref_audio, est_audio, use_dtw=False):
        """
        Evaluate F0 correlation and RMSE for a single pair of audio signals

        Args:
            ref_audio: Reference AudioSignal object
            est_audio: Estimated AudioSignal object
            use_dtw (bool): Whether to use DTW alignment

        Returns:
            tuple: (correlation, rmse_cents)
        """
        f0_ref = self.extract_f0_pyin(ref_audio)
        f0_est = self.extract_f0_pyin(est_audio)

        # Length alignment (usually consistent under same hop, but trim to shortest for safety)
        L = min(len(f0_ref), len(f0_est))
        return self.f0corr_rmse_from_hz(f0_ref[:L], f0_est[:L], use_dtw=use_dtw)

    def evaluate_batch(self, ref_audios, est_audios, use_dtw=False):
        """
        Evaluate F0 correlation and RMSE for batches of audio signals (sequential)

        Args:
            ref_audios: List of reference AudioSignal objects
            est_audios: List of estimated AudioSignal objects
            use_dtw (bool): Whether to use DTW alignment

        Returns:
            tuple: (per_sample_results, summary)
                - per_sample_results: List of dicts with 'idx', 'F0CORR', 'F0RMSE_cents'
                - summary: Dict with mean values and unit
        """
        assert len(ref_audios) == len(est_audios), "Reference and estimated audio lists must have same length"

        per_sample = []
        corr_list, rmse_list = [], []

        for i in range(len(ref_audios)):
            r, e = self.evaluate_single(ref_audios[i], est_audios[i], use_dtw=use_dtw)

            per_sample.append({'idx': i, 'F0CORR': r, 'F0RMSE_cents': e})
            if not (np.isnan(r) or np.isnan(e)):
                corr_list.append(r)
                rmse_list.append(e)

        summary = {
            'F0CORR_mean': float(np.mean(corr_list)) if corr_list else np.nan,
            'F0RMSE_cents_mean': float(np.mean(rmse_list)) if corr_list else np.nan,
            'unit': 'cents'
        }
        return per_sample, summary

    def evaluate_batch_mp(self, ref_audios, est_audios, use_dtw=False,
                         num_workers=None, min_batch_size=4):
        """
        Evaluate F0 correlation and RMSE for batches of audio signals using multiprocessing

        Args:
            ref_audios: List of reference AudioSignal objects
            est_audios: List of estimated AudioSignal objects
            use_dtw (bool): Whether to use DTW alignment
            num_workers (int): Number of worker processes (default: min(cpu_count, batch_size))
            min_batch_size (int): Minimum batch size to use multiprocessing

        Returns:
            tuple: (per_sample_results, summary)
                - per_sample_results: List of dicts with 'idx', 'F0CORR', 'F0RMSE_cents'
                - summary: Dict with mean values and unit
        """
        assert len(ref_audios) == len(est_audios), "Reference and estimated audio lists must have same length"

        batch_size = len(ref_audios)

        # Use sequential processing for small batches
        if batch_size < min_batch_size:
            return self.evaluate_batch(ref_audios, est_audios, use_dtw)

        # Determine number of workers
        if num_workers is None:
            num_workers = min(mp.cpu_count(), batch_size)
        num_workers = min(num_workers, batch_size)

        # Prepare arguments for multiprocessing
        evaluator_params = {
            'ref_hz': self.ref_hz,
            'fmin': self.fmin,
            'fmax': self.fmax,
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
        }

        args_list = []
        for i in range(batch_size):
            args_tuple = (ref_audios[i], est_audios[i], evaluator_params, use_dtw, i)
            args_list.append(args_tuple)

        # Process with multiprocessing
        per_sample = []
        corr_list, rmse_list = [], []

        try:
            with mp.Pool(processes=num_workers) as pool:
                results = pool.map(_f0_evaluation_worker, args_list)

                # Sort results by index and collect
                results.sort(key=lambda x: x[0])  # Sort by idx
                for idx, corr, rmse in results:
                    per_sample.append({'idx': idx, 'F0CORR': corr, 'F0RMSE_cents': rmse})
                    if not (np.isnan(corr) or np.isnan(rmse)):
                        corr_list.append(corr)
                        rmse_list.append(rmse)

        except Exception as e:
            print(f"Multiprocessing failed, falling back to sequential: {e}")
            return self.evaluate_batch(ref_audios, est_audios, use_dtw)

        summary = {
            'F0CORR_mean': float(np.mean(corr_list)) if corr_list else np.nan,
            'F0RMSE_cents_mean': float(np.mean(rmse_list)) if corr_list else np.nan,
            'unit': 'cents'
        }
        return per_sample, summary


# ---------- 向後相容性函數 ----------
def hz_to_cents(f_hz, ref_hz=440.0):
    """Legacy function for backward compatibility"""
    evaluator = F0Evaluator(ref_hz=ref_hz)
    return evaluator.hz_to_cents(f_hz)

def f0corr_rmse_from_hz(f_ref_hz, f_est_hz, use_dtw=False):
    """Legacy function for backward compatibility"""
    evaluator = F0Evaluator()
    return evaluator.f0corr_rmse_from_hz(f_ref_hz, f_est_hz, use_dtw)

def extract_f0_pyin(wav, sr=44100, fmin=50, fmax=1100, frame_length=1024, hop_length=256):
    """Legacy function for backward compatibility"""
    evaluator = F0Evaluator(fmin=fmin, fmax=fmax, frame_length=frame_length, hop_length=hop_length)
    # Create a mock AudioSignal-like object for compatibility
    class MockAudioSignal:
        def __init__(self, audio_data, sample_rate):
            self.audio_data = audio_data
            self.sample_rate = sample_rate

    mock_signal = MockAudioSignal(wav, sr)
    return evaluator.extract_f0_pyin(mock_signal)

def f0corr_rmse_librosa_batch(
    ref_wavs, est_wavs, sr=44100,
    fmin=50, fmax=1100, frame_length=1024, hop_length=256,
    use_dtw=False
):
    """Legacy function for backward compatibility"""
    evaluator = F0Evaluator(fmin=fmin, fmax=fmax, frame_length=frame_length, hop_length=hop_length)

    # Create mock AudioSignal objects for compatibility
    class MockAudioSignal:
        def __init__(self, audio_data, sample_rate):
            self.audio_data = audio_data
            self.sample_rate = sample_rate

    ref_audios = [MockAudioSignal(wav, sr) for wav in ref_wavs]
    est_audios = [MockAudioSignal(wav, sr) for wav in est_wavs]

    return evaluator.evaluate_batch(ref_audios, est_audios, use_dtw)



if __name__ == "__main__":
    # Initialize the loss function
    device = torch.device("cuda:0")
    f0_loss = F0EvalLoss(
        hop_length=160,
        fmin=50.0,
        fmax=1100.0,
        model_size='full',
        voicing_thresh=0.5,
        weight=1.0
    ).to(device)
