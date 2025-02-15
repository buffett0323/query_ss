# Using original track, without .dbfss
import os
import random
import math
import torch
import torchaudio
import json
import librosa
import argparse
import scipy.interpolate
import scipy.stats
import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp
import torchaudio.transforms as T
import torchvision.transforms as transforms

from librosa import effects
from tqdm import tqdm
from torchaudio.functional import pitch_shift
from utils import yaml_config_hook


class CLARTransform(nn.Module):
    def __init__(
        self, 
        sample_rate,
        duration,
    ):
        super(CLARTransform, self).__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.transforms = [
            self.pitch_shift_transform,
            self.add_fade_transform,
            self.add_noise_transform,
            self.time_masking_transform,
            self.time_shift_transform,
            self.time_stretch_transform,
        ]
        
    def pitch_shift_transform(self, x, n_steps=15):
        return effects.pitch_shift(x, sr=self.sample_rate, n_steps=torch.randint(low=-n_steps, high=n_steps, size=[1]).item())
    
    
    def time_stretch_augmentation(self, x):
        """
        Apply Time Stretching (TS) Augmentation following the TSPS methodology.
        
        Steps:
        1. Load an audio file and ensure it is at least `context_length` seconds long.
        2. Randomly crop a 4.5s segment (if necessary).
        3. Sample a time-stretch factor τ from the given probability distribution.
        4. Apply time stretching using cubic spline interpolation.
        5. Truncate the resulting signal to `target_length` seconds.
        """
        target_length = 3.0
        
        # Sample τ from the given probability distribution and apply Time Stretching (TS)
        x = librosa.effects.time_stretch(x, rate=sample_tau())

        # Truncate to `target_length` seconds
        target_samples = int(self.sample_rate * target_length)
        if len(x) > target_samples:
            x = x[:target_samples]
        else:
            x = np.pad(x, (0, target_samples - len(x)), mode='constant')

        return x #torch.tensor(x).float().unsqueeze(0)  # Add batch dimension
    
        
    def add_fade_transform(self, x, max_fade_size=.5):
        def _fade_in(fade_shape, waveform_length, fade_in_len):
            fade = np.linspace(0, 1, fade_in_len)
            ones = np.ones(waveform_length - fade_in_len)
            if fade_shape == 0:
                fade = fade
            if fade_shape == 1:
                fade = np.power(2, (fade - 1)) * fade
            if fade_shape == 2:
                fade = np.log10(.1 + fade) + 1
            if fade_shape == 3:
                fade = np.sin(fade * math.pi / 2)
            if fade_shape == 4:
                fade = np.sin(fade * math.pi - math.pi / 2) / 2 + 0.5
            return np.clip(np.concatenate((fade, ones)), 0, 1)

        def _fade_out(fade_shape, waveform_length, fade_out_len):
            fade = torch.linspace(0, 1, fade_out_len)
            ones = torch.ones(waveform_length - fade_out_len)
            if fade_shape == 0:
                fade = - fade + 1
            if fade_shape == 1:
                fade = np.power(2, - fade) * (1 - fade)
            if fade_shape == 2:
                fade = np.log10(1.1 - fade) + 1
            if fade_shape == 3:
                fade = np.sin(fade * math.pi / 2 + math.pi / 2)
            if fade_shape == 4:
                fade = np.sin(fade * math.pi + math.pi / 2) / 2 + 0.5
            return np.clip(np.concatenate((ones, fade)), 0, 1)

        waveform_length = x.shape[0]
        fade_shape = np.random.randint(5)
        fade_out_len = np.random.randint(int(x.shape[0] * max_fade_size))
        fade_in_len = np.random.randint(int(x.shape[0] * max_fade_size))
        return np.float32(
            _fade_in(fade_shape, waveform_length, fade_in_len) * 
            _fade_out(fade_shape, waveform_length, fade_out_len) * 
            x
        )


    def add_noise_transform(self, x):
        noise_type = random.choice(['white', 'brown', 'pink'])
        snr = random.uniform(0.5, 1.5)  # Signal-to-noise ratio
        
        if noise_type == 'white':
            noise = np.random.normal(0, 1, len(x))
        elif noise_type == 'brown':
            noise = np.cumsum(np.random.normal(0, 1, len(x)))
            noise = noise / np.max(np.abs(noise))
        else:  # pink noise
            freqs = np.fft.rfftfreq(len(x))
            noise = np.fft.irfft(np.random.randn(len(freqs)) / (freqs + 1e-6))
        
        noise = noise / np.max(np.abs(noise))
        x = x + noise / snr
        return np.clip(x, -1, 1)


    def time_masking_transform(self, x, sr=0.125):
        sr = int(x.shape[0] * sr)
        start = np.random.randint(x.shape[0] - sr)
        x[start: start + sr] = np.float32(np.random.normal(0, 0.01, sr))
        return x

    def time_shift_transform(self, x, shift_rate=8000):
        return np.roll(x, torch.randint(low=-shift_rate, high=shift_rate, size=[1]).item())

    def time_stretch_transform(self, x):
        x = effects.time_stretch(x, rate=random.uniform(0.5, 1.5))
        x = librosa.resample(x, orig_sr=x.shape[0] / self.duration, target_sr=self.sample_rate)
        if x.shape[0] > (self.sample_rate * self.duration):
            return x[:(self.sample_rate * self.duration)]
        return np.pad(x, [0, (self.sample_rate * self.duration) - x.shape[0]])


    def __call__(self, x1, x2):
        # Apply augmentations
        x1 = self.add_noise_transform(x1)
        x1 = self.time_stretch_augmentation(x1)
        # x1 = self.time_masking_transform(x1)
        # x1 = self.time_shift_transform(x1)
        
        x2 = self.add_noise_transform(x2)
        x2 = self.time_stretch_augmentation(x2)
        # x2 = self.time_masking_transform(x2)
        # x2 = self.time_shift_transform(x2)
        return x1, x2



def sample_tau():
    """
    Sample τ from the given probability distribution: τ ∼ 1/(τ log(1.5/0.75))
    within the range [0.75, 1.5].
    """
    def pdf(tau):
        return 1 / (tau * np.log(1.5 / 0.75))
    
    tau_range = np.linspace(0.75, 1.5, 1000)
    probs = pdf(tau_range)
    probs /= probs.sum()  # Normalize to create a proper probability distribution
    return np.random.choice(tau_range, p=probs)


def sample_mu():
    """
    Sample μ from the given probability distribution: μ ∼ 1/(μ log(1.335/0.749))
    within the range [0.749, 1.335].
    """
    def pdf(mu):
        return 1 / (mu * np.log(1.335 / 0.749))
    
    mu_range = np.linspace(0.749, 1.335, 1000)
    probs = pdf(mu_range)
    probs /= probs.sum()  # Normalize to create a proper probability distribution
    return np.random.choice(mu_range, p=probs)



class AudioFXAugmentation(nn.Module):
    def __init__(
        self, 
        sample_rate,
        duration,
        n_mels=128,
    ):
        super(AudioFXAugmentation, self).__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
    

    def time_stretch_augmentation(self, x):
        """
        Apply Time Stretching (TS) Augmentation following the TSPS methodology.
        
        Steps:
        1. Load an audio file and ensure it is at least `context_length` seconds long.
        2. Randomly crop a 4.5s segment (if necessary).
        3. Sample a time-stretch factor τ from the given probability distribution.
        4. Apply time stretching using cubic spline interpolation.
        5. Truncate the resulting signal to `target_length` seconds.
        """
        target_length = 3.0
        
        # Sample τ from the given probability distribution and apply Time Stretching (TS)
        x = librosa.effects.time_stretch(x, rate=sample_tau())

        # Truncate to `target_length` seconds
        target_samples = int(self.sample_rate * target_length)
        if len(x) > target_samples:
            x = x[:target_samples]
        else:
            x = np.pad(x, (0, target_samples - len(x)), mode='constant')

        return x #torch.tensor(x).float().unsqueeze(0)  # Add batch dimension


    def pitch_shift_augmentation(self, x):
        """
        Apply Pitch Shifting (PS) Augmentation following the TSPS methodology.

        Steps:
        1. Compute the mel spectrogram of the input audio.
        2. Sample a pitch shift factor μ.
        3. Apply cubic spline interpolation on the frequency axis.
        4. If μ < 1.0, zero out frequency bins above μ * max frequency.
        """

        # Compute Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=x, 
            sr=self.sample_rate, 
            n_mels=self.n_mels,
            fmax=8000,
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)  # Convert to dB

        # Sample μ
        mu = sample_mu()

        # Compute frequency bin warping
        U = self.n_mels
        SU = U / np.log10(1 + self.sample_rate / 700)  # Compute scaling factor
        mel_bins = np.linspace(0, U - 1, U)  # Original mel bins
        warped_bins = SU * np.log10(1 + mu * (10 ** (mel_bins / SU) - 1))  # Apply pitch shift transformation

        # Apply cubic spline interpolation
        interpolator = scipy.interpolate.interp1d(mel_bins, mel_spectrogram_db, axis=0, kind='cubic', fill_value="extrapolate")
        shifted_spectrogram = interpolator(warped_bins)

        # Zero out bins if μ < 1.0
        if mu < 1.0:
            cutoff_bin = int(mu * U)
            shifted_spectrogram[cutoff_bin:] = -80.0  # Set to silence (approximate dB floor)
        
        return shifted_spectrogram
    
    
    def butterworth_filter(self, filter_type="low", cutoff_freq=None):
        """
        Apply a third-order Butterworth filter (lowpass/highpass).

        Parameters:
            sr (int): Sampling rate.
            n_mels (int): Number of mel bins.
            filter_type (str): "low" or "high".
            cutoff_freq (float): Cutoff frequency in Hz.

        Returns:
            np.ndarray: Butterworth filter response.
        """
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff_freq / nyquist  # Normalize w.r.t Nyquist frequency
        b, a = scipy.signal.butter(N=3, Wn=normal_cutoff, btype=filter_type, analog=False)
        return b, a


    def apply_equalization_filter(self, x):
        """
        Apply Equalization Augmentation (EQ) using Butterworth filters.

        Steps:
        1. Randomly select a filter type (lowpass, highpass, or none).
        2. Design a Butterworth filter based on sampled cutoff frequency.
        3. Apply the filter to the spectrogram.
        4. Add the log-scaled filter response to the original spectrogram.

        Parameters:
            mel_spectrogram (np.ndarray): Input mel spectrogram.
            sr (int): Sampling rate.
            n_mels (int): Number of mel bins.

        Returns:
            np.ndarray: Augmented spectrogram.
            str: Applied filter type.
            float: Chosen cutoff frequency.
        """
        # Choose filter type with equal probability
        filter_choice = np.random.choice(["lowpass", "highpass", "none"], p=[1/3, 1/3, 1/3])

        if filter_choice == "none":
            return x

        # Sample cutoff frequency
        if filter_choice == "lowpass":
            cutoff_freq = np.random.uniform(2200, 4000)  # Hz
        else:  # Highpass
            cutoff_freq = np.random.uniform(200, 1200)  # Hz

        # Get Butterworth filter and Apply filter along the mel bins axis
        b, a = self.butterworth_filter(filter_type=filter_choice, cutoff_freq=cutoff_freq)
        filtered_spectrogram = scipy.signal.filtfilt(b, a, x, axis=0)

        # Add the log-transformed filter response
        return x + np.log10(np.abs(filtered_spectrogram) + 1e-6)



    def __call__(self, x1, x2):        
        # Apply augmentations
        x1 = self.time_stretch_augmentation(x1)
        x1 = self.pitch_shift_augmentation(x1)
        # x1 = self.apply_equalization_filter(x1)
        
        x2 = self.time_stretch_augmentation(x2)
        x2 = self.pitch_shift_augmentation(x2)
        # x2 = self.apply_equalization_filter(x2)
        return x1, x2



class S3TAugmentation(nn.Module):
    def __init__(self):
        super(S3TAugmentation, self).__init__()
        
        
    def random_multi_crop(self, spectrogram, min_ratio=0.1, max_ratio=0.9):
        T_full = spectrogram.shape[1]  # Time dimension
        r1, r2 = random.uniform(min_ratio, max_ratio), random.uniform(min_ratio, max_ratio)
        T_r1, T_r2 = int(T_full * r1), int(T_full * r2)
        
        start1 = random.randint(0, T_full - T_r1)
        start2 = random.randint(0, T_full - T_r2)
        
        crop1 = spectrogram[:, start1:start1 + T_r1]
        crop2 = spectrogram[:, start2:start2 + T_r2]
        
        return crop1, crop2

    def random_frequency_masking(self, spectrogram, p=0.5, N_range=(1, 5), F_range=(5, 30)):
        if random.random() < p:
            N = random.randint(*N_range)
            F = random.randint(*F_range)
            transform = T.FrequencyMasking(freq_mask_param=F)
            for _ in range(N):
                spectrogram = transform(spectrogram)
        return spectrogram

    def random_time_masking(self, spectrogram, p=0.5, N_range=(1, 10), r_range=(0.01, 0.2)):
        if random.random() < p:
            T_full = spectrogram.shape[1]
            N = random.randint(*N_range)
            r = random.uniform(*r_range)
            t = int(T_full * r)
            transform = T.TimeMasking(time_mask_param=t)
            for _ in range(N):
                spectrogram = transform(spectrogram)
        return spectrogram

    def time_warping(self, spectrogram, p=0.4, W_range=(0, 10)):
        if random.random() < p:
            T_full = spectrogram.shape[1]
            W = random.randint(*W_range)
            center = T_full // 2
            shift = random.choice([-1, 1]) * W
            warped_spectrogram = torch.roll(spectrogram, shifts=shift, dims=1)
            return warped_spectrogram
        return spectrogram

    def random_shifting(self, spectrogram, p=0.4, shift_range=(1, 10)):
        if random.random() < p:
            t_shift = random.randint(*shift_range)
            f_shift = random.randint(*shift_range)
            shift_t = random.choice([-1, 1]) * t_shift
            shift_f = random.choice([-1, 1]) * f_shift
            shifted_spectrogram = torch.roll(spectrogram, shifts=(shift_f, shift_t), dims=(0, 1))
            return shifted_spectrogram
        return spectrogram

    def frequency_tiling(self, spectrogram, n):
        F, T = spectrogram.shape
        tiled = spectrogram.repeat(n, 1)  # Repeat along frequency axis
        return tiled[:T, :]  # Cut extra high frequency part to T x T

    def time_folding(self, spectrogram, n):
        F, T = spectrogram.shape
        folded = spectrogram.view(F * n, T // n)  # Reshape to (F * n) x (T/n)
        return folded

    def apply_augmentations(self, spectrogram):
        crop1, crop2 = self.random_multi_crop(spectrogram)
        
        crop1 = self.random_frequency_masking(crop1)
        crop1 = self.random_time_masking(crop1)
        crop1 = self.time_warping(crop1)
        crop1 = self.random_shifting(crop1)
        
        crop2 = self.random_frequency_masking(crop2)
        crop2 = self.random_time_masking(crop2)
        crop2 = self.time_warping(crop2)
        crop2 = self.random_shifting(crop2)
        
        return crop1, crop2
    
    
if __name__ == "__main__":
    Aug = S3TAugmentation()
        # Example usage:
    spectrogram = torch.randn(128, 1000)  # Example spectrogram with 128 frequency bins and 1000 time steps
    crop1, crop2 = Aug.apply_augmentations(spectrogram)
    print("Crop 1 shape:", crop1.shape, crop2.shape) 
    # Applying preprocessors
    crop1 = Aug.frequency_tiling(crop1, 2)
    crop1 = Aug.time_folding(crop1, 2)
    print("Crop 1 shape:", crop1.shape, crop2.shape) 
