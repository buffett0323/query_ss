import random

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import get_crossfade_mask_pair

# Buffett added
class TimeMaskBack(BaseWaveformTransform):
    """
    Make a randomly chosen part of the audio silent starting after 0.3 seconds.
    Inspired by https://arxiv.org/pdf/1904.08779.pdf
    """

    supports_multichannel = True

    def __init__(
        self,
        min_band_part: float = 0.0,
        max_band_part: float = 0.5,
        fade: bool = True,
        p: float = 0.5,
        min_mask_start_time: float = 0.3,  # Mask only after 0.3 seconds
    ):
        """
        :param min_band_part: Minimum length of the silent part as a fraction of the
            total sound length. Must be between 0.0 and 1.0
        :param max_band_part: Maximum length of the silent part as a fraction of the
            total sound length. Must be between 0.0 and 1.0
        :param fade: When set to True, a smooth fade-in and fade-out is added to the silent part.
            This can smooth out unwanted abrupt changes between consecutive samples, which might
            otherwise sound like transients/clicks/pops.
        :param p: The probability of applying this transform
        :param min_mask_start_time: The minimum starting time in seconds after which to start masking
        """
        super().__init__(p)
        if min_band_part < 0.0 or min_band_part > 1.0:
            raise ValueError("min_band_part must be between 0.0 and 1.0")
        if max_band_part < 0.0 or max_band_part > 1.0:
            raise ValueError("max_band_part must be between 0.0 and 1.0")
        if min_band_part > max_band_part:
            raise ValueError("min_band_part must not be greater than max_band_part")
        if min_mask_start_time < 0.0:
            raise ValueError("min_mask_start_time must be non-negative")

        self.min_band_part = min_band_part
        self.max_band_part = max_band_part
        self.fade = fade
        self.min_mask_start_time = min_mask_start_time

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            num_samples = samples.shape[-1]
            start_sample = int(self.min_mask_start_time * sample_rate)
            num_samples_after_start = num_samples - start_sample
            self.parameters["t"] = random.randint(
                int(num_samples_after_start * self.min_band_part),
                int(num_samples_after_start * self.max_band_part),
            )
            self.parameters["t0"] = random.randint(
                start_sample, num_samples - self.parameters["t"]
            )

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        new_samples = samples.copy()
        t = self.parameters["t"]
        t0 = self.parameters["t0"]
        mask = np.zeros(t, dtype=np.float32)

        if self.fade:
            fade_length = min(int(sample_rate * 0.01), int(t * 0.1))
            if fade_length >= 2:
                fade_in, fade_out = get_crossfade_mask_pair(fade_length, equal_energy=False)
                mask[:fade_length] = fade_out
                mask[-fade_length:] = fade_in

        new_samples[..., t0 : t0 + t] *= mask
        if t0 < self.min_mask_start_time * sample_rate:
            raise ValueError("t0 cannot be less than min_mask_start_time * sample_rate")
            print("T0 < min_mask_start_time * sample_rate")
        return new_samples



class TimeMask(BaseWaveformTransform):
    """
    Make a randomly chosen part of the audio silent.
    Inspired by https://arxiv.org/pdf/1904.08779.pdf
    """

    supports_multichannel = True

    def __init__(
        self,
        min_band_part: float = 0.0,
        max_band_part: float = 0.5,
        fade: bool = True,
        p: float = 0.5,
    ):
        """
        :param min_band_part: Minimum length of the silent part as a fraction of the
            total sound length. Must be between 0.0 and 1.0
        :param max_band_part: Maximum length of the silent part as a fraction of the
            total sound length. Must be between 0.0 and 1.0
        :param fade: When set to True, a smooth fade-in and fade-out is added to the silent part.
            This can smooth out unwanted abrupt changes between consecutive samples, which might
            otherwise sound like transients/clicks/pops.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        if min_band_part < 0.0 or min_band_part > 1.0:
            raise ValueError("min_band_part must be between 0.0 and 1.0")
        if max_band_part < 0.0 or max_band_part > 1.0:
            raise ValueError("max_band_part must be between 0.0 and 1.0")
        if min_band_part > max_band_part:
            raise ValueError("min_band_part must not be greater than max_band_part")
        self.min_band_part = min_band_part
        self.max_band_part = max_band_part
        self.fade = fade

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            num_samples = samples.shape[-1]
            self.parameters["t"] = random.randint(
                int(num_samples * self.min_band_part),
                int(num_samples * self.max_band_part),
            )
            self.parameters["t0"] = random.randint(
                0, num_samples - self.parameters["t"]
            )

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        new_samples = samples.copy()
        t = self.parameters["t"]
        t0 = self.parameters["t0"]
        mask = np.zeros(t, dtype=np.float32)
        if self.fade:
            fade_length = min(int(sample_rate * 0.01), int(t * 0.1))
            if fade_length >= 2:
                fade_in, fade_out = get_crossfade_mask_pair(fade_length, equal_energy=False)
                mask[:fade_length] = fade_out
                mask[-fade_length:] = fade_in
        new_samples[..., t0 : t0 + t] *= mask
        return new_samples
