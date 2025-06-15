import random
import numpy as np
from numpy.typing import NDArray
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import get_crossfade_mask_pair
from audiomentations import Reverse

# 0.00025 seconds corresponds to 2 samples at 8000 Hz
DURATION_EPSILON = 0.00025

class SeqPerturb_Reverse(BaseWaveformTransform):
    """
    A perturbation transform for audio sequences, including segmentations and shuffling.
    The input is expected to be an audio waveform (samples).
    """

    def __init__(
        self,
        method: str = 'fixed',
        num_segments: int = 5,# 10,
        p: float = 0.5,
        fixed_second: float = 0.3,
    ):
        """
        :param num_segments: Number of segments to divide the waveform into.
        :param p: The probability of applying this transform.
        """
        super().__init__(p)
        self.method = method
        self.num_segments = num_segments
        self.fixed_second = fixed_second
        print(f"Sequence Perturbation + Reverse: {self.method}, {self.num_segments} segments, {self.fixed_second} second")


    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            # No additional randomization needed for the segmentation
            pass


    def apply(self, x: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        """
        Perform the fixed segmentation and shuffling on the input waveform (samples).
        The waveform will be split into `num_segments` segments, shuffled, and concatenated back.

        :param samples: Audio waveform as NDArray of shape (C, T), where C is channels and T is time (samples).
        :param sample_rate: The sample rate of the audio.
        :return: The transformed audio waveform after segmentation and shuffling.
        """

        if self.method == 'fixed':
            return self.fixed_segmentation(x, sample_rate)
        elif self.method == 'random':
            return self.random_segmentation(x, sample_rate)
        else:
            raise ValueError(f"Unknown method: {self.method}")


    def fixed_segmentation(self, samples: NDArray[np.float32], sample_rate: int):
        # Fixed second is 0.3
        x_front = samples[:int(sample_rate * self.fixed_second)]
        x_back = samples[int(sample_rate * self.fixed_second):]

        # T = x_back.shape[-1] # T=15200-4800=10400
        # segment_length = T // self.num_segments # 10400 // 5 = 2080

        # Split the input into segments
        segments = list(np.split(x_back, indices_or_sections=self.num_segments, axis=-1))


        # Randomly pick some segments to reverse
        reverse_amount = random.randint(1, self.num_segments-1)
        reverse_ids = random.sample(range(self.num_segments), reverse_amount)

        # Reverse the segments
        rev_func = Reverse(p=1)
        for reverse_id in reverse_ids:
            segments[reverse_id] = rev_func(segments[reverse_id], sample_rate)

        # Shuffle the segments
        random.shuffle(segments)

        # Concatenate the shuffled segments back into one waveform
        return np.concatenate([x_front, np.concatenate(segments, axis=-1)], axis=-1)


    # TODO: Implement other segmentation methods
    def random_segmentation(self, x, sample_rate):
        return x
