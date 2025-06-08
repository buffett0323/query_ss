import random
import numpy as np

from audiotools import AudioSignal, transforms as tfm
from audiotools.transforms.base import BaseTransform


class SeqPerturb_Reverse(BaseTransform):
    """
    A perturbation transform for audio sequences, including segmentations and shuffling.
    The input is expected to be an audio waveform (samples).
    """

    def __init__(
        self,
        method: str = 'fixed',
        num_segments: int = 5,
        fixed_second: float = 0.3,
        name: str = None,
        prob: float = 0.5,
    ):
        """
        :param method: Segmentation method ('fixed' or 'random')
        :param num_segments: Number of segments to divide the waveform into.
        :param fixed_second: Duration in seconds for the front segment (used in 'fixed' method)
        :param name: Name of the transform
        :param prob: The probability of applying this transform.
        """
        super().__init__(name=name, prob=prob)
        self.method = method
        self.num_segments = num_segments
        self.fixed_second = fixed_second
        print(f"Sequence Perturbation + Reverse: {self.method}, {self.num_segments} segments, {self.fixed_second} second")

    def _instantiate(self, state: RandomState):
        return {
            "method": self.method,
            "num_segments": self.num_segments,
            "fixed_second": self.fixed_second
        }

    def _transform(self, signal, method, num_segments, fixed_second):
        """
        Perform the segmentation and shuffling on the input signal.
        The waveform will be split into segments, some reversed, shuffled, and concatenated back.
        """
        if method == 'fixed':
            return self._fixed_segmentation(signal, num_segments, fixed_second)
        elif method == 'random':
            return self._random_segmentation(signal, num_segments)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _fixed_segmentation(self, signal, num_segments, fixed_second):
        # Get the sample rate and samples
        sample_rate = signal.sample_rate
        samples = signal.audio_data

        # Fixed second is 0.3
        x_front = samples[:, :int(sample_rate * fixed_second)]
        x_back = samples[:, int(sample_rate * fixed_second):]

        # Split the input into segments
        segments = list(np.split(x_back, indices_or_sections=num_segments, axis=-1))

        # Randomly pick some segments to reverse
        reverse_amount = random.randint(1, num_segments-1)
        reverse_ids = random.sample(range(num_segments), reverse_amount)

        # Reverse the segments
        for reverse_id in reverse_ids:
            segments[reverse_id] = np.flip(segments[reverse_id], axis=-1)

        # Shuffle the segments
        random.shuffle(segments)

        # Concatenate the shuffled segments back into one waveform
        processed_audio = np.concatenate([x_front, np.concatenate(segments, axis=-1)], axis=-1)

        # Return a new signal with the processed audio
        return signal.clone()._audio_data_setitem(processed_audio)

    def _random_segmentation(self, signal, num_segments):
        # TODO: Implement other segmentation methods
        return signal
