import torch.nn.functional as F
import torch
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import nnAudio.features
from audiomentations import Compose, SeqPerturb_Reverse, TimeMaskBack


class SequencePerturbation(torch.nn.Module):
    def __init__(self, method='random', sample_rate=16000, **kwargs):
        super().__init__()
        self.method = method
        self.kwargs = kwargs
        self.sample_rate = sample_rate
        self.input_length = int(0.95 * sample_rate)  # 0.95 seconds in samples

    def forward(self, x):
        if self.method == 'random':
            return self.random_segmentation(x)
        elif self.method == 'fixed':
            return self.fixed_segmentation(x)
        elif self.method == 'adaptive':
            return self.adaptive_segmentation(x)
        elif self.method == 'reverse':
            return self.reverse_segmentation(x)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def random_segmentation(self, x):
        C, F, T = x.shape
        min_length = max(1, int(T * 0.1))
        max_length = int(T * 0.25)

        segments = []
        start = 0
        while start < T:
            length = random.randint(min_length, min(max_length, T - start))
            segments.append(x[:, :, start:start+length])
            start += length

        random.shuffle(segments)
        return torch.cat(segments, dim=2)


    def fixed_segmentation(self, x, num_segments=10):
        C, F, T = x.shape
        segment_length = T // num_segments

        # Split the input into segments
        segments = list(torch.split(x, segment_length, dim=2))

        # If the number of segments is more than requested (due to uneven division),
        # combine the last segments
        if len(segments) > num_segments:
            segments[num_segments-1] = torch.cat(segments[num_segments-1:], dim=2)
            segments = segments[:num_segments]

        # If we have fewer segments than requested (shouldn't happen normally),
        # pad the last segment to match the expected number
        while len(segments) < num_segments:
            pad_length = segment_length - segments[-1].size(2)
            segments.append(torch.zeros_like(segments[0][:, :, :pad_length]))

        # Shuffle the segments
        random.shuffle(segments)

        # Concatenate the shuffled segments
        return torch.cat(segments, dim=2)


    def adaptive_segmentation(self, x):
        C, F, T = x.shape
        min_segment_length = max(1, int(T * 0.1))

        # Compute frame-wise energy
        energy = torch.mean(x ** 2, dim=1).squeeze(0)

        # Find local maxima in energy
        peaks = torch.nonzero((energy[1:-1] > energy[:-2]) & (energy[1:-1] > energy[2:])).squeeze() + 1

        boundaries = [0]
        for peak in peaks:
            if peak - boundaries[-1] >= min_segment_length and len(boundaries) < 5:  # Maximum 5 segments
                boundaries.append(peak.item())
        boundaries.append(T)

        segments = [x[:, :, boundaries[i]:boundaries[i+1]] for i in range(len(boundaries) - 1)]
        random.shuffle(segments)
        return torch.cat(segments, dim=2)


    def reverse_segmentation(self, x):
        C, F, T = x.shape
        num_segments = 3
        segment_length = T // num_segments

        # Split the input into segments
        segments = list(torch.split(x, segment_length, dim=2))

        # If the number of segments is more than requested (due to uneven division),
        # combine the last segments
        if len(segments) > num_segments:
            segments[num_segments-1] = torch.cat(segments[num_segments-1:], dim=2)
            segments = segments[:num_segments]

        # If we have fewer segments than requested (shouldn't happen normally),
        # pad the last segment to match the expected number
        while len(segments) < num_segments:
            pad_length = segment_length - segments[-1].size(2)
            segments.append(torch.zeros_like(segments[0][:, :, :pad_length]))

        # Reverse each segment individually
        reversed_segments = [seg.flip(dims=[2]) for seg in segments]

        # Shuffle the reversed segments
        random.shuffle(reversed_segments)

        # Concatenate the shuffled reversed segments
        return torch.cat(reversed_segments, dim=2)


    def __repr__(self):
        return f"{self.__class__.__name__}(method={self.method}, kwargs={self.kwargs})"


# Pre-norm
class PrecomputedNorm(nn.Module):
    """Normalization using Pre-computed Mean/Std.

    Args:
        stats: Precomputed (mean, std).
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, stats):
        super().__init__()
        self.mean, self.std = stats

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return ((X - self.mean) / self.std)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
        return format_string


# Post-norm
class NormalizeBatch(nn.Module):
    """Normalization of Input Batch.

    Note:
        Unlike other blocks, use this with *batch inputs*.

    Args:
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, axis=[0, 2, 3]):
        super().__init__()
        self.axis = axis

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _mean = X.mean(dim=self.axis, keepdims=True)
        _std = torch.clamp(X.std(dim=self.axis, keepdims=True), torch.finfo().eps, torch.finfo().max)
        return ((X - _mean) / _std)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(axis={self.axis})'
        return format_string




if __name__ == "__main__":

    sample_rate = 16000
    seg_second = 0.3
    piece_second = 0.95

    transform1 = SeqPerturb_Reverse(method='fixed', num_segments=5, fixed_second=seg_second)
    transform2 = TimeMaskBack(min_band_part=0.0, max_band_part=0.5, fade=True, p=0.5, min_mask_start_time=0.3)
    augment = Compose([transform1, transform2])


    for i in range(10):
        x = np.random.randn(1, int(sample_rate * piece_second)).astype(np.float32)

        print(x.shape)
        x = augment(x, 16000)
        print(x.shape)
