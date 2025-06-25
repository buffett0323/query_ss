"""BYOL for Audio: Augmentation blocks.

Legends:
    F: Number of frequency bins.
    T: Number of time frames.
"""

import torch
import torch.nn as nn
import psutil

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


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024**3)  # Convert to GB

def format_memory_size(size_bytes):
    """Format memory size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def print_memory_usage(label="Current"):
    """Print current memory usage with a label."""
    memory_gb = get_memory_usage()
    print(f"{label} memory usage: {memory_gb:.2f} GB")
