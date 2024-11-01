import torch
import os
import shutil
import librosa
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

from mir_eval.separation import bss_eval_sources
from torchmetrics.functional.audio.snr import signal_noise_ratio

def cal_metrics(gt_stft, S_k):
    
    # Init meters
    # sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    gt_stft = gt_stft.cpu().detach().numpy()
    S_k = S_k.cpu().detach().numpy()
    sdr_list = []
    sir_list = []
    sar_list = []
    
    for i in range(gt_stft.shape[0]):
        gt_stft[i] = gt_stft[i] + 1e-10
        S_k[i] = S_k[i] + 1e-10
        sdr, sir, sar, _ = bss_eval_sources(gt_stft[i], S_k[i])
        sdr_list.append(sdr)
        sir_list.append(sir)
        sar_list.append(sar)
        
        # sdr_mix_meter.update(sdr_mix.mean())
        sdr_meter.update(sdr.mean())
        sir_meter.update(sir.mean())
        sar_meter.update(sar.mean())
    
    return [#sdr_mix_meter.average(),
            sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average()]


def safe_signal_noise_ratio(
    preds: torch.Tensor, target: torch.Tensor, zero_mean: bool = False
) -> torch.Tensor:

    return torch.nan_to_num(
        signal_noise_ratio(preds, target, zero_mean=zero_mean), nan=torch.nan, posinf=100.0, neginf=-100.0
    )
    
    
class MetricHandler(nn.Module):
    def __init__(self, stems: List[str]):
        super(MetricHandler, self).__init__()
        self.stems = stems
        self.metrics = {stem: [] for stem in self.stems}
    
    def calculate_snr(self, preds: torch.Tensor, target: torch.Tensor, stem_names: List[str], zero_mean: bool = False):
        for i, stem in enumerate(stem_names):
            snr_value = safe_signal_noise_ratio(preds[i], target[i], zero_mean=zero_mean)
            self.metrics[stem].append(snr_value)
    
    def get_mean_median(self, get_mean: bool = False) -> Dict[str, Dict[str, torch.Tensor]]:
        mean_median_results = {}
        for stem, values in self.metrics.items():
            if values:
                values_tensor = torch.stack(values)
                mean_snr = torch.mean(values_tensor)
                median_snr = torch.median(values_tensor)
                if get_mean:
                    mean_median_results[stem] = {
                        'mean': float(mean_snr),
                        'median': float(median_snr)
                    }
                else:
                    mean_median_results[stem] = {
                        'median': float(median_snr)
                    }
        if get_mean:
            mean_median_results = {f"{stem}/median": result['median'] for stem, result in mean_median_results.items()}
        else:
            mean_median_results = {f"{stem}/median": result['median'] for stem, result in mean_median_results.items()}
        return mean_median_results

    def reset(self):
        self.metrics = {stem: [] for stem in self.stems}
    
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val*weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        val = np.asarray(val)
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        if self.val is None:
            return 0.
        else:
            return self.val.tolist()

    def average(self):
        if self.avg is None:
            return 0.
        else:
            return self.avg.tolist()
        
        

if __name__ == "__main__":
    # Example usage:
    stems = ['vocals', 'drums', 'bass', 'other']

    # Assuming you have 32 (pred, target) pairs and a list of corresponding stem names
    stem_names = ['vocals', 'drums', 'drums', 'vocals', 'bass', 'other', 'vocals', 'drums', 'bass', 'bass', 'vocals', 'drums', 
                'vocals', 'drums', 'drums', 'bass', 'vocals', 'other', 'bass', 'drums', 'vocals', 'bass', 'drums', 'vocals',
                'drums', 'vocals', 'other', 'bass', 'vocals', 'drums', 'vocals', 'bass']

    # Example tensors with shape (32, 60000)
    preds = torch.randn(32, 60000)
    target = torch.randn(32, 60000)

    # Initialize MetricHandler
    metric_handler = MetricHandler(stems)

    # Calculate SNR for each (pred, target) pair using the stem names
    metric_handler.calculate_snr(preds, target, stem_names)

    # Get mean and median SNR values for each stem
    results = metric_handler.get_mean_median()
    print(results)

    # Reset metrics for future use
    metric_handler.reset()