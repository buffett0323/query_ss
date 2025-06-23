from dataset import ADSRDataModule, AudioADSRDataset
from pathlib import Path
from tqdm import tqdm
import yaml
import nnAudio.features
import torch
import numpy as np
from torch.utils.data import DataLoader

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def calc_norm_stats(data_loader, n_stats=10000, device='cuda:3'):
    # Calculate normalization statistics from the training dataset.
    n_stats = min(n_stats, len(data_loader.dataset))


    to_spec = nnAudio.features.MelSpectrogram(
        sr=44100,
        n_mels=128,
        fmin=20,
        fmax=22050,
        hop_length=512,
        n_fft=2048,
        window='hann',
        center=True,
        power=2.0,
    ).to(device)
    
    
    X = []
    for wav, _ in tqdm(data_loader): 
        mel = to_spec(wav.to(device))
        mel = mel.unsqueeze(1) # [BS, 1, 128, 256]
        eps = torch.finfo(mel.dtype).eps
        mel = (mel + eps).log()
        
        if mel.isnan().any():
            print("mel is nan")
            print(mel)
            break
            
        X.extend([x for x in mel.detach().cpu().numpy()])
        if len(X) >= n_stats:
            break
        
    X = np.stack(X)
    norm_stats = np.array([X.mean(), X.std()])
    print(f"Created PrecomputedNorm with stats: {norm_stats}")
    print(f"Mean: {norm_stats[0]:.6f}, Std: {norm_stats[1]:.6f}")
    return norm_stats


if __name__ == "__main__":
    config = load_config('config.yaml')
    dataset = AudioADSRDataset(
        data_dir=Path(config['data_dir']),
        split="train",
    )
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    norm_stats = calc_norm_stats(data_loader, n_stats=10000)
    print(norm_stats)