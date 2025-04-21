import os
import torch
import torchaudio
import argparse
import torchvision.transforms as transforms
import torchaudio.transforms as T
import numpy as np
import nnAudio.features

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from utils import yaml_config_hook
from tqdm import tqdm
from nsynth_dataset import NsynthDataset

class BPDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        segment_second,
        data_dir,
        augment_func,
        piece_second=3,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        split="train",
        melspec_transform=False,
        data_augmentation=True,
        random_slice=False,
        stems=["other"], #["vocals", "bass", "drums", "other"], # VBDO
    ):
        with open("info/sum_8secs.txt", "r") as f:
            bp_listdir = [line.strip() for line in f.readlines()]
            
        self.stems = stems
        self.data_path_list = [
            os.path.join(data_dir, folder, f"{stem}.npy")
            for folder in bp_listdir
                for stem in stems
        ]        
        
        self.sample_rate = sample_rate
        self.segment_second = segment_second
        self.duration = sample_rate * piece_second # 3 seconds for each piece
        self.augment_func = augment_func # CLARTransform
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.split = split
        self.melspec_transform = melspec_transform
        self.data_augmentation = data_augmentation
        self.random_slice = random_slice
        
        # Mel-spec transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            f_max=args.fmax,
        )
        self.db_transform = T.AmplitudeToDB(
            stype="power"
        )
        self.resizer = transforms.Resize((args.img_size, args.img_size))

    
    
    def mel_spec_transform(self, x):
        return self.db_transform(self.mel_transform(x))
    
    
    def data_pipeline(self, x):
        # Mel spectrogram transform
        x = self.mel_spec_transform(x).unsqueeze(0) # 1, 128, 251 --> 1, 128, 125
        
        # Resize
        x = self.resizer(x) # transform to 1, 224, 224
        
        # Normalize
        return x

    
        
    def __len__(self): #""" Total we got 175698 files * 4 tracks """
        return len(self.data_path_list)
    
    
    def __getitem__(self, idx):
        """ 
            1. Mel-Spectrogram Transformation
            2. Resize
            3. Normalization
            4. Data Augmentation
        """
        
        # Load audio data
        path = self.data_path_list[idx]
        x = torch.tensor(np.load(path))
        
        # Slice into half # TODO: Random Crop for 4 seconds
        x_i, x_j = self.data_pipeline(x[:self.duration]), self.data_pipeline(x[self.duration:])
        return x_i.float(), x_j.float()





def compute_mean_std(dataset, device):
    loader = DataLoader(
        dataset, batch_size=128, shuffle=False, 
        num_workers=24, prefetch_factor=4)

    sum_ = 0.0
    sum_sq_ = 0.0
    count = 0
    
    to_spec = nnAudio.features.MelSpectrogram(
        sr=args.sample_rate,
        n_fft=args.n_fft,
        win_length=args.window_size,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        center=True,
        power=2,
        verbose=False,
    ).to(device)

    for x, _ in tqdm(loader):
        x = x.to(device)
        logmel = (to_spec(x) + torch.finfo().eps).log()  # [B, n_mels, T]

        sum_ += logmel.sum().item()
        sum_sq_ += (logmel ** 2).sum().item()
        count += logmel.numel()

    mean = sum_ / count
    std = (sum_sq_ / count - mean ** 2) ** 0.5

    return mean, std



if __name__ == "__main__":
    # Example usage with your dataset
    parser = argparse.ArgumentParser(description="Simsiam_BP")

    config = yaml_config_hook("config/nsynth_convnext.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()


    train_dataset = NsynthDataset(
        sample_rate=args.sample_rate,
        data_dir=args.data_dir,
        piece_second=args.piece_second,
        segment_second=args.segment_second,
        window_size=args.window_size,
        hop_length=args.hop_length,
    )

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    mean, std = compute_mean_std(train_dataset, device)
    
    print(f"Mean: {mean}, Std: {std}")
