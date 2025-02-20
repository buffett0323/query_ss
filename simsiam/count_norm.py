import os
import torch
import torchaudio
import argparse
import torchvision.transforms as transforms
import torchaudio.transforms as T
import numpy as np

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from utils import yaml_config_hook
from tqdm import tqdm


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





def compute_mean_std(dataset):
    loader = DataLoader(
        dataset, batch_size=128, shuffle=False, 
        num_workers=24, prefetch_factor=4)

    sum_ = 0.0
    sum_sq_ = 0.0
    count = 0

    for x_i, x_j in tqdm(loader):

        # Compute sum and sum of squares
        sum_ += torch.sum(x_i)
        sum_sq_ += torch.sum(x_i ** 2)

        # Count total elements
        count += x_i.numel()
        
        # Compute sum and sum of squares
        sum_ += torch.sum(x_j)
        sum_sq_ += torch.sum(x_j ** 2)

        # Count total elements
        count += x_j.numel()

        
    mean = sum_ / count
    std = torch.sqrt(sum_sq_ / count - mean ** 2)  # Variance formula: E[x^2] - (E[x])^2

    return mean.item(), std.item()



# Example usage with your dataset
parser = argparse.ArgumentParser(description="Simsiam_BP")

config = yaml_config_hook("config/ssbp_swint.yaml")
for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))

args = parser.parse_args()

ds = BPDataset(
    sample_rate=args.sample_rate, 
    segment_second=args.segment_second, 
    piece_second=args.piece_second,
    data_dir=args.data_dir,
    augment_func=None,
    n_mels=args.n_mels,
    n_fft=args.n_fft,
    hop_length=args.hop_length,
    split="train",
    melspec_transform=False, #args.melspec_transform,
    data_augmentation=False, #args.data_augmentation,
    random_slice=False, #args.random_slice,
    stems=['other'],
)



mean, std = compute_mean_std(ds)
print(f"Mean: {mean}, Std: {std}")
