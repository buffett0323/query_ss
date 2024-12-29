import os
import random
import shutil
import torch
import torchaudio
import argparse
import torch.nn as nn
import torch.multiprocessing as mp
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm
from mutagen.mp3 import MP3
from utils import yaml_config_hook


class BeatportDataset(Dataset):
    def __init__(
        self, 
        args,
        dataset_dir='/mnt/gestalt/home/ddmanddman/beatport_preprocess/chorus',
        preprocessed_dir='/mnt/gestalt/home/ddmanddman/beatport_preprocess/pt',
        device='cpu',
        split="train", 
        n_fft=2048, 
        hop_length=1024,
        filter_short=False,
        pre_process=False,
    ):
        """
        Args:
            dataset_dir (str): Path to the dataset directory.
            split (str): The split to load ('train', 'test', 'unlabeled').
            transform (callable, optional): A function/transform to apply to the audio data.
        """
        self.dataset_dir = dataset_dir
        self.preprocessed_dir = preprocessed_dir
        self.my_device = device
        self.split = split
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = n_fft + hop_length
        self.max_length = args.max_length
        
        if filter_short: self.filter_short()
        if pre_process: self.preprocess()


        self.audio_files = [
            os.path.join(dataset_dir, folder_name, file_name)
            for folder_name in os.listdir(dataset_dir)
                for file_name in os.listdir(os.path.join(dataset_dir, folder_name))
                    if file_name.endswith(".mp3")
        ]
        
        self.pt_files = [
            os.path.join(preprocessed_dir, folder_name, file_name)
            for folder_name in os.listdir(preprocessed_dir)
                for file_name in os.listdir(os.path.join(preprocessed_dir, folder_name))
                    if file_name.endswith(".pt")
        ]
        
        self.transform = SimCLRTransform(
            sample_rate=args.sample_rate, 
            n_mels=args.n_mels, 
            n_fft=args.n_fft, 
            hop_length=args.hop_length,
            device=self.my_device,
        )#.to(device)

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        # Load the audio file
        pt_path = self.pt_files[idx]
        waveform = torch.load(pt_path, weights_only=True)

        x_i, x_j = self.transform(waveform)
        x_i, x_j = self.consist_size(x_i), self.consist_size(x_j)
        return x_i, x_j
    
    
    def consist_size(self, mel_spectrogram):
        if mel_spectrogram.size(-1) < self.max_length:
            pad_size = self.max_length - mel_spectrogram.size(-1)
            return torch.nn.functional.pad(mel_spectrogram, (0, pad_size))
        else:
            # Randomly select a starting point for the partition
            start_idx = torch.randint(0, mel_spectrogram.size(-1) - self.max_length + 1, (1,)).item()
            return mel_spectrogram[:, :, start_idx:start_idx + self.max_length]
        
    
    def filter_short(self, min_duration=1):
        self.folders = [
            os.path.join(dataset_dir, folder_name)
            for folder_name in os.listdir(dataset_dir)
        ]
        print("Before Filter, List Length:", len(self.folders))
        for a in tqdm(self.folders):
            num = str(a).split('_')[-1]
            a_mp3 = os.path.join(a, f"bass_chorus_{str(num)}.mp3")                
            audio = MP3(a_mp3)
            if audio.info.length < min_duration:
                shutil.rmtree(a)
                
        self.folders = [
            os.path.join(dataset_dir, folder_name)
            for folder_name in os.listdir(dataset_dir)
        ]
        print("After Filter, List Length:", len(self.folders))
    
    
    def preprocess(self):
        """Preprocess audio files using multiprocessing."""
        for file_name in tqdm(self.audio_files):
            if file_name.endswith(".mp3"):
                song_name = file_name.split('/')[-2]
                os.makedirs(os.path.join(self.preprocessed_dir, song_name), exist_ok=True)
                save_path = os.path.join(
                    self.preprocessed_dir, 
                    song_name, 
                    f"{file_name.split('/')[-1].split('.mp3')[0]}.pt"
                )
                if not os.path.exists(save_path):
                    waveform, _ = torchaudio.load(file_name)                
                    torch.save(waveform, save_path)
            

        

# TODO: transform random
class SimCLRTransform:
    def __init__(
        self, 
        sample_rate=44100, 
        n_mels=128, 
        n_fft=2048, 
        hop_length=1024,
        device="cpu", 
    ):
        self.my_device = torch.device(device)
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        ).to(self.my_device)
        self.amplitude_to_db = T.AmplitudeToDB().to(self.my_device)
        self.transforms = [
            lambda spectrogram: self.time_mask(spectrogram, mask_param=30),
            lambda spectrogram: self.frequency_mask(spectrogram, mask_param=15),
            lambda spectrogram: self.random_crop(spectrogram, crop_size=128),
            lambda spectrogram: self.add_noise(spectrogram, noise_level=0.005),
        ]

    def time_mask(self, spectrogram, mask_param=30):
        time_mask = T.TimeMasking(time_mask_param=mask_param).to(self.my_device)
        return time_mask(spectrogram)


    def frequency_mask(self, spectrogram, mask_param=15):
        freq_mask = T.FrequencyMasking(freq_mask_param=mask_param).to(self.my_device)
        return freq_mask(spectrogram)


    def random_crop(self, spectrogram, crop_size):
        max_start = spectrogram.size(-1) - crop_size
        if max_start > 0:
            start = random.randint(0, max_start)
            return spectrogram[:, :, start:start + crop_size]
        return spectrogram


    def add_noise(self, spectrogram, noise_level=0.005):
        noise = noise_level * torch.randn_like(spectrogram).to(self.my_device)
        return spectrogram + noise


    def __call__(self, waveform):
        # Convert waveform to mel-spectrogram
        mel_spectrogram = self.mel_spectrogram(waveform)
        mel_spectrogram = self.amplitude_to_db(mel_spectrogram)
        
        # Apply random augmentations
        transform1, transform2 = random.sample(self.transforms, 2)
        return transform1(mel_spectrogram), transform2(mel_spectrogram)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")

    config = yaml_config_hook("config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dataset = BeatportDataset(
        dataset_dir=args.dataset_dir,
        args=args,
        split="train",
        n_fft=args.n_fft, 
        hop_length=args.hop_length,
    )
    print("LEN", len(train_dataset))
    
    train_dataset = BeatportDataset(
        dataset_dir=args.dataset_dir,
        args=args,
        split="train",
        n_fft=args.n_fft, 
        hop_length=args.hop_length,
        filter_short=True,
        pre_process=True,
    )
    print("After filter and pre_process LEN", len(train_dataset))
    
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers)
