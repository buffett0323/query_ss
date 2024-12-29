import os
import random
import shutil
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm
from mutagen.mp3 import MP3


class BeatportDataset(Dataset):
    def __init__(
        self, 
        dataset_dir, 
        split="train", 
        transform=None,
        max_length=64,
        n_fft=2048, 
        hop_length=1024,
        filter=False,
    ):
        """
        Args:
            dataset_dir (str): Path to the dataset directory.
            split (str): The split to load ('train', 'test', 'unlabeled').
            transform (callable, optional): A function/transform to apply to the audio data.
        """
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = n_fft + hop_length
        
        if filter: self.filter()
        self.audio_files = [
            os.path.join(dataset_dir, folder_name, file_name)
            for folder_name in os.listdir(dataset_dir)
            for file_name in os.listdir(os.path.join(dataset_dir, folder_name))
            if file_name.endswith(".mp3")
        ]
        
    def filter(self, min_duration=1):
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


    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Load the audio file
        audio_path = self.audio_files[idx]
        waveform, _ = torchaudio.load(audio_path)

        # TODO: transform random
        mel1, mel2 = self.transform(waveform)
        mel1, mel2 = self.consist_size(mel1), self.consist_size(mel2)
        return mel1, mel2
    
    def consist_size(self, mel_spectrogram):
        if mel_spectrogram.size(-1) < self.max_length:
            pad_size = self.max_length - mel_spectrogram.size(-1)
            return torch.nn.functional.pad(mel_spectrogram, (0, pad_size))
        else:
            return mel_spectrogram[:, :, :self.max_length]
        
    def replicate_waveform(self, waveform):

        current_length = waveform.size(-1)
        repeat_count = self.target_length // current_length
        remainder = self.target_length % current_length

        # Replicate and concatenate the waveform
        repeated_waveform = waveform.repeat(1, repeat_count)  # Repeat full waveform
        if remainder > 0:
            repeated_waveform = torch.cat([repeated_waveform, waveform[:, :remainder]], dim=-1)

        return repeated_waveform
        


class SimCLRTransform:
    def __init__(self, sample_rate=44100, n_mels=128, n_fft=2048, hop_length=1024):
        """
        A transform class for mel-spectrograms with various audio augmentations.
        
        Args:
            sample_rate (int): Sample rate of the audio.
            n_mels (int): Number of mel bands.
            n_fft (int): Size of FFT window.
            hop_length (int): Hop length between frames.
        """
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        self.transforms = [
            lambda spectrogram: self.time_mask(spectrogram, mask_param=30),
            lambda spectrogram: self.frequency_mask(spectrogram, mask_param=15),
            lambda spectrogram: self.random_crop(spectrogram, crop_size=128),
            lambda spectrogram: self.add_noise(spectrogram, noise_level=0.005),
        ]

    def time_mask(self, spectrogram, mask_param=30):
        time_mask = T.TimeMasking(time_mask_param=mask_param)
        return time_mask(spectrogram)

    def frequency_mask(self, spectrogram, mask_param=15):
        freq_mask = T.FrequencyMasking(freq_mask_param=mask_param)
        return freq_mask(spectrogram)

    def random_crop(self, spectrogram, crop_size):
        max_start = spectrogram.size(-1) - crop_size
        if max_start > 0:
            start = random.randint(0, max_start)
            return spectrogram[:, :, start:start + crop_size]
        return spectrogram

    def add_noise(self, spectrogram, noise_level=0.005):
        noise = noise_level * torch.randn_like(spectrogram)
        return spectrogram + noise

    def __call__(self, waveform):
        # Convert waveform to mel-spectrogram
        mel_spectrogram = self.mel_spectrogram(waveform)
        mel_spectrogram = self.amplitude_to_db(mel_spectrogram)
        
        # Apply random augmentations
        transform1, transform2 = random.sample(self.transforms, 2)
        return transform1(mel_spectrogram), transform2(mel_spectrogram)
    
    

if __name__ == "__main__":

    dataset_dir="/mnt/gestalt/home/ddmanddman/beatport_preprocess/chorus"
    train_dataset = BeatportDataset(
        dataset_dir=dataset_dir,
        split="train",
        transform=SimCLRTransform(),
        filter=True,
    )
    # for i in range(100):
    #     print(train_dataset[i][0].shape, train_dataset[i][1].shape)
