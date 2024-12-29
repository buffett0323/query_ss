import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram
import torchaudio
import torchaudio.transforms as T

class BeatportDataset(Dataset):
    def __init__(
        self, 
        dataset_dir, 
        split="train", 
        transform=None
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
        

        self.audio_files = [
            os.path.join(dataset_dir, folder_name, file_name)
            for folder_name in os.listdir(dataset_dir)
            for file_name in os.listdir(os.path.join(dataset_dir, folder_name))
            if file_name.endswith(".mp3")
        ]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Load the audio file
        audio_path = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        # TODO: transform random
        mel1 = self.transform(waveform)
        mel2 = self.transform(waveform)
        return (mel1, mel2)
    


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
            self.time_mask(),
            self.frequency_mask(),
            self.add_noise(),
            self.random_crop(),
        ]
    
    def time_mask(self, spectrogram, mask_param=30):
        """Apply time masking."""
        time_mask = T.TimeMasking(time_mask_param=mask_param)
        return time_mask(spectrogram)

    def frequency_mask(self, spectrogram, mask_param=15):
        """Apply frequency masking."""
        freq_mask = T.FrequencyMasking(freq_mask_param=mask_param)
        return freq_mask(spectrogram)

    def random_crop(self, spectrogram, crop_size):
        """Crop a random segment of the spectrogram."""
        max_start = spectrogram.size(-1) - crop_size
        if max_start > 0:
            start = random.randint(0, max_start)
            return spectrogram[:, :, start:start + crop_size]
        return spectrogram

    def add_noise(self, spectrogram, noise_level=0.005):
        """Additive Gaussian Noise to the spectrogram."""
        noise = noise_level * torch.randn_like(spectrogram)
        return spectrogram + noise

    def __call__(self, waveform):
        # Convert waveform to mel-spectrogram
        mel_spectrogram = self.mel_spectrogram(waveform)
        mel_spectrogram = self.amplitude_to_db(mel_spectrogram)

        # Apply random augmentations
        transform = random.choice(self.transforms)
        return transform(mel_spectrogram)
    
    

if __name__ == "__main__":

    dataset_dir="/mnt/gestalt/home/ddmanddman/beatport_preprocess/chorus"
    train_dataset = BeatportDataset(
        dataset_dir=dataset_dir,
        split="train",
        transform=SimCLRTransform(),
    )
    print(len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
