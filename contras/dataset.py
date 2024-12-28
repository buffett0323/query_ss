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
        if self.transform:
            waveform1 = self.transform(waveform)
            waveform2 = self.transform(waveform)
            return (waveform1, waveform2)

        return waveform
    


class TransformsSimCLR:
    def __init__(self, n_mels=128, sample_rate=44100):
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=512,
            n_fft=2048,
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        self.add_noise = lambda x: x + 0.005 * torch.randn_like(x)
        

    def __call__(self, waveform):
        # Apply mel spectrogram and amplitude to db
        mel = self.mel_spectrogram(waveform)
        mel_db = self.amplitude_to_db(mel)

        # Apply data augmentation (e.g., adding noise)
        mel_db = self.add_noise(mel_db)
        return mel_db

class AudioTransformsSimCLR:
    def __init__(self):
        self.transforms = [
            T.TimeStretch(),
            T.PitchShift(sample_rate=44100, n_steps=random.choice([-2, 2])),
            T.TimeMasking(time_mask_param=30),
            T.FrequencyMasking(freq_mask_param=15),
        ]

    def __call__(self, waveform):
        transform = random.choice(self.transforms)
        return transform(waveform)



if __name__ == "__main__":

    dataset_dir="/mnt/gestalt/home/ddmanddman/beatport_preprocess/chorus"
    train_dataset = BeatportDataset(
        dataset_dir=dataset_dir,
        split="train",
        transform=TransformsSimCLR(n_mels=128, sample_rate=44100),
    )
    print(len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
