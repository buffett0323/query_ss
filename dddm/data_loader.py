import random
import torch
import os
import librosa
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchaudio.functional as AF
import torchaudio.transforms as T

from torchvision.transforms.functional import crop
from torchvision import transforms
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from typing import Optional
from functools import partial
from tqdm import tqdm

import utils

np.random.seed(1234)
class BP_DDDM_Dataset(torch.utils.data.Dataset):
    """
    Provides dataset management for given filelist.
    """
    def __init__(
        self, 
        config, 
        split="train",
        stems=["other"], #["vocals", "bass", "drums", "other"], # VBDO
        training=True
    ):
        super(BP_DDDM_Dataset, self).__init__()
        self.config = config
        self.hop_length = config.data.hop_length
        self.training = training
        self.mel_length = config.train.segment_size // config.data.hop_length
        if self.training:
            self.segment_length = config.train.segment_size
        self.sample_rate = config.data.sampling_rate
        
        # Load split files from txt file
        with open(f"../simsiam/info/{split}_bp_8secs.txt", "r") as f:
            bp_listdir = [line.strip() for line in f.readlines()]

        self.stems = stems
        self.audio_paths = [
            os.path.join(config.data.filelist_path, folder, f"{stem}.npy")
            for folder in bp_listdir
                for stem in stems
        ]

    def load_audio_to_torch(self, audio_path):
        # audio, sample_rate = torchaudio.load(audio_path)
        audio = np.load(audio_path)
        audio = torch.tensor(audio)

        if not self.training:
            p = (audio.shape[-1] // 1280 + 1) * 1280 - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, p), mode='constant').data
        return audio#.squeeze()


    def mel_timbre(self, x):
        # Infos:
        img_mean = -1.100174903869629
        img_std = 14.353998184204102
        
        # To numpy
        x = x.numpy()
        x = librosa.feature.melspectrogram(
            y=x, 
            sr=16000, 
            n_fft=1024,
            hop_length=256,
        )
        x = librosa.power_to_db(np.abs(x))
        x = torch.from_numpy(x).unsqueeze(0)
        
        # Resize to 256x256
        resizer = transforms.Resize((256, 256))
        x = resizer(x)
        return (x - img_mean) / img_std
    

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        audio = self.load_audio_to_torch(audio_path)

        if not self.training:  
            return audio#, f0_norm
        
        if audio.shape[-1] > self.segment_length:
            audio_start = np.random.randint(0, audio.shape[-1] - self.segment_length + 1)
            audio_segment = audio[audio_start:audio_start + self.segment_length]
            length = torch.LongTensor([self.mel_length])
        else:
            audio_segment = torch.nn.functional.pad(
                audio, (0, self.segment_length - audio.shape[-1]), 'constant'
            ).data
            length = torch.LongTensor([audio.shape[-1] // self.hop_length])
        
        return audio_segment, length
        # # Get Mel-Spectrogram for Timbre Encoder
        # mel_audio = self.mel_timbre(audio_segment)
        # return audio_segment, mel_audio, length


    def __len__(self):
        return len(self.audio_paths)






class CocoChorale_Simple_DS(Dataset):
    def __init__(
        self, 
        config, 
        split="train", 
        training=True,
        ensemble="random",
    ):
        super(CocoChorale_Simple_DS, self).__init__()
        self.split = split
        self.config = config
        self.hop_length = config.data.hop_length
        self.training = training
        self.mel_length = config.train.segment_size // config.data.hop_length
        if self.training:
            self.segment_length = config.train.segment_size
        self.sample_rate = config.data.sampling_rate
        self.filelist_path = []
        self.file_dir = os.path.join(config.data.filelist_path, split)
            
        txt_file = os.path.join(config.data.txt_path, f"{split}_list.txt")
        if os.path.exists(txt_file):
            with open(txt_file, "r") as f:
                self.filelist_path = [line.strip() for line in f] 

        else:
            for track in tqdm(os.listdir(self.file_dir)):
                if track.startswith(f'{ensemble}_track'):
                    for stem in os.listdir(os.path.join(self.file_dir, track, "stems_audio")):
                        if stem.endswith(".wav"):
                            stem_path = os.path.join(self.file_dir, track, "stems_audio", stem)
                            self.filelist_path.append(stem_path)

            with open(txt_file, "w") as f:
                for path in self.filelist_path:
                    f.write(path + "\n")

    def __len__(self):
        return len(self.filelist_path)
    
    
    def get_pitch_sequence(self, dataframe, start):
        end = start + self.segment_length
        pitch_sequence = []
        
        for _, row in dataframe.iterrows():
            onset = row['onset']
            offset = row['offset']
            pitch = row['pitch']
            
            # Check if the current note overlaps with the specified range
            if onset < end and offset > start:
                # Determine the overlap range
                overlap_start = max(onset, start)
                overlap_end = min(offset, end)
                pitch_sequence.extend([int(pitch)] * int(overlap_end - overlap_start))
        
        return pitch_sequence

    
    def __getitem__(self, idx):
        # TODO: npy faster load
        audio_path = self.filelist_path[idx]
        audio, _ = torchaudio.load(audio_path)
        audio = audio.squeeze()
        
        if not self.training:  
            return audio
        
        # TODO: CSV file
        if audio.shape[-1] > self.segment_length:
            audio_start = np.random.randint(0, audio.shape[-1] - self.segment_length + 1)
            audio_segment = audio[audio_start:audio_start + self.segment_length]
            length = torch.LongTensor([self.mel_length])
        else:
            audio_segment = torch.nn.functional.pad(
                audio, (0, self.segment_length - audio.shape[-1]), 'constant'
            ).data
            length = torch.LongTensor([audio.shape[-1] // self.hop_length])
            
        return audio_segment, length
        



class MelSpectrogramFixed(torch.nn.Module):
    """In order to remove padding of torchaudio package + add log10 scale."""

    def __init__(self, **kwargs):
        super(MelSpectrogramFixed, self).__init__()
        self.torchaudio_backend = MelSpectrogram(**kwargs)

    def forward(self, x):
        outputs = torch.log(self.torchaudio_backend(x) + 0.001)

        return outputs[..., :-1]


if __name__ == "__main__":
    hps = utils.get_hparams()
    n_gpus = 1
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    
    test_dataset = BP_DDDM_Dataset(hps, split="test", training=False)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=4, #hps.train.batch_size, 
        num_workers=hps.train.num_workers,
        shuffle=False,
    )
    
    train_dataset = BP_DDDM_Dataset(hps, split="train", training=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, #hps.train.batch_size, 
        num_workers=hps.train.num_workers,
        shuffle=True,
    )
    
    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    )
    
    
    # # HIFIGAN TESTING
    # from vocoder.hifigan import HiFi
    # import utils
    
    # hps = utils.get_hparams()
    # net_v = HiFi(
    #     hps.data.n_mel_channels,
    #     hps.train.segment_size // hps.data.hop_length,
    #     **hps.model).to(device)
    # path_ckpt = '/mnt/gestalt/home/ddmanddman/hifigan_ckpt/voc_ckpt.pth'

    # utils.load_checkpoint(path_ckpt, net_v, None)
    # net_v.eval()
    # net_v.dec.remove_weight_norm()
    
    
    
    for y in test_loader:
        print(y)
        y_mel = mel_fn(y).to(device)
        print(y_mel.shape)
        
        # torchaudio.save("examples/orig_y2.wav", y.cpu(), 16000)
        # torchaudio.save("examples/recon_y2.wav", recon_y.cpu(), 16000)
        break
    
    for (y, length) in train_loader:
        print(y.shape, length.shape)
        y_mel = mel_fn(y).to(device)
        print(y_mel.shape)
        
        # torchaudio.save("examples/orig_y2.wav", y.cpu(), 16000)
        # torchaudio.save("examples/recon_y2.wav", recon_y.cpu(), 16000)
        break