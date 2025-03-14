import random
import torch
import os
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
        fmax=8000,
        img_size=256,
        img_mean=0,
        img_std=0,
    ):
        # Load split files from txt file
        with open(f"../simsiam/info/{split}_bp_8secs.txt", "r") as f:
            bp_listdir = [line.strip() for line in f.readlines()]

        self.stems = stems
        self.data_path_list = [
            os.path.join(data_dir, folder, f"{stem}.npy")
            for folder in bp_listdir
                for stem in stems
        ]

        self.sample_rate = sample_rate
        self.segment_second = segment_second
        self.duration = sample_rate * piece_second # 4 seconds for each piece
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
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_max=fmax,
        )
        self.db_transform = T.AmplitudeToDB(
            stype="power"
        )
        self.resizer = transforms.Resize((img_size, img_size))
        self.img_mean = img_mean
        self.img_std = img_std

    
    def __len__(self): #""" Total we got 175698 files * 4 tracks """
        return len(self.data_path_list)
    
    
    def mel_spec_transform(self, x):
        x = x.float()
        mel_spec = self.mel_transform(x).float()
        return self.db_transform(mel_spec)
        
    
    
    def data_pipeline(self, x):
        x = torch.tensor(x)
        x = self.mel_spec_transform(x).unsqueeze(0)
        x = self.resizer(x) # transform to 1, img_size, img_size
        return (x - self.img_mean) / self.img_std


    def random_crop(self, x):
        """ Random crop for 4 seconds """
        max_idx = int(x.shape[0]) - self.duration
        idx = random.randint(0, max_idx)
        return x[idx:idx+self.duration]


    def __getitem__(self, idx):
        """ 
            1. Mel-Spectrogram Transformation
            2. Resize
            3. Normalization
            4. Data Augmentation
        """
        # Load audio data
        path = self.data_path_list[idx]
        x_audio = np.load(path)
        
        
        # Random Crop
        if self.random_slice:
            x_audio = self.random_crop(x_audio)
        
        # Mel-Spectrogram Transformation
        x_mel = self.data_pipeline(x_audio)
        
        return x_mel.float(), x_audio.float(), path





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
            
        
        # # Get pitch annotation by reading pickle
        # stem_csv_file = os.path.basename(audio_path).replace(".wav", ".csv")
        # stem_csv_file = str(int(stem_csv_file[0])-1) + str(stem_csv_file[1:])
        # csv_file = os.path.join(
        #     audio_path.split("/stems_audio")[0].replace("main_dataset", "note_expression"), 
        #     stem_csv_file)
        
        # csv_df = pd.read_csv(csv_file)
        # print(audio_start, csv_file)
        # pitch_sequence = self.get_pitch_sequence(csv_df, audio_start)
        # pitch_annotation = torch.tensor(pitch_sequence).unsqueeze(0) if not isinstance(pitch_sequence, torch.Tensor) else pitch_sequence
        # print(pitch_annotation)
        return audio_segment, length # torch.tensor(pitch_sequence)
        

class CocoChoraleDataset(Dataset):
    def __init__(
        self, 
        N_s=4,
        file_dir='/mnt/gestalt/home/ddmanddman/cocochorales_output/main_dataset',
        split='train',
        segment_duration=4.0,
        ensemble="random",
        sample_rate=16000, 
        n_mels=64, 
        n_fft=1024, 
        hop_length=160,
        num_pitches=129,
        training=True,
    ):
        self.N_s = N_s
        self.segment_duration = segment_duration
        self.ensemble = ensemble
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_pitches = num_pitches
        self.training = training
        self.segment_length = int(self.sample_rate // self.hop_length * self.segment_duration)
        self.file_dir = os.path.join(file_dir, split)
        self.file_path = [
            os.path.join(self.file_dir, i) 
            for i in os.listdir(path=self.file_dir) 
                if i.startswith(f'{ensemble}_track')
        ]
        
    
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


    def __len__(self):
        return len(self.file_path)
    
    
    def __getitem__(self, idx):
        audio_path = self.file_path[idx]
        mix_melspec = np.load(os.path.join(audio_path, 'mix_melspec.npy')) #, mmap_mode='r')
        stems_melspec = np.load(os.path.join(audio_path, 'stems_melspec.npy')) #, mmap_mode='r')
        
        # Randomly sample segment
        start_frame = np.random.randint(0, mix_melspec.shape[-1] - self.segment_length + 1)
        mix_melspec = mix_melspec[:, :, start_frame:start_frame + self.segment_length]
        stems_melspec = stems_melspec[:, :, start_frame:start_frame + self.segment_length]

        # Get pitch annotation by reading pickle
        csv_folder = audio_path.replace('main_dataset', 'note_expression')
        pitch_annotation = []
        for csv_file in os.listdir(csv_folder):
            csv_file = os.path.join(csv_folder, csv_file)
            csv_df = pd.read_csv(csv_file)
            pitch_sequence = self.get_pitch_sequence(csv_df, start_frame)
            pitch_annotation.append(pitch_sequence)  
        
        pitch_annotation = [torch.tensor(i).unsqueeze(0) if not isinstance(i, torch.Tensor) else i for i in pitch_annotation]
        pitch_annotation = torch.cat(pitch_annotation, dim=0)


        return torch.tensor(mix_melspec), torch.tensor(stems_melspec), pitch_annotation



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
    
    train_dataset = CocoChorale_Simple_DS(hps, split="train", training=True)
    train_sampler = None
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, #hps.train.batch_size, 
        num_workers=hps.train.num_workers,
        sampler=train_sampler, 
        drop_last=True, 
        persistent_workers=True, 
        pin_memory=True,
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
    
    
    # HIFIGAN TESTING
    from vocoder.hifigan import HiFi
    import utils
    
    hps = utils.get_hparams()
    net_v = HiFi(
        hps.data.n_mel_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    path_ckpt = '/mnt/gestalt/home/ddmanddman/hifigan_ckpt/voc_ckpt.pth'

    utils.load_checkpoint(path_ckpt, net_v, None)
    net_v.eval()
    net_v.dec.remove_weight_norm()
    
    
    
    for (y, length) in train_loader:
        y_mel = mel_fn(y).cuda()
        print(y.shape, y_mel.shape)

        recon_y = net_v(y_mel).squeeze(1)
        print(recon_y.shape)
        
        torchaudio.save("examples/orig_y2.wav", y.cpu(), 16000)
        torchaudio.save("examples/recon_y2.wav", recon_y.cpu(), 16000)
        break