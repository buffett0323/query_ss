import torch
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T
from torchvision.transforms.functional import crop
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


import glob
import os
import yaml
import json
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

def custom_collate_fn(batch):
    mix_melspec = torch.stack([item[0] for item in batch])
    stems_melspec = torch.stack([item[1] for item in batch])
    pitch_annotation = torch.stack([item[2] for item in batch])
    return mix_melspec, stems_melspec, pitch_annotation

   
def music_object_collate_fn(batch):
    chord_spec = torch.stack([item[0] for item in batch])
    note_specs = [item[1] for item in batch]
    midi_label = [item[2] for item in batch]
    inst_label = [item[3] for item in batch]
    return chord_spec, note_specs, midi_label, inst_label
   
   
def spec_crop(image, height, width):
    return crop(image, top=0, left=0, height=height, width=width)

def instrument_to_int(instrument_label, ref):
    int_label = np.array([np.where(inst == ref) for inst in instrument_label]).squeeze()
    return torch.from_numpy(int_label)

def midi2int(note, note_list):
    # note_list = [43, 45, 46] + [i for i in range(48, 97)]
    return note_list.index(note)

def chord2int(chord, note_list):
    chord_int = [torch.from_numpy(
        np.array(midi2int(note.item(), note_list))).unsqueeze(dim=0) for note in chord if note.item() != 0]
    return torch.cat(chord_int)

def int2chord(chord_int, note_list):
    # note_list = [43, 45, 46] + [i for i in range(48, 97)]

    if len(chord_int) == 4:
        return [note_list[note] for note in chord_int]
    else:
        return [0] * (4 - len(chord_int)) + [note_list[note] for note in chord_int]

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", savefig=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (energy)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    # im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    im = axs.imshow(specgram, origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    if savefig is not None:
        plt.savefig(savefig)
    plt.show(block=False)

def plot_multiple_spectrograms(specgrams: list,
                               figsize: tuple,
                               fontsize=None,
                               title=None,
                               suptitle=None,
                               xlabel='Time',
                               ylabel='Mel Bins',
                               colorbar_label='Decibels / dB',
                               aspect="auto"):
    
    fig, axs = plt.subplots(
        1, len(specgrams), figsize=figsize)

    vmin = np.min(get_vmin(specgrams))
    vmax = np.max(get_vmax(specgrams))

    if suptitle is not None:
        plt.suptitle(
            suptitle,
            fontsize=int(36 * max(figsize) / 32) if fontsize is None else fontsize)
    
    for i, spec in enumerate(specgrams):
        if title is not None:
            assert len(title) == len(specgrams)
            axs[i].set_title(title[i])
        if i == 0:
            axs[i].set_ylabel(ylabel)
        else:
            axs[i].set_yticks([])
        axs[i].set_xlabel(xlabel)
        im = axs[i].imshow(spec, origin="lower", aspect=aspect, vmin=vmin, vmax=vmax)
    
    fig.colorbar(im, ax=axs.ravel().tolist(), label=colorbar_label)
    plt.show()
    return fig

def get_vmin(spec_list):
    vmin = torch.inf
    for spec in spec_list:
        if np.min(spec) < vmin:
            vmin = np.min(spec)
    return vmin

def get_vmax(spec_list):
    vmax = -torch.inf
    for spec in spec_list:
        if np.max(spec) > vmax:
            vmax = np.max(spec)
    return vmax

def compare_spectrograms(specs1: list,
                         specs2: list,
                         figsize: tuple,
                         fontsize=None,
                         suptitle=None,
                         subtitle1=None,
                         subtitle2=None,
                         xlabel='Time',
                         ylabel='Mel Bins',
                         colorbar_label='Decibels / dB',
                         aspect="auto"):

    num_specs = len(specs1) if len(specs1) > len(specs2) else len(specs2)
    vmin = min(get_vmin(specs1), get_vmin(specs2))
    vmax = max(get_vmax(specs1), get_vmax(specs2))

    if subtitle1 is not None:
        assert len(subtitle1) == len(specs1), 'subtitle1 length: {}, specs1 length: {}'.format(len(subtitle1), len(specs1))
    if subtitle2 is not None:
        assert len(subtitle2) == len(specs2), 'subtitle2 length: {}, specs2 length: {}'.format(len(subtitle2), len(specs2))
    
    fig, axs = plt.subplots(
        2, num_specs, figsize=figsize)

    for i, spec1 in enumerate(specs1):
        axs[0, i].set_xticks([])
        if i == 0:
            axs[0, i].set_ylabel(ylabel)
            if subtitle1 is not None:
                axs[0, i].title.set_text(subtitle1[i])
        else:
            axs[0, i].set_yticks([])
            if subtitle1 is not None:
                axs[0, i].title.set_text(subtitle1[i])
                
        im = axs[0, i].imshow(spec1, origin="lower", aspect=aspect, vmin=vmin, vmax=vmax)
        
    for j, spec2 in enumerate(specs2):
        axs[1, j].set_xlabel(xlabel)
        if j == 0:
            axs[1, j].set_ylabel(ylabel)
            if subtitle2 is not None:
                axs[1, j].title.set_text(subtitle2[j])
        else:
            axs[1, j].set_yticks([])
            if subtitle2 is not None:
                axs[1, j].title.set_text(subtitle2[j])
            if j >= len(specs1):
                axs[0, j].set_visible(False)
        im = axs[1, j].imshow(spec2, origin="lower", aspect=aspect, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axs.ravel().tolist(), label=colorbar_label)

    if suptitle is not None:
        plt.suptitle(
            suptitle,
            fontsize=int(36 * max(figsize) / 32) if fontsize is None else fontsize)
    plt.show(block=False)
    return fig


class MusicalObjectDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 to_db: bool = True,
                 spec: str = 'mel',
                 top_db: float = 80.0,
                 transform = None):
        super().__init__()

        self.root = root
        self.split = split

        json_file = os.path.join(root, 'mel_metadata.json')

        with open(json_file, 'rb') as handle:
            self.metadata = json.load(handle)

        self.all_instrument_names = self.metadata['all_instrument_names']

        self.to_db = to_db
        self.spec = spec
        self.top_db = top_db
        self.transform = transform

        self.out_path = os.path.join(self.root, self.split)
        self.spec_list = []

        self.spec_list = sorted(
            glob.glob(
                os.path.join(
                    self.out_path,
                    '{}_spec-*-of-*.pt'.format(spec)
                    )
                )
            )
        self.instrument_list = np.load(
            os.path.join(self.out_path, '{}_instrument_order.npy'.format(split)),
            allow_pickle=True)
        
        self.examples = torch.load(
            os.path.join(self.out_path, '{}_examples.pt'.format(split)), weights_only=True)
        
        # notes = torch.unique(self.examples)
        # self.notes = notes[notes.nonzero(as_tuple=True)]
        self.notes = self.metadata["note_list"]
        self.instrument_tokens = np.array(self.metadata["instrument_tokens"])
    
    def __getitem__(self, index):
        spec_file = self.spec_list[index]
        spec = torch.load(spec_file, weights_only=True)
        instrument_list = self.instrument_list[index]

        if self.to_db:
            spec = AF.amplitude_to_DB(spec, 10, 1e-10, 0, top_db=self.top_db)

        chord = self.examples[index % len(self.examples)]
        note_spec_list = []
        note_audio_list = []

        for note, inst in zip(chord[chord.nonzero(as_tuple=True)], instrument_list):
            instrument_name = self.all_instrument_names[inst].replace(" ", "")
            note_spec = torch.load(os.path.join(
                self.root, 'notes/{}'.format(instrument_name), str(note.item()),
                '{}_spec.pt'.format(self.spec)), weights_only=True)
            note_audio_tmp = torch.load(os.path.join(
                self.root, 'notes/{}'.format(instrument_name), str(note.item()),
                'audio.pt'))
            note_spec_list.append(note_spec)
            note_audio_list.append(note_audio_tmp)

        note_tensors = torch.cat(note_spec_list, dim=0)
        example_audio_tensor = torch.cat(
                note_audio_list, dim=0).sum(dim=0).unsqueeze(dim=0)

        spec_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=None,
            hop_length=512,
            n_mels = 128
        )
        example_spec = spec_transform(example_audio_tensor)
        
        if self.transform is not None:
            spec = self.transform(spec)
            note_tensors = self.transform(note_tensors)

        midi_label = chord2int(chord, self.notes)
        instrument_label = instrument_to_int(instrument_list, self.instrument_tokens)

        # Mixture, note, midi label, instrument_label: 
        # torch.Size([1, 128, 35]) torch.Size([4, 128, 35]) tensor([18, 20, 25, 34]) tensor([0, 1, 1, 2])
        return spec.squeeze(0), note_tensors, midi_label, instrument_label

    def __len__(self):
        return len(self.spec_list)
    

class MusicalObjectDataModule(LightningDataModule):
    def __init__(self,
                 root: str,
                 batch_size: int,
                 to_db: bool = True,
                 spec: str = 'mel',
                 top_db: float = 80.0,
                 num_workers: int = 0,
                 seed: int = 42,
                 shuffle: bool = True,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 train_transforms = None,
                 val_transforms = None,
                 test_transforms = None,
                 *args,
                 **kwargs):
        
        super(MusicalObjectDataModule, self).__init__(
            *args, **kwargs
        )

        self.root = root
        self.batch_size = batch_size

        self.to_db = to_db
        self.spec = spec
        self.top_db = top_db
        
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_ds = MusicalObjectDataset(
                root=self.root,
                split='train',
                to_db=self.to_db,
                spec=self.spec,
                top_db=self.top_db,
                transform=self.train_transforms)
            
            self.val_ds = MusicalObjectDataset(
                root=self.root,
                split='valid',
                to_db=self.to_db,
                spec=self.spec,
                top_db=self.top_db,
                transform=self.val_transforms)
        
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = MusicalObjectDataset(
                root=self.root,
                split='test',
                to_db=self.to_db,
                spec=self.spec,
                top_db=self.top_db,
                transform=self.test_transforms)
            
    def train_dataloader(self):
        """The train dataloader."""
        return self._data_loader(
            self.train_ds,
            shuffle=self.shuffle)

    def val_dataloader(self):
        """The val dataloader."""
        return self._data_loader(
            self.val_ds,
            shuffle=False)

    def test_dataloader(self):
        """The test dataloader."""
        return self._data_loader(
            self.test_ds,
            shuffle=False)
            
    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=music_object_collate_fn,
        )
    
    @property
    def num_samples(self) -> int:
        self.setup(stage = 'fit')
        return len(self.train_ds)

    @property
    def num_notes(self) -> int:
        self.setup(stage = 'fit')
        return len(self.train_ds.notes)
    
    @property
    def num_instruments(self) -> int:
        self.setup(stage = 'fit')
        return len(self.train_ds.instrument_tokens)
    

class CocoChoraleDataset(Dataset):
    def __init__(
        self, 
        N_s=4,
        file_dir='/home/buffett/NAS_189/cocochorales_full_v1_output/main_dataset',
        split='train',
        segment_duration=4.0,
        ensemble="random",
        sample_rate=16000, 
        n_mels=64, 
        n_fft=1024, 
        hop_length=160,
        num_pitches=129,
    ):
        self.transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        self.N_s = N_s
        self.segment_duration = segment_duration
        self.ensemble = ensemble
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_pitches = num_pitches
        self.segment_length = int(self.sample_rate // self.hop_length * self.segment_duration)
        self.file_dir = os.path.join(file_dir, split)
        self.file_path = [os.path.join(self.file_dir, i) for i in os.listdir(path=self.file_dir) if i.startswith(f'{ensemble}_track')]
    
    
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
        mix_audio = os.path.join(audio_path, 'mix.wav')
        
        with open(os.path.join(audio_path, 'metadata.yaml'), 'r') as file:
            stems = yaml.safe_load(file).get('instrument_name', {})
        stems_audio = [os.path.join(audio_path, 'stems_audio', f'{i+1}_{j}.wav') for i, j in stems.items()]

        # Load audio
        mix_wav, _ = torchaudio.load(mix_audio)
        stems_wav = [torchaudio.load(i)[0] for i in stems_audio]

        # Apply MelSpectrogram
        mix_melspec = self.transform(mix_wav)
        stems_melspec = [self.transform(i) for i in stems_wav]
        
        # Randomly sample segment
        start_frame = np.random.randint(0, mix_melspec.shape[-1] - self.segment_length + 1)
        mix_melspec = mix_melspec[:, :, start_frame:start_frame + self.segment_length]
        stems_melspec = [i[:, :, start_frame:start_frame + self.segment_length] for i in stems_melspec]

        # Get pitch annotation by reading pickle
        csv_folder = audio_path.replace('main_dataset', 'note_expression')
        pitch_annotation = []
        for csv_file in os.listdir(csv_folder):
            csv_file = os.path.join(csv_folder, csv_file)
            csv_df = pd.read_csv(csv_file)
            pitch_sequence = self.get_pitch_sequence(csv_df, start_frame)
            pitch_annotation.append(pitch_sequence)  
        
        pitch_annotation = [torch.tensor(i).unsqueeze(0) if not isinstance(i, torch.Tensor) else i for i in pitch_annotation]

        stems_melspec = torch.cat(stems_melspec, dim=0)
        pitch_annotation = torch.cat(pitch_annotation, dim=0)

        return mix_melspec, stems_melspec, pitch_annotation

class CocoChoraleDataModule(LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int = 8,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args,
        **kwargs
    ):
        
        super(CocoChoraleDataModule, self).__init__(*args, **kwargs)

        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_ds = CocoChoraleDataset(file_dir=self.root, split='train')
            self.val_ds = CocoChoraleDataset(file_dir=self.root, split='valid')
        
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = CocoChoraleDataset(file_dir=self.root, split='test')
            
    def train_dataloader(self):
        """The train dataloader."""
        return self._data_loader(
            self.train_ds,
            shuffle=self.shuffle)

    def val_dataloader(self):
        """The val dataloader."""
        return self._data_loader(
            self.val_ds,
            shuffle=False)

    def test_dataloader(self):
        """The test dataloader."""
        return self._data_loader(
            self.test_ds,
            shuffle=False)
            
    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=custom_collate_fn
        )
    
    @property
    def num_samples(self) -> int:
        self.setup(stage = 'fit')
        return len(self.train_ds)

    # @property
    # def num_notes(self) -> int:
    #     self.setup(stage = 'fit')
    #     return len(self.train_ds.notes)
    
    # @property
    # def num_instruments(self) -> int:
    #     self.setup(stage = 'fit')
    #     return len(self.train_ds.instrument_tokens)



if __name__ == '__main__':
    
    # check that all datasets load correctly
    comp_path = "/mnt/gestalt/home/ddmanddman"
    root = f"{comp_path}/cocochorales_output/main_dataset"
    
    dm = CocoChoraleDataModule(root=root, batch_size=2)
    dm.setup(stage='fit')

    print('Dataset sample count: {}'.format(dm.num_samples))
    
    mix_melspec, stems_melspec, pitch_annotation = dm.train_ds[0]
    print(mix_melspec.shape, stems_melspec.shape, pitch_annotation.shape)
    mix_melspec, stems_melspec, pitch_annotation = dm.val_ds[0]
    print(mix_melspec.shape, stems_melspec.shape, pitch_annotation.shape)
    
    for batch in dm.train_dataloader():
        x_m, x_s_i, pitch_annotation = batch
        print(x_m.shape, x_s_i.shape, pitch_annotation.shape); break
    
    
        


