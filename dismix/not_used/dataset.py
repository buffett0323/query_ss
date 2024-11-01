import torch
import torchaudio
import pretty_midi
import os
import pickle
import glob
import json

import numpy as np
import matplotlib.pyplot as plt
import torchaudio.functional as AF
from torchvision.transforms.functional import crop
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from torch.utils.data import Dataset, DataLoader
from typing import Optional
from midi2audio import FluidSynth
from collections import defaultdict
from tqdm import tqdm


class CocoChoralesTinyDataset(Dataset):
    def __init__(
        self, 
        data_dir="/home/buffett/NAS_189/cocochorales_output/main_dataset/", 
        split="train",
        sample_rate=16000, 
        n_fft=1024,
        n_mels=128,
        hop_length=512,
        window_size=1024,
        time_resolution=0.01,  # For pitch extraction
        crop_frames=10,  # Number of frames to crop (320ms)
        start_pitch=33,
        num_pitch_classes=52,
    ):
        """
        Initialize the dataset.

        Parameters:
        - data_dir: Directory containing the dataset with mixture and query audio files.
        - sample_rate: Sample rate for audio processing.
        - n_mels: Number of mel frequency bins for spectrogram transformation.
        - time_resolution: Time resolution for pitch extraction.
        - crop_frames: Number of frames to crop from the sustain phase.
        """
        self.data_dir = os.path.join(data_dir, split)
        self.split = split
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.window_size = window_size
        self.time_resolution = time_resolution
        self.crop_frames = crop_frames
        self.strategy = "mode"
        self.start_pitch = start_pitch
        self.num_pitch_classes = num_pitch_classes
        
        self.file_list = self._load_folder_list(self.data_dir)
        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
    
    def _load_folder_list(self, data_dir):
        file_list = []
        if os.path.exists(f'{self.split}_file_list.pkl'):
            with open(f'{self.split}_file_list.pkl', 'rb') as file:
                file_list = pickle.load(file)
        else:
            for f in tqdm(os.listdir(data_dir)):
                for stem in os.listdir(os.path.join(data_dir, f, "stems_audio")):
                    file_list.append(os.path.join(data_dir, f, "stems_audio", stem))

            with open(f'{self.split}_file_list.pkl', 'wb') as file:
                pickle.dump(file_list, file)
                
        return file_list
    
    def _load_audio(self, file_path, start_frame=None):
        """
        Load an audio file and convert it to a mel spectrogram with proper cropping.
        
        Parameters:
        - file_path: Path to the audio file.
        - start_frame: Starting frame for cropping the same segment.
        
        Returns:
        - Cropped mel spectrogram tensor, start_frame (used for alignment).
        """
        # Load the waveform & Convert the waveform to a mel spectrogram
        waveform, _ = torchaudio.load(file_path)
        mel_spectrogram = self.mel_spectrogram_transform(waveform)
        
        # Convert amplitude to decibel scale
        mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel_spectrogram)
        
        # Crop a 320ms segment (10 frames) from the sustain phase
        cropped_mel_spectrogram, start_frame = self._crop_sustain_phase(mel_spectrogram_db.squeeze(0), crop_frames=self.crop_frames, start_frame=start_frame)
        
        return cropped_mel_spectrogram, start_frame
    
    def _crop_sustain_phase(self, mel_spectrogram, crop_frames=10, start_frame=None):
        """
        Crop a 320ms segment (10 frames) from the sustain phase of the mel spectrogram.
        
        Parameters:
        - mel_spectrogram: Mel spectrogram to crop.
        - crop_frames: Number of frames to crop (10 frames corresponds to 320ms).
        - start_frame: Starting frame for cropping (if None, find from sustain phase).
        
        Returns:
        - Cropped mel spectrogram segment, start_frame used for alignment.
        """
        # Calculate energy for each frame
        frame_energy = torch.sum(mel_spectrogram, dim=0)
        
        # Find the maximum energy frame index (attack phase) if start_frame is not provided
        if start_frame is None:
            max_energy_frame = torch.argmax(frame_energy)
            # Define the starting frame of the sustain phase, a few frames after the peak energy
            start_frame = max_energy_frame + 5  # Shift 5 frames after peak to avoid attack phase
        
        # Ensure the crop window does not exceed the spectrogram length
        if start_frame + crop_frames > mel_spectrogram.size(1):
            start_frame = max(0, mel_spectrogram.size(1) - crop_frames)
        
        # Crop the mel spectrogram segment
        cropped_segment = mel_spectrogram[:, start_frame:start_frame + crop_frames]
        
        return cropped_segment, start_frame

    def _extract_pitch_annotations(self, midi_file_path, start_time, end_time):
        """
        Extract pitch annotations for a specific segment from a MIDI file.

        Args:
            midi_file_path (str): Path to the MIDI file.
            start_time (float): Start time of the segment.
            end_time (float): End time of the segment.

        Returns:
            pitch_vector (np.ndarray): A binary vector of shape (52,) indicating pitch presence.
        """
        # Load the MIDI file using pretty_midi
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        
        # Define the pitch range for 52 possible pitches
        num_pitches = 128  # MIDI pitch values range from 0 to 127
        
        # Pitch annotation data
        pitch_annotation = np.zeros(self.num_pitch_classes, dtype=np.float32)

        num_time_steps = int(np.ceil((end_time - start_time) / self.time_resolution))
        pitch_matrix = np.zeros((num_time_steps, num_pitches), dtype=np.float32)
        
        # Iterate through each instrument in the MIDI file
        for idx, instrument in enumerate(midi_data.instruments):
            # Skip drum tracks if any
            if instrument.is_drum:
                continue
            # Iterate through each note in the instrument
            for note in instrument.notes:
                # Check if the note falls within the start and end time of the segment
                if note.start < end_time and note.end > start_time:
                    # Calculate the overlapping duration
                    overlap_start = max(note.start, start_time)
                    overlap_end = min(note.end, end_time)
                    overlap_duration = overlap_end - overlap_start
                    
                    if isinstance(overlap_duration, torch.Tensor):
                        overlap_duration = overlap_duration.numpy().astype(np.float32)
                    
                    start_frame = int((overlap_start - start_time) / self.time_resolution)
                    end_frame = int((overlap_end - start_time) / self.time_resolution)
                
                    # Add the pitch information weighted by the overlap duration
                    pitch_matrix[start_frame:end_frame, note.pitch] += overlap_duration
                    
                    # Pitch annotation
                    pitch_idx = note.pitch - self.start_pitch
                    if 0 <= pitch_idx and pitch_idx < self.num_pitch_classes:
                        pitch_annotation[pitch_idx] = 1
    
    
        # Binarize the pitch matrix (1 if present, 0 if absent)
        pitch_matrix = (pitch_matrix > 0).astype(np.float32)

        # Reduce to a smaller pitch range (e.g., first 52 pitches)
        pitch_matrix = pitch_matrix[:, self.start_pitch:self.start_pitch + 52]
        
        # Determine the single pitch label using the specified strategy
        pitch_label = self.get_segment_pitch_label(pitch_matrix, strategy=self.strategy)
        
        return pitch_label, pitch_annotation
    
    
    def get_segment_pitch_label(self, pitch_matrix, strategy='mode'):
        """
        Determine the ground truth pitch label for a segment based on the pitch matrix.

        Args:
            pitch_matrix (np.ndarray): A binary matrix of shape (num_time_steps, num_pitches).
                                    Each element indicates the presence of a pitch at a time step.
            strategy (str): Strategy to determine the pitch label. Options are:
                            'mode' (most frequent pitch), 'mean', 'median'.

        Returns:
            pitch_label (np.ndarray): A one-hot encoded pitch label of shape (52,).
        """
        num_pitches = pitch_matrix.shape[1]

        if strategy == 'mode':
            # Sum across time steps to find the most frequently occurring pitch
            pitch_counts = np.sum(pitch_matrix, axis=0)
            # Determine the pitch with the maximum count (most frequent pitch)
            pitch_index = np.argmax(pitch_counts)
        elif strategy == 'mean':
            # Compute a weighted average of the pitch indices based on their occurrence
            pitch_indices = np.arange(num_pitches)
            pitch_counts = np.sum(pitch_matrix, axis=0)
            pitch_index = int(np.dot(pitch_counts, pitch_indices) / np.sum(pitch_counts))
        elif strategy == 'median':
            # Flatten the pitch matrix and find the median pitch
            pitches = []
            for pitch_index in range(num_pitches):
                pitches.extend([pitch_index] * int(np.sum(pitch_matrix[:, pitch_index])))
            pitch_index = int(np.median(pitches)) if pitches else 0
        else:
            raise ValueError("Invalid strategy. Use 'mode', 'mean', or 'median'.")

        # Create a one-hot encoded vector of shape (52,)
        pitch_label = np.zeros(num_pitches, dtype=np.float32)
        pitch_label[pitch_index] = 1  # Set the actual pitch to 1
        
        return pitch_label

    

    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Parameters:
        - idx: Index of the sample.
        
        Returns:
        - A dictionary containing the mixture, query, pitch, and timbre.
        """
        # For simplicity, assume that files are named in a structured way to pair mixture and query.
        query_path = self.file_list[idx]
        mixture_path = os.path.join(query_path.split('/stems_audio')[0], "mix.wav")
        
        # Load mel spectrograms with the same segment alignment
        query, start_frame = self._load_audio(query_path)
        mixture, _ = self._load_audio(mixture_path, start_frame=start_frame)
        
        # Convert start_frame to start_time and end_time in seconds
        start_time = (start_frame * self.hop_length) / self.sample_rate 
        end_time = start_time + (self.crop_frames * self.hop_length) / self.sample_rate
        
        # Load pitch annotations for the corresponding segment
        midi_path = query_path.replace(".wav", ".mid").replace("stems_audio", "stems_midi")  # Assuming corresponding MIDI path
        pitch_label, pitch_annotation = self._extract_pitch_annotations(midi_path, start_time, end_time)
        stem = midi_path.split('/')[-1].split('.mid')[0]
        
        return {
            'mixture': mixture,         # Mixture mel spectrogram
            'query': query,             # Query mel spectrogram
            'pitch_label': torch.tensor(pitch_label), # Ground-truth pitch label for the segment
            'pitch_annotation': torch.tensor(pitch_annotation), 
            'stem': stem,
        }


class JSBChoralesDataset:
    def __init__(
        self, 
        chorales_dir="/home/buffett/NAS_189/MusicSlots/data/jsb_multi/", 
        soundfonts, 
        sample_rate=16000
    ):
        """
        Initializes the dataset.
        Args:
            chorales_dir (str): Directory containing the JSB Chorales in MIDI format.
            soundfonts (list): List of paths to the soundfont files (SF2 format) for piano, violin, and flute.
            sample_rate (int): Sampling rate for audio synthesis.
        """
        self.chorales_dir = chorales_dir
        self.soundfonts = soundfonts
        self.sample_rate = sample_rate
        self.chords = self._load_chords()
        self.train_set, self.val_set, self.test_set = self._split_dataset()

    def _load_chords(self):
        """
        Loads chords from the JSB Chorales MIDI files.

        Returns:
            list: A list of chords, each chord is a list of MIDI note numbers.
        """
        chords = []
        # Placeholder for actual MIDI parsing logic
        # For this example, we'll simulate loading chords
        for _ in range(3131):
            chord = random.sample(range(60, 72), k=random.randint(2, 5))  # Random notes between C4 and B4
            chords.append(chord)
        return chords

    def _synthesize_note(self, midi_note, soundfont):
        """
        Synthesizes a single note using FluidSynth.

        Args:
            midi_note (int): MIDI note number.
            soundfont (str): Path to the soundfont file.

        Returns:
            np.ndarray: Audio waveform of the note.
        """
        fs = FluidSynth(sound_font=soundfont, sample_rate=self.sample_rate)
        midi_data = f"""\
            MFile 0 1 96
            MTrk
                0 On ch=1 n={midi_note} v=100
                480 Off ch=1 n={midi_note} v=0
                0 Meta TrkEnd
            TrkEnd
        """
        temp_midi = 'temp_note.mid'
        temp_wav = 'temp_note.wav'
        with open(temp_midi, 'w') as f:
            f.write(midi_data)
        fs.midi_to_audio(temp_midi, temp_wav)
        y, _ = librosa.load(temp_wav, sr=self.sample_rate)
        os.remove(temp_midi)
        os.remove(temp_wav)
        return y

    def _synthesize_chord(self, chord):
        """
        Synthesizes a chord by summing the waveforms of individual notes.

        Args:
            chord (list): A list of MIDI note numbers.

        Returns:
            np.ndarray: Audio waveform of the chord.
        """
        note_waveforms = []
        soundfont_indices = np.random.choice(len(self.soundfonts), size=len(chord), replace=True)
        for note, idx in zip(chord, soundfont_indices):
            soundfont = self.soundfonts[idx]
            y = self._synthesize_note(note, soundfont)
            note_waveforms.append(y)
        # Pad waveforms to the same length
        max_length = max(map(len, note_waveforms))
        note_waveforms = [np.pad(y, (0, max_length - len(y))) for y in note_waveforms]
        chord_waveform = np.sum(note_waveforms, axis=0)
        return chord_waveform

    def _convert_to_mel_spectrogram(self, y):
        """
        Converts waveform to mel spectrogram.

        Args:
            y (np.ndarray): Audio waveform.

        Returns:
            np.ndarray: Mel spectrogram.
        """
        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )
        return S

    def _crop_sustain_phase(self, S):
        """
        Crops a 320ms segment (10 frames) from the sustain phase.

        Args:
            S (np.ndarray): Mel spectrogram.

        Returns:
            np.ndarray: Cropped mel spectrogram segment.
        """
        # Assuming sustain phase starts after the attack, we'll take frames from the middle
        num_frames = S.shape[1]
        start_frame = max(0, num_frames // 2 - 5)
        end_frame = start_frame + 10
        S_cropped = S[:, start_frame:end_frame]
        return S_cropped

    def _split_dataset(self):
        """
        Splits the dataset into training, validation, and test sets.

        Returns:
            tuple: Three lists containing the chords for training, validation, and test sets.
        """
        total_samples = len(self.chords)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        train_end = int(0.7 * total_samples)
        val_end = int(0.9 * total_samples)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        train_set = [self.chords[i] for i in train_indices]
        val_set = [self.chords[i] for i in val_indices]
        test_set = [self.chords[i] for i in test_indices]

        return train_set, val_set, test_set

    def _generate_sample(self, chord):
        """
        Generates a data sample from a chord.

        Args:
            chord (list): A list of MIDI note numbers.

        Returns:
            dict: A dictionary containing the mel spectrogram and multi-hot pitch annotation.
        """
        chord_waveform = self._synthesize_chord(chord)
        S = self._convert_to_mel_spectrogram(chord_waveform)
        S_cropped = self._crop_sustain_phase(S)

        # Create multi-hot pitch annotation
        pitch_annotation = np.zeros(128)
        for note in chord:
            pitch_annotation[note] = 1

        return {
            'spectrogram': S_cropped,
            'pitch_annotation': pitch_annotation
        }

    def get_data_loader(self, split='train'):
        """
        Generator that yields data samples for the specified split.

        Args:
            split (str): One of 'train', 'val', or 'test'.

        Yields:
            dict: Data sample containing 'spectrogram' and 'pitch_annotation'.
        """
        if split == 'train':
            dataset = self.train_set
        elif split == 'val':
            dataset = self.val_set
        elif split == 'test':
            dataset = self.test_set
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'.")

        for chord in tqdm(dataset, desc=f"Processing {split} data"):
            sample = self._generate_sample(chord)
            yield sample


# Code from https://github.com/arahosu/MusicSlots/blob/main/dataset.py
def custom_collate_fn(batch):
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
            os.path.join(self.out_path, '{}_examples.pt'.format(split)))
        
        # notes = torch.unique(self.examples)
        # self.notes = notes[notes.nonzero(as_tuple=True)]
        self.notes = self.metadata["note_list"]
        self.instrument_tokens = np.array(self.metadata["instrument_tokens"])
    
    def __getitem__(self, index):
        spec_file = self.spec_list[index]
        spec = torch.load(spec_file)
        instrument_list = self.instrument_list[index]

        if self.to_db:
            spec = AF.amplitude_to_DB(spec, 10, 1e-10, 0, top_db=self.top_db)

        chord = self.examples[index % len(self.examples)]
        note_spec_list = []

        for note, inst in zip(chord[chord.nonzero(as_tuple=True)], instrument_list):
            instrument_name = self.all_instrument_names[inst].replace(" ", "")
            note_spec = torch.load(os.path.join(
                self.root, 'notes/{}'.format(instrument_name), str(note.item()),
                '{}_spec.pt'.format(self.spec)))
            note_spec_list.append(note_spec)

        note_tensors = torch.cat(note_spec_list, dim=0)

        if self.transform is not None:
            spec = self.transform(spec)
            note_tensors = self.transform(note_tensors)

        midi_label = chord2int(chord, self.notes)
        instrument_label = instrument_to_int(instrument_list, self.instrument_tokens)

        return spec, note_tensors, midi_label, instrument_label

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
            collate_fn=custom_collate_fn
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



if __name__ == "__main__":
    # # Usage Example
    # data_dir = '/home/buffett/NAS_189/cocochorales_output/main_dataset/'
    # print(CocoChoralesTinyDataset(data_dir, split='train')[0])

    # roots = ['/home/buffett/NAS_189/MusicSlots/data/jazznet_single',
    #          '/home/buffett/NAS_189/MusicSlots/data/jazznet_multi',
    #          '/home/buffett/NAS_189/MusicSlots/data/jsb_single',
    #          '/home/buffett/NAS_189/MusicSlots/data/jsb_multi']

    roots = ['/home/buffett/NAS_189/MusicSlots/data/jsb_multi']
    
    for root in roots:
        dm = MusicalObjectDataModule(root=root,
                                     batch_size=32)
        
        dm.setup(stage='fit')

        print('Dataset sample count: {}, instrument count: {}, note count: {}'.format(
            dm.num_samples, dm.num_notes, dm.num_instruments
        ))
        
        spec, note_tensors, midi_label, instrument_label = dm.train_ds[0]
        print('Spectrogram shape: {}, note spectrograms shape: {}'.format(spec.shape, note_tensors.shape))
    
 