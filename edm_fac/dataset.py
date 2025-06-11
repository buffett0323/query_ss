import os
import json
import random
import math
import numpy as np
import torch
import soundfile as sf
import pretty_midi
import argparse
import itertools
import librosa

from audiotools import AudioSignal
from audiotools import transforms as tfm
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import List, Tuple, Dict, Any
from pathlib import Path
from tqdm import tqdm
from utils import yaml_config_hook
import audiotools
print("Audiotools imported from", audiotools.__file__)


# Mixed Dataset
class EDM_Paired_Dataset(Dataset):
    def __init__(
        self,
        root_path: str,
        midi_path: str,
        data_path: str,
        duration: float = 1.0,
        sample_rate: int = 44100,
        hop_length: int = 512,
        min_note: int = 21,
        max_note: int = 108,
        split: str = "train",
    ):
        midi_path = os.path.join(midi_path, split, "midi")
        self.root_path = Path(os.path.join(root_path, split))
        self.midi_path = Path(midi_path)
        self.data_path = Path(data_path)
        self.duration = duration
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.min_note = min_note
        self.max_note = max_note
        self.n_notes = max_note - min_note + 1
        self.split = split


        # Create ID mappings for all three levels
        self.timbre_to_id = {}
        self.id_to_timbre = {}
        self.midi_to_id = {}
        self.id_to_midi = {}
        self.timbre_to_files = {}
        self.content_to_files = {}

        # Storing names
        self.unique_timbres = [] # timbre names
        self.unique_midis = [] # midi names
        self._preprocess()

        # Create index mappings
        self.paired_data = []  # Rendered data
        self._build_paired_index()
        self._shuffle_paired_index()
        self.peak_records = {}
        self._get_offset_pos()


    def _preprocess(self):

        with open(f'info/timbre_names_mixed.txt', 'r') as f:
            for line in f:
                self.unique_timbres.append(line.strip())

        with open(f'info/midi_names_mixed_{self.split}.txt', 'r') as f:
            for line in f:
                self.unique_midis.append(line.strip())

        # Create sequential mappings
        for label, timbre in enumerate(sorted(self.unique_timbres)):
            self.timbre_to_id[timbre] = label
            self.id_to_timbre[label] = timbre

        for label, midi in enumerate(sorted(self.unique_midis)):
            self.midi_to_id[midi] = label
            self.id_to_midi[label] = midi



    def _build_paired_index(self):

        # Rendered Dataset
        for timbre in self.unique_timbres: # 1st for-loop: 59/510 timbres
            for midi in self.unique_midis: # 2nd for-loop: 100/500 midis
                wav_path = os.path.join(self.root_path, f"{timbre}_{midi}.wav")

                # Transfer to class ids
                midi_id = self.midi_to_id[midi]
                timbre_id = self.timbre_to_id[timbre]
                counter = len(self.paired_data)

                if not os.path.exists(wav_path):
                    continue

                # Update mapping dictionaries
                if timbre_id not in self.timbre_to_files:
                    self.timbre_to_files[timbre_id] = []
                self.timbre_to_files[timbre_id].append(counter)

                if midi_id not in self.content_to_files:
                    self.content_to_files[midi_id] = []
                self.content_to_files[midi_id].append(counter)

                midi_path = os.path.join(self.midi_path, f"{midi}.mid")
                self.paired_data.append((timbre_id, midi_id, wav_path, midi_path))


        # # Beatport Dataset
        # for file in tqdm(os.listdir(self.beatport_path), desc=f"Pre-loading {self.split} beatport dataset"):
        #     wav_path = os.path.join(self.beatport_path, file)
        #     self.unpaired_data.append((-1, -1, wav_path, None))

        print(f"-> Got {len(self.paired_data)} paired tracks for {self.split}")


    def _shuffle_paired_index(self):
        """Shuffle the file index while maintaining matching relationships"""
        indices = list(range(len(self.paired_data)))
        random.shuffle(indices)

        # Create a mapping from old to new indices
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
        self.paired_data = [self.paired_data[i] for i in indices]


        for content_id in self.content_to_files:
            self.content_to_files[content_id] = [index_map[idx] for idx in self.content_to_files[content_id]]

        for timbre_id in self.timbre_to_files:
            self.timbre_to_files[timbre_id] = [index_map[idx] for idx in self.timbre_to_files[timbre_id]]


    def _get_offset_pos(self):
        with open(f"{self.data_path}/json/onset_records_mixed_{self.split}.json", "r") as f:
            self.peak_records = json.load(f)

        print(f"-> Got {len(self.peak_records) * len(self.unique_timbres)} different tracks for {self.split}")



    def _get_random_match_content(self, timbre_id: int, content_id: int) -> int:
        if self.split == "train":
            return random.choice(self.content_to_files[content_id])

        else:
            possible_matches = [
                idx for idx in self.content_to_files[content_id]
                    if self.paired_data[idx][0] != timbre_id
            ]
            return random.choice(possible_matches)


    def _get_random_match_timbre(self, timbre_id: int, content_id: int) -> int:
        if self.split == "train":
            return random.choice(self.timbre_to_files[timbre_id])

        else:
            possible_matches = [
                idx for idx in self.timbre_to_files[timbre_id]
                    if self.paired_data[idx][1] != content_id
            ]
            return random.choice(possible_matches)


    def _midi_to_pitch_sequence(self, midi_path: Path, duration: float, offset: float = 0.0) -> torch.Tensor:
        """Convert MIDI file to pitch sequence tensor starting from offset"""
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            n_frames = math.ceil(duration * self.sample_rate / self.hop_length)
            pitch_sequence = np.zeros((n_frames, self.n_notes))

            for instrument in pm.instruments:
                for note in instrument.notes:
                    # Adjust note timing by subtracting offset
                    note_start = note.start - offset
                    note_end = note.end - offset

                    # Skip notes that end before our offset window starts
                    if note_end <= 0:
                        continue

                    # Skip notes that start after our duration window ends
                    if note_start >= duration:
                        continue

                    # Clamp note timing to our duration window
                    note_start = max(0, note_start)
                    note_end = min(duration, note_end)

                    start_frame = int(note_start * self.sample_rate / self.hop_length)
                    end_frame = int(note_end * self.sample_rate / self.hop_length)

                    start_frame = max(0, min(start_frame, n_frames-1))
                    end_frame = max(0, min(end_frame, n_frames-1))

                    note_idx = note.pitch - self.min_note

                    if 0 <= note_idx < self.n_notes and start_frame <= end_frame:
                        pitch_sequence[start_frame:end_frame+1, note_idx] = 1

            return torch.FloatTensor(pitch_sequence)

        except Exception as e:
            print(f"Error processing MIDI file {midi_path}: {e}")
            n_frames = math.ceil(duration * self.sample_rate / self.hop_length)
            return torch.zeros((n_frames, self.n_notes))


    def _load_audio(self, file_path: Path, offset: float = 0.0) -> AudioSignal:
        signal, _ = sf.read(
            file_path,
            start=int(offset*self.sample_rate),
            frames=int(self.duration*self.sample_rate)
        )
        signal = signal.mean(axis=1, keepdims=False)
        return AudioSignal(signal, self.sample_rate)



    def extract_edm_timbre_features(self, waveform):
        # Extract numpy array from AudioSignal object
        if hasattr(waveform, 'audio_data'):
            audio_data = waveform.audio_data.squeeze().cpu().numpy()
        elif hasattr(waveform, 'samples'):
            audio_data = waveform.samples.squeeze().cpu().numpy()
        else:
            # Assume it's already a numpy array or tensor
            if isinstance(waveform, torch.Tensor):
                audio_data = waveform.squeeze().cpu().numpy()
            else:
                audio_data = waveform

        # Brightness (frame-level)
        brightness = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate, hop_length=self.hop_length)[0]
        brightness = torch.from_numpy(brightness).float()

        # Modulation (frame-level)
        modulation = librosa.onset.onset_strength(y=audio_data, sr=self.sample_rate, hop_length=self.hop_length)
        modulation = torch.from_numpy(modulation).float()

        # FX tail decay (global scalar)
        rms = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=self.hop_length)[0]
        tail_len = int(len(rms) * 0.2)
        tail_rms = rms[-tail_len:]
        x = np.arange(tail_len)
        slope = np.polyfit(x, np.log(tail_rms + 1e-7), 1)[0]
        fx_tail = torch.tensor(slope).float()

        return {
            "brightness": brightness,        # shape: (n_frames,)
            "modulation": modulation,        # shape: (n_frames,)
            "fx_tail": fx_tail               # shape: scalar
        }



    def __len__(self) -> int:
        return len(self.paired_data) # 255000


    def __getitem__(self, idx: int) -> Dict:
        timbre_id, midi_id, wav_path, midi_path = self.paired_data[idx]

        # 1. Sample a random offset
        midi_name = self.id_to_midi[midi_id]
        offset = np.random.choice(self.peak_records[midi_name])

        # 2. Load input audio with this offset
        target_gt = self._load_audio(wav_path, offset=offset)
        pitch_info = self._midi_to_pitch_sequence(midi_path, self.duration, offset)

        # 3. Find matches for content and timbre
        content_match_idx = self._get_random_match_content(timbre_id, midi_id)
        timbre_converted_idx1 = self._get_random_match_timbre(timbre_id, midi_id)
        timbre_converted_idx2 = self._get_random_match_timbre(timbre_id, midi_id)

        """ Perturbation """
        # 1). Load content match
        content_match = self._load_audio(self.paired_data[content_match_idx][2], offset=offset)
        # content_pitch = self._midi_to_pitch_sequence(self.paired_data[content_match_idx][3], self.duration) # Actually the same as input_pitch -> no need

        # 2). Load timbre match
        timbre_converted1 = self._load_audio(self.paired_data[timbre_converted_idx1][2], offset=offset)
        timbre_converted2 = self._load_audio(self.paired_data[timbre_converted_idx2][2], offset=offset)
        # timbre_pitch = self._midi_to_pitch_sequence(self.paired_data[timbre_match_idx][3], self.duration) # No need

        # # Validation use for conversion
        # if self.split == "evaluation":
        #     orig_audio_input = self._get_random_match_content(timbre_id, midi_id)


        # Timbre features
        timbre_features = self.extract_edm_timbre_features(target_gt)

        return {
            'target': target_gt,
            'pitch': pitch_info,
            'timbre_id': timbre_id,
            'content_match': content_match,
            'timbre_converted': timbre_converted1,
            'timbre_pair': timbre_converted2,
            'timbre_brightness': timbre_features['brightness'],
            'timbre_modulation': timbre_features['modulation'],
            'timbre_fx_tail': timbre_features['fx_tail'],
            'metadata': {
                'target': {
                    'timbre_id': timbre_id,
                    'content_id': midi_id,
                    'path': str(wav_path),
                    'offset': offset
                },
                'content_match': {
                    'timbre_id': self.paired_data[content_match_idx][0],
                    'content_id': self.paired_data[content_match_idx][1],
                    'path': str(self.paired_data[content_match_idx][2])
                },
                'timbre_converted': {
                    'timbre_id': self.paired_data[timbre_converted_idx1][0],
                    'content_id': self.paired_data[timbre_converted_idx1][1],
                    'path': str(self.paired_data[timbre_converted_idx1][2])
                },
                'timbre_pair':{
                    'timbre_id': self.paired_data[timbre_converted_idx2][0],
                    'content_id': self.paired_data[timbre_converted_idx2][1],
                    'path': str(self.paired_data[timbre_converted_idx2][2])
                }
            }
        }

    @staticmethod
    def collate(batch: List[Dict]) -> Dict:
        """Custom collate function for batching"""
        return {
            'target': AudioSignal.batch([item['target'] for item in batch]),
            'pitch': torch.stack([item['pitch'] for item in batch]),
            'timbre_id': torch.tensor([item['timbre_id'] for item in batch]),
            'content_match': AudioSignal.batch([item['content_match'] for item in batch]),
            'timbre_converted': AudioSignal.batch([item['timbre_converted'] for item in batch]),
            'timbre_pair': AudioSignal.batch([item['timbre_pair'] for item in batch]),
            'timbre_brightness': torch.stack([item['timbre_brightness'] for item in batch]),
            'timbre_modulation': torch.stack([item['timbre_modulation'] for item in batch]),
            'timbre_fx_tail': torch.stack([item['timbre_fx_tail'] for item in batch]),
            'metadata': [item['metadata'] for item in batch]
        }



class EDM_Unpaired_Dataset(Dataset):
    def __init__(
        self,
        beatport_path: str,
        data_path: str,
        duration: float = 1.0,
        total_duration: float = 5.0,
        sample_rate: int = 44100,
        split: str = "train",
    ):
        self.beatport_path = Path(os.path.join(beatport_path, split))
        self.data_path = Path(data_path)
        self.duration = duration
        self.total_duration = total_duration
        self.sample_rate = sample_rate
        self.split = split


        # Create index mappings
        self.unpaired_data = [
            os.path.join(self.beatport_path, file)
                for file in os.listdir(self.beatport_path)
        ]
        print(f"-> Got {len(self.unpaired_data)} unpaired tracks for {self.split}")

        # Load peak records
        self.peak_records = {}
        with open(f"{self.data_path}/json/beatport_peak_records_{self.split}.json", "r") as f:
            self.peak_records = json.load(f)


        # Augmentation
        self.cont_aug = tfm.Compose(
            [
                tfm.LowPass(prob=0.5),
                tfm.HighPass(prob=0.5),
                tfm.ClippingDistortion(perc=("uniform", 0.0, 0.8)),
            ],
            name="content_augmentation",
            prob=0.9,
        )

        self.tim_aug = tfm.Compose(
            [
                tfm.SeqPerturbReverse(
                    num_segments=5,
                    fixed_second=0.3,
                    reverse_prob=("uniform", 0.5, 1),
                ),
                tfm.PitchShift(
                    n_semitones=("uniform", -2, 2),
                    quick=True,
                )
            ],
            name="timbre_augmentation",
            prob=0.9,
        )

    def _load_audio(self, file_path: Path, offset: float = 0.0) -> AudioSignal:
        signal, _ = sf.read(
            file_path,
            start=int(offset*self.sample_rate),
            frames=int(self.duration*self.sample_rate)
        )
        return AudioSignal(signal, self.sample_rate)


    def __len__(self) -> int:
        return len(self.unpaired_data) # 168139


    def content_augment(self, audio: AudioSignal) -> AudioSignal:
        """
        Low Pass --> High Pass --> Clipping Distortion
        """
        rn_state = random.randint(0, 1000)
        kwargs = self.cont_aug.instantiate(state=rn_state, signal=audio) # Instantiate random parameters
        output = self.cont_aug(audio.clone(), **kwargs) # Apply transform
        return output



    def timbre_augment(self, audio: AudioSignal) -> AudioSignal:
        """
        SeqPerturbReverse --> Pitch Shift
        """
        rn_state = random.randint(0, 1000)
        kwargs = self.tim_aug.instantiate(state=rn_state)
        output = self.tim_aug(audio.clone(), **kwargs)
        return output


    def __getitem__(self, idx: int) -> Dict:
        wav_path = self.unpaired_data[idx]

        # 1. Sample a random offset
        wav_name = wav_path.split("/")[-1]
        offset = np.random.choice(self.peak_records[wav_name])

        # 2. Load input audio with this offset
        target_gt = self._load_audio(wav_path, offset=offset)

        # 3. Perturbation of augmentation
        content_match = self.content_augment(target_gt.clone())
        timbre_converted = self.timbre_augment(target_gt.clone())

        return {
            'target': target_gt,
            'content_match': content_match,
            'timbre_converted': timbre_converted,
        }


    @staticmethod
    def collate(batch: List[AudioSignal]) -> Dict:
        """Custom collate function for batching"""
        return {
            'target': AudioSignal.batch([item['target'] for item in batch]),
            'content_match': AudioSignal.batch([item['content_match'] for item in batch]),
            'timbre_converted': AudioSignal.batch([item['timbre_converted'] for item in batch]),
        }



# For Training
class EDM_Simple_Dataset(Dataset):
    def __init__(
        self,
        root_path: str,
        midi_path: str,
        data_path: str,
        duration: float = 0.38,
        sample_rate: int = 44100,
        hop_length: int = 512,
        min_note: int = 21,
        max_note: int = 108,
        stems: list[str] = ["lead"], #["lead", "pad", "bass", "keys", "pluck"],
        split: str = "train",
    ):
        midi_path = os.path.join(midi_path, split, "midi")
        self.root_path = Path(root_path)
        self.midi_path = Path(midi_path)
        self.data_path = Path(data_path)
        self.duration = duration
        self.sample_rate = sample_rate
        self.stems = stems
        self.hop_length = hop_length
        self.min_note = min_note
        self.max_note = max_note
        self.n_notes = max_note - min_note + 1
        self.split = split

        # Create ID mappings for all three levels
        self.timbre_to_id = {}
        self.id_to_timbre = {}
        self.midi_to_id = {}
        self.id_to_midi = {}
        self.timbre_to_files = {}
        self.content_to_files = {}

        # Storing names
        self.unique_timbres = [] # timbre names
        self.unique_midis = [] # midi names
        self._preprocess()

        # Create index mappings
        self.file_index = []  # List of (timbre_id, content_id, full_path)
        self._build_index()
        self._shuffle_file_index()
        self.peak_records = {}
        self._get_offset_pos()


    def _get_offset_pos(self):
        with open(f"{self.data_path}/onset_records_{self.stems[0]}_{self.split}.json", "r") as f:
            self.peak_records = json.load(f)

        print(f"-> Got {len(self.peak_records) * len(self.unique_timbres)} different tracks for {self.split}")



    def _preprocess(self):
        """Build mappings for timbre, DI, and tone IDs"""
        with open(f'info/timbre_names_{self.stems[0]}.txt', 'r') as f:
            for line in f:
                self.unique_timbres.append(line.strip())

        with open(f'info/{self.split}_midi_names.txt', 'r') as f:
            for line in f:
                self.unique_midis.append(line.strip())

        # Create sequential mappings
        for idx, timbre in enumerate(sorted(self.unique_timbres)):
            self.timbre_to_id[timbre] = idx
            self.id_to_timbre[idx] = timbre

        for idx, midi in enumerate(sorted(self.unique_midis)):
            self.midi_to_id[midi] = idx
            self.id_to_midi[idx] = midi


    def _build_index(self):

        for stem in tqdm(self.stems, desc=f"Pre-loading {self.split} info"):

            for timbre in self.unique_timbres: # 1st for-loop: 59 timbres
                for midi in self.unique_midis: # 2nd for-loop: 100 midis
                    wav_path = os.path.join(self.root_path, stem, f"{timbre}_{midi}.wav")

                    # Transfer to class ids
                    midi_id = self.midi_to_id[midi]
                    timbre_id = self.timbre_to_id[timbre]
                    counter = len(self.file_index)

                    if not os.path.exists(wav_path): continue

                    # Update mapping dictionaries
                    if timbre_id not in self.timbre_to_files:
                        self.timbre_to_files[timbre_id] = []
                    self.timbre_to_files[timbre_id].append(counter)

                    if midi_id not in self.content_to_files:
                        self.content_to_files[midi_id] = []
                    self.content_to_files[midi_id].append(counter)

                    midi_path = os.path.join(self.midi_path, f"{midi}.mid")
                    self.file_index.append((timbre_id, midi_id, wav_path, midi_path))


    def _shuffle_file_index(self):
        """Shuffle the file index while maintaining matching relationships"""
        indices = list(range(len(self.file_index)))
        random.shuffle(indices)

        # Create a mapping from old to new indices
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
        self.file_index = [self.file_index[i] for i in indices]


        for content_id in self.content_to_files:
            self.content_to_files[content_id] = [index_map[idx] for idx in self.content_to_files[content_id]]

        for timbre_id in self.timbre_to_files:
            self.timbre_to_files[timbre_id] = [index_map[idx] for idx in self.timbre_to_files[timbre_id]]


    def _get_random_match_content(self, timbre_id: int, content_id: int) -> int:
        if self.split == "train":
            return random.choice(self.content_to_files[content_id])

        else:
            possible_matches = [
                idx for idx in self.content_to_files[content_id]
                    if self.file_index[idx][0] != timbre_id
            ]
            return random.choice(possible_matches)


    def _get_random_match_timbre(self, timbre_id: int, content_id: int) -> int:
        if self.split == "train":
            return random.choice(self.timbre_to_files[timbre_id])

        else:
            possible_matches = [
                idx for idx in self.timbre_to_files[timbre_id]
                    if self.file_index[idx][1] != content_id
            ]
            return random.choice(possible_matches)


    def _midi_to_pitch_sequence(self, midi_path: Path, duration: float, offset: float = 0.0) -> torch.Tensor:
        """Convert MIDI file to pitch sequence tensor starting from offset"""
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            n_frames = math.ceil(duration * self.sample_rate / self.hop_length)
            pitch_sequence = np.zeros((n_frames, self.n_notes))

            for instrument in pm.instruments:
                for note in instrument.notes:
                    # Adjust note timing by subtracting offset
                    note_start = note.start - offset
                    note_end = note.end - offset

                    # Skip notes that end before our offset window starts
                    if note_end <= 0:
                        continue

                    # Skip notes that start after our duration window ends
                    if note_start >= duration:
                        continue

                    # Clamp note timing to our duration window
                    note_start = max(0, note_start)
                    note_end = min(duration, note_end)

                    start_frame = int(note_start * self.sample_rate / self.hop_length)
                    end_frame = int(note_end * self.sample_rate / self.hop_length)

                    start_frame = max(0, min(start_frame, n_frames-1))
                    end_frame = max(0, min(end_frame, n_frames-1))

                    note_idx = note.pitch - self.min_note

                    if 0 <= note_idx < self.n_notes and start_frame <= end_frame:
                        pitch_sequence[start_frame:end_frame+1, note_idx] = 1

            return torch.FloatTensor(pitch_sequence)

        except Exception as e:
            print(f"Error processing MIDI file {midi_path}: {e}")
            n_frames = math.ceil(duration * self.sample_rate / self.hop_length)
            return torch.zeros((n_frames, self.n_notes))


    def _load_audio(self, file_path: Path, offset: float = 0.0) -> AudioSignal:
        signal, _ = sf.read(
            file_path,
            start=int(offset*self.sample_rate),
            frames=int(self.duration*self.sample_rate)
        )
        signal = signal.mean(axis=1, keepdims=False)
        return AudioSignal(signal, self.sample_rate)


    def __len__(self) -> int:
        return len(self.file_index)


    def __getitem__(self, idx: int) -> Dict:
        timbre_id, midi_id, wav_path, midi_path = self.file_index[idx]

        # 1. Sample a random offset
        midi_name = self.id_to_midi[midi_id]
        offset = np.random.choice(self.peak_records[midi_name])

        # 2. Load input audio with this offset
        target_gt = self._load_audio(wav_path, offset=offset)
        pitch_info = self._midi_to_pitch_sequence(midi_path, self.duration, offset)

        # 3. Find matches for content and timbre
        content_match_idx = self._get_random_match_content(timbre_id, midi_id)
        timbre_converted_idx = self._get_random_match_timbre(timbre_id, midi_id)


        """ Perturbation """
        # 1). Load content match
        content_match = self._load_audio(self.file_index[content_match_idx][2], offset=offset)
        # content_pitch = self._midi_to_pitch_sequence(self.file_index[content_match_idx][3], self.duration) # Actually the same as input_pitch -> no need

        # 2). Load timbre match
        timbre_converted = self._load_audio(self.file_index[timbre_converted_idx][2], offset=offset)
        # timbre_pitch = self._midi_to_pitch_sequence(self.file_index[timbre_match_idx][3], self.duration) # No need

        # # Validation use for conversion
        # if self.split == "evaluation":
        #     orig_audio_input = self._get_random_match_content(timbre_id, midi_id)


        return {
            # "input": orig_audio_input,
            'target': target_gt,
            'pitch': pitch_info,
            'timbre_id': timbre_id,
            'content_match': content_match,
            # 'content_pitch': content_pitch,
            'timbre_converted': timbre_converted,
            # 'timbre_pitch': timbre_pitch,
            'metadata': {
                'target': {
                    'timbre_id': timbre_id,
                    'content_id': midi_id,
                    'path': str(wav_path)
                },
                'content_match': {
                    'timbre_id': self.file_index[content_match_idx][0],
                    'content_id': self.file_index[content_match_idx][1],
                    'path': str(self.file_index[content_match_idx][2])
                },
                'timbre_converted': {
                    'timbre_id': self.file_index[timbre_converted_idx][0],
                    'content_id': self.file_index[timbre_converted_idx][1],
                    'path': str(self.file_index[timbre_converted_idx][2])
                }
            }
        }

    @staticmethod
    def collate(batch: List[Dict]) -> Dict:
        """Custom collate function for batching"""
        return {
            # 'input': AudioSignal.batch([item['input'] for item in batch]),
            'target': AudioSignal.batch([item['target'] for item in batch]),
            'pitch': torch.stack([item['pitch'] for item in batch]),
            'timbre_id': torch.tensor([item['timbre_id'] for item in batch]),
            'content_match': AudioSignal.batch([item['content_match'] for item in batch]),
            # 'content_pitch': torch.stack([item['content_pitch'] for item in batch]),
            'timbre_converted': AudioSignal.batch([item['timbre_converted'] for item in batch]),
            # 'timbre_pitch': torch.stack([item['timbre_pitch'] for item in batch]),
            'metadata': [item['metadata'] for item in batch]
        }



# Optimized loading for training
class EDM_Simple_Dataset_Optimized(Dataset):
    """
    Key changes:
    1. Audio Caching (cache_audio = True)
    2. MIDI Caching (cache_midi = True)
    3. Match Pre-computation (precompute_matches=True)
    4. Offset Caching
    5. Fast Accessor Methods

    """
    def __init__(
        self,
        root_path: str,
        midi_path: str,
        data_path: str,
        duration: float = 0.38,
        sample_rate: int = 44100,
        hop_length: int = 512,
        min_note: int = 21,
        max_note: int = 108,
        stems: list[str] = ["lead"],
        split: str = "train",
        cache_audio: bool = True,  # Whether to pre-load audio into memory
        cache_midi: bool = True,   # Whether to pre-compute pitch sequences
        precompute_matches: bool = True,  # Whether to pre-compute random matches
    ):
        midi_path = os.path.join(midi_path, split, "midi")
        self.root_path = Path(root_path)
        self.midi_path = Path(midi_path)
        self.data_path = Path(data_path)
        self.duration = duration
        self.sample_rate = sample_rate
        self.stems = stems
        self.hop_length = hop_length
        self.min_note = min_note
        self.max_note = max_note
        self.n_notes = max_note - min_note + 1
        self.split = split

        # Caching options
        self.cache_audio = cache_audio
        self.cache_midi = cache_midi
        self.precompute_matches = precompute_matches

        # Create ID mappings for all three levels
        self.timbre_to_id = {}
        self.id_to_timbre = {}
        self.midi_to_id = {}
        self.id_to_midi = {}
        self.timbre_to_files = {}
        self.content_to_files = {}

        # Storing names
        self.unique_timbres = []
        self.unique_midis = []
        self._preprocess()

        # Create index mappings
        self.file_index = []
        self._build_index()
        self._shuffle_file_index()

        # Load peak records
        self.peak_records = {}
        self._get_offset_pos()

        # Initialize caches
        self.audio_cache = {} if cache_audio else None
        self.pitch_cache = {} if cache_midi else None
        self.match_cache = {} if precompute_matches else None

        # Pre-load data if caching is enabled
        if cache_audio:
            self._preload_audio()
        if cache_midi:
            self._preload_pitch_sequences()
        if precompute_matches:
            self._precompute_random_matches()

    def _preload_audio(self):
        """Pre-load all audio files into memory"""
        print(f"Pre-loading {len(self.file_index)} audio files into memory...")
        for idx, (timbre_id, midi_id, wav_path, midi_path) in enumerate(tqdm(self.file_index, desc="Loading audio")):
            # Load full audio file
            try:
                signal, _ = sf.read(wav_path)
                if len(signal.shape) > 1:
                    signal = signal.mean(axis=1)
                self.audio_cache[idx] = signal
            except Exception as e:
                print(f"Error loading {wav_path}: {e}")
                # Create dummy audio
                self.audio_cache[idx] = np.zeros(int(8.0 * self.sample_rate))

    def _preload_pitch_sequences(self):
        """Pre-compute all MIDI to pitch sequences"""
        print(f"Pre-computing pitch sequences for {len(set([midi_path for _, _, _, midi_path in self.file_index]))} MIDI files...")
        unique_midis = set()
        for _, midi_id, _, midi_path in self.file_index:
            if midi_path not in unique_midis:
                unique_midis.add(midi_path)
                try:
                    pitch_seq = self._midi_to_pitch_sequence(midi_path, self.duration)
                    self.pitch_cache[midi_path] = pitch_seq
                except Exception as e:
                    print(f"Error processing MIDI {midi_path}: {e}")
                    n_frames = math.ceil(self.duration * self.sample_rate / self.hop_length)
                    self.pitch_cache[midi_path] = torch.zeros((n_frames, self.n_notes))

    def _precompute_random_matches(self):
        """Pre-compute random matches to avoid runtime computation"""
        print("Pre-computing random matches...")
        self.match_cache = {
            'content_matches': {},
            'timbre_matches': {}
        }

        # Pre-compute multiple matches for each sample
        matches_per_sample = 10  # Number of pre-computed matches per sample

        for idx in tqdm(range(len(self.file_index)), desc="Computing matches"):
            timbre_id, midi_id, _, _ = self.file_index[idx]

            # Pre-compute content matches
            content_matches = []
            for _ in range(matches_per_sample):
                if self.split == "train":
                    match_idx = random.choice(self.content_to_files[midi_id])
                else:
                    possible_matches = [
                        i for i in self.content_to_files[midi_id]
                        if self.file_index[i][0] != timbre_id
                    ]
                    match_idx = random.choice(possible_matches) if possible_matches else idx
                content_matches.append(match_idx)
            self.match_cache['content_matches'][idx] = content_matches

            # Pre-compute timbre matches
            timbre_matches = []
            for _ in range(matches_per_sample):
                if self.split == "train":
                    match_idx = random.choice(self.timbre_to_files[timbre_id])
                else:
                    possible_matches = [
                        i for i in self.timbre_to_files[timbre_id]
                        if self.file_index[i][1] != midi_id
                    ]
                    match_idx = random.choice(possible_matches) if possible_matches else idx
                timbre_matches.append(match_idx)
            self.match_cache['timbre_matches'][idx] = timbre_matches


    def _get_offset_pos(self):
        with open(f"{self.data_path}/onset_records_{self.stems[0]}_{self.split}.json", "r") as f:
            self.peak_records = json.load(f)
        print(f"-> Got {len(self.peak_records) * len(self.unique_timbres)} different tracks for {self.split}")

        # Pre-compute offsets for faster access
        if hasattr(self, 'unique_midis'):
            self.offset_cache = {}
            for midi_name in self.unique_midis:
                if midi_name in self.peak_records and len(self.peak_records[midi_name]) > 0:
                    # Pre-generate multiple random offsets
                    self.offset_cache[midi_name] = np.random.choice(
                        self.peak_records[midi_name],
                        size=min(100, len(self.peak_records[midi_name]) * 10),
                        replace=True
                    )
                else:
                    self.offset_cache[midi_name] = [0.0]

    def _preprocess(self):
        """Build mappings for timbre, DI, and tone IDs"""
        with open(f'info/timbre_names_{self.stems[0]}.txt', 'r') as f:
            for line in f:
                self.unique_timbres.append(line.strip())

        with open(f'info/{self.split}_midi_names.txt', 'r') as f:
            for line in f:
                self.unique_midis.append(line.strip())

        # Create sequential mappings
        for idx, timbre in enumerate(sorted(self.unique_timbres)):
            self.timbre_to_id[timbre] = idx
            self.id_to_timbre[idx] = timbre

        for idx, midi in enumerate(sorted(self.unique_midis)):
            self.midi_to_id[midi] = idx
            self.id_to_midi[idx] = midi

    def _build_index(self):
        for stem in tqdm(self.stems, desc=f"Pre-loading {self.split} info"):
            for timbre in self.unique_timbres:
                for midi in self.unique_midis:
                    wav_path = os.path.join(self.root_path, stem, f"{timbre}_{midi}.wav")

                    midi_id = self.midi_to_id[midi]
                    timbre_id = self.timbre_to_id[timbre]
                    counter = len(self.file_index)

                    if not os.path.exists(wav_path):
                        continue

                    if timbre_id not in self.timbre_to_files:
                        self.timbre_to_files[timbre_id] = []
                    self.timbre_to_files[timbre_id].append(counter)

                    if midi_id not in self.content_to_files:
                        self.content_to_files[midi_id] = []
                    self.content_to_files[midi_id].append(counter)

                    midi_path = os.path.join(self.midi_path, f"{midi}.mid")
                    self.file_index.append((timbre_id, midi_id, wav_path, midi_path))

    def _shuffle_file_index(self):
        """Shuffle the file index while maintaining matching relationships"""
        indices = list(range(len(self.file_index)))
        random.shuffle(indices)

        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
        self.file_index = [self.file_index[i] for i in indices]

        for content_id in self.content_to_files:
            self.content_to_files[content_id] = [index_map[idx] for idx in self.content_to_files[content_id]]

        for timbre_id in self.timbre_to_files:
            self.timbre_to_files[timbre_id] = [index_map[idx] for idx in self.timbre_to_files[timbre_id]]

    def _get_fast_audio(self, idx: int, offset: float) -> AudioSignal:
        """Fast audio loading using cache or direct loading"""
        if self.cache_audio:
            # Use cached audio
            full_audio = self.audio_cache[idx]
            start_frame = int(offset * self.sample_rate)
            end_frame = start_frame + int(self.duration * self.sample_rate)
            end_frame = min(end_frame, len(full_audio))

            if end_frame <= start_frame:
                # Create silence if offset is too large
                signal = np.zeros(int(self.duration * self.sample_rate))
            else:
                signal = full_audio[start_frame:end_frame]
                # Pad with zeros if too short
                if len(signal) < int(self.duration * self.sample_rate):
                    signal = np.pad(signal, (0, int(self.duration * self.sample_rate) - len(signal)))

            return AudioSignal(signal, self.sample_rate)
        else:
            # Direct loading (original method)
            _, _, wav_path, _ = self.file_index[idx]
            return self._load_audio(wav_path, offset)

    def _get_fast_pitch(self, midi_path: str, offset: float = 0.0) -> torch.Tensor:
        """Fast pitch sequence retrieval using cache or direct computation"""
        if self.cache_midi and offset == 0.0:
            # Use cached pitch sequence only for zero offset
            return self.pitch_cache[midi_path]
        else:
            # Compute dynamically for non-zero offsets or when not cached
            return self._midi_to_pitch_sequence(midi_path, self.duration, offset)

    def _get_fast_matches(self, idx: int):
        """Fast match retrieval using pre-computed matches"""
        if self.precompute_matches:
            # Use random pre-computed match
            content_match_idx = random.choice(self.match_cache['content_matches'][idx])
            timbre_match_idx = random.choice(self.match_cache['timbre_matches'][idx])
            return content_match_idx, timbre_match_idx
        else:
            # Original method
            timbre_id, midi_id, _, _ = self.file_index[idx]
            content_match_idx = self._get_random_match_content(timbre_id, midi_id)
            timbre_match_idx = self._get_random_match_timbre(timbre_id, midi_id)
            return content_match_idx, timbre_match_idx

    def _get_random_match_content(self, timbre_id: int, content_id: int) -> int:
        if self.split == "train":
            return random.choice(self.content_to_files[content_id])
        else:
            possible_matches = [
                idx for idx in self.content_to_files[content_id]
                    if self.file_index[idx][0] != timbre_id
            ]
            return random.choice(possible_matches)

    def _get_random_match_timbre(self, timbre_id: int, content_id: int) -> int:
        if self.split == "train":
            return random.choice(self.timbre_to_files[timbre_id])
        else:
            possible_matches = [
                idx for idx in self.timbre_to_files[timbre_id]
                    if self.file_index[idx][1] != content_id
            ]
            return random.choice(possible_matches)

    def _midi_to_pitch_sequence(self, midi_path: Path, duration: float, offset: float = 0.0) -> torch.Tensor:
        """Convert MIDI file to pitch sequence tensor starting from offset"""
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            n_frames = math.ceil(duration * self.sample_rate / self.hop_length)
            pitch_sequence = np.zeros((n_frames, self.n_notes))

            for instrument in pm.instruments:
                for note in instrument.notes:
                    # Adjust note timing by subtracting offset
                    note_start = note.start - offset
                    note_end = note.end - offset

                    # Skip notes that end before our offset window starts
                    if note_end <= 0:
                        continue

                    # Skip notes that start after our duration window ends
                    if note_start >= duration:
                        continue

                    # Clamp note timing to our duration window
                    note_start = max(0, note_start)
                    note_end = min(duration, note_end)

                    start_frame = int(note_start * self.sample_rate / self.hop_length)
                    end_frame = int(note_end * self.sample_rate / self.hop_length)

                    start_frame = max(0, min(start_frame, n_frames-1))
                    end_frame = max(0, min(end_frame, n_frames-1))

                    note_idx = note.pitch - self.min_note

                    if 0 <= note_idx < self.n_notes and start_frame <= end_frame:
                        pitch_sequence[start_frame:end_frame+1, note_idx] = 1

            return torch.FloatTensor(pitch_sequence)

        except Exception as e:
            print(f"Error processing MIDI file {midi_path}: {e}")
            n_frames = math.ceil(duration * self.sample_rate / self.hop_length)
            return torch.zeros((n_frames, self.n_notes))

    def _load_audio(self, file_path: Path, offset: float = 0.0) -> AudioSignal:
        signal, _ = sf.read(
            file_path,
            start=int(offset*self.sample_rate),
            frames=int(self.duration*self.sample_rate)
        )
        if len(signal.shape) > 1:
            signal = signal.mean(axis=1)
        return AudioSignal(signal, self.sample_rate)

    def __len__(self) -> int:
        return len(self.file_index)

    def __getitem__(self, idx: int) -> Dict:
        timbre_id, midi_id, wav_path, midi_path = self.file_index[idx]

        # Fast offset selection
        midi_name = self.id_to_midi[midi_id]
        if hasattr(self, 'offset_cache') and midi_name in self.offset_cache:
            offset = np.random.choice(self.offset_cache[midi_name])
        else:
            offset = np.random.choice(self.peak_records[midi_name]) if midi_name in self.peak_records else 0.0

        # Fast audio and pitch loading
        target_gt = self._get_fast_audio(idx, offset)
        pitch_info = self._get_fast_pitch(midi_path, offset)

        # Fast match retrieval
        content_match_idx, timbre_converted_idx = self._get_fast_matches(idx)

        # Load matched audio
        content_match = self._get_fast_audio(content_match_idx, offset)
        timbre_converted = self._get_fast_audio(timbre_converted_idx, offset)

        return {
            'target': target_gt,
            'pitch': pitch_info,
            'timbre_id': timbre_id,
            'content_match': content_match,
            'timbre_converted': timbre_converted,
            'metadata': {
                'target': {
                    'timbre_id': timbre_id,
                    'content_id': midi_id,
                    'path': str(wav_path)
                },
                'content_match': {
                    'timbre_id': self.file_index[content_match_idx][0],
                    'content_id': self.file_index[content_match_idx][1],
                    'path': str(self.file_index[content_match_idx][2])
                },
                'timbre_converted': {
                    'timbre_id': self.file_index[timbre_converted_idx][0],
                    'content_id': self.file_index[timbre_converted_idx][1],
                    'path': str(self.file_index[timbre_converted_idx][2])
                }
            }
        }

    @staticmethod
    def collate(batch: List[Dict]) -> Dict:
        """Custom collate function for batching"""
        return {
            'target': AudioSignal.batch([item['target'] for item in batch]),
            'pitch': torch.stack([item['pitch'] for item in batch]),
            'timbre_id': torch.tensor([item['timbre_id'] for item in batch]),
            'content_match': AudioSignal.batch([item['content_match'] for item in batch]),
            'timbre_converted': AudioSignal.batch([item['timbre_converted'] for item in batch]),
            'metadata': [item['metadata'] for item in batch]
        }



def build_dataloader(
    dataset,
    batch_size=32,
    num_workers=0,
    prefetch_factor=16,
    split="train",
):
    collate_fn = dataset.collate
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        shuffle=True if split == "train" else False,
    )

    return data_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDM-FAC")

    config = yaml_config_hook("configs/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    train_paired_data = EDM_Paired_Dataset(
        root_path=args.root_path,
        midi_path=args.midi_path,
        data_path=args.data_path,
        duration=args.duration,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        min_note=args.min_note,
        max_note=args.max_note,
        split="train",
    )

    dt = train_paired_data[0]
    print(dt['target'].shape)
    print(dt['content_match'].shape)
    print(dt['timbre_converted'].shape)
    print(dt['timbre_pair'].shape)
    print(dt['metadata'])
    # os.makedirs("sample_audio", exist_ok=True)
    # for idx in tqdm(range(1000)):
    #     a2 = train_unpaired_data[idx]
    #     # a2["target"].write(f"sample_audio/target_{idx}.wav")
    #     # a2["content_match"].write(f"sample_audio/content_match_{idx}.wav")
    #     # a2["timbre_converted"].write(f"sample_audio/timbre_converted_{idx}.wav")
    #     if a2['target'].shape[-1] != 44100:
    #         print("target", a2['target'].shape)

    #     if a2['content_match'].shape[-1] != 44100:
    #         print("content_match", a2['content_match'].shape)

    #     if a2['timbre_converted'].shape[-1] != 44100:
    #         print("timbre_converted", a2['timbre_converted'].shape)
