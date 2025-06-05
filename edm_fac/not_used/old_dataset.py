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

from audiotools import AudioSignal
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import List, Tuple, Dict, Any
from pathlib import Path
from tqdm import tqdm
from utils import yaml_config_hook

class EDM_Render_Dataset(Dataset):
    def __init__(
        self,
        root_path: str,
        midi_path: str,
        data_path: str,
        duration: float = 0.38,
        sample_rate: int = 44100,
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

                    for timbre_selected in self.unique_timbres: # 3rd for-loop: 59 timbre pairs
                        target_wav_path = os.path.join(self.root_path, stem, f"{timbre_selected}_{midi}.wav")

                        # Transfer to class ids
                        midi_id = self.midi_to_id[midi]
                        timbre_id = self.timbre_to_id[timbre]
                        timbre_id_converted = self.timbre_to_id[timbre_selected]
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
                        self.file_index.append((timbre_id, midi_id, wav_path, midi_path, timbre_id_converted, target_wav_path))


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


    def _get_random_match_content(self, curr_idx: int) -> int:
        curr_timbre_id, curr_content_id, _, _, _, _ = self.file_index[curr_idx]
        possible_matches = [
            idx for idx in self.content_to_files[curr_content_id]
                if self.file_index[idx][0] != curr_timbre_id
        ]
        return random.choice(possible_matches)


    def _get_random_match_timbre(self, timbre_id: int, content_id: int) -> int:
        possible_matches = [
            idx for idx in self.timbre_to_files[timbre_id]
                if self.file_index[idx][1] != content_id
        ]
        return random.choice(possible_matches)


    def _midi_to_pitch_sequence(self, midi_path: Path, duration: float) -> torch.Tensor:
        """Convert MIDI file to pitch sequence tensor"""
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            hop_length = 512
            n_frames = math.ceil(duration * self.sample_rate / hop_length)
            pitch_sequence = np.zeros((n_frames, self.n_notes))

            for instrument in pm.instruments:
                for note in instrument.notes:
                    start_frame = int(note.start * self.sample_rate / hop_length)
                    end_frame = int(note.end * self.sample_rate / hop_length)

                    start_frame = max(0, min(start_frame, n_frames-1))
                    end_frame = max(0, min(end_frame, n_frames-1))

                    note_idx = note.pitch - self.min_note

                    if 0 <= note_idx < self.n_notes:
                        pitch_sequence[start_frame:end_frame+1, note_idx] = 1

            return torch.FloatTensor(pitch_sequence)

        except Exception as e:
            print(f"Error processing MIDI file {midi_path}: {e}")
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
        timbre_id, midi_id, wav_path, midi_path, timbre_id_converted, target_wav_path = self.file_index[idx]

        # 1. Compute max offset for this file
        file_duration = AudioSignal(wav_path).duration

        # 2. Sample a random offset
        # assert file_duration >= self.duration, f"File duration {file_duration} is less than duration {self.duration}"
        if self.peak_records:
            offset = np.random.choice(self.peak_records[self.id_to_midi[midi_id]])
        else:
            offset = np.random.uniform(0, file_duration - self.duration)

        assert offset <= file_duration - self.duration, f"Offset {offset} is greater than file duration {file_duration} - duration {self.duration}"

        # 3. Load input audio with this offset
        input_signal = self._load_audio(wav_path, offset=offset)
        input_pitch = self._midi_to_pitch_sequence(midi_path, self.duration)

        # 4. Find matches for content and timbre
        content_match_idx = self._get_random_match_content(idx)
        timbre_converted_idx = self._get_random_match_timbre(timbre_id_converted, midi_id)


        """ Perturbation """
        # 1). Load content match
        content_match = self._load_audio(self.file_index[content_match_idx][2], offset=offset)
        # content_pitch = self._midi_to_pitch_sequence(self.file_index[content_match_idx][3], self.duration) # Actually the same as input_pitch -> no need

        # 2). Load timbre match
        timbre_converted = self._load_audio(self.file_index[timbre_converted_idx][2], offset=offset)
        # timbre_pitch = self._midi_to_pitch_sequence(self.file_index[timbre_match_idx][3], self.duration) # No need


        # 5. Load target gt
        target_gt = self._load_audio(target_wav_path, offset=offset)

        # assert input_pitch.shape == content_pitch.shape, f"Shape mismatch: input_pitch {input_pitch.shape} != content_pitch {content_pitch.shape}"
        # assert input_pitch.shape == timbre_pitch.shape, f"Shape mismatch: input_pitch {input_pitch.shape} != timbre_pitch {timbre_pitch.shape}"
        # assert torch.allclose(input_pitch, content_pitch), "Pitch tensors are not exactly equal"

        return {
            'input': input_signal,
            'pitch': input_pitch,
            'timbre_id': timbre_id,
            'content_match': content_match,
            # 'content_pitch': content_pitch,
            'timbre_converted': timbre_converted,
            # 'timbre_pitch': timbre_pitch,
            'target_gt': target_gt,
            'metadata': {
                'input': {
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
                },
                'target_info':{
                    'path': str(target_wav_path)
                }
            }
        }

    @staticmethod
    def collate(batch: List[Dict]) -> Dict:
        """Custom collate function for batching"""
        return {
            'input': AudioSignal.batch([item['input'] for item in batch]),
            'pitch': torch.stack([item['pitch'] for item in batch]),
            'timbre_id': torch.tensor([item['timbre_id'] for item in batch]),
            'content_match': AudioSignal.batch([item['content_match'] for item in batch]),
            # 'content_pitch': torch.stack([item['content_pitch'] for item in batch]),
            'timbre_converted': AudioSignal.batch([item['timbre_converted'] for item in batch]),
            # 'timbre_pitch': torch.stack([item['timbre_pitch'] for item in batch]),
            'target_gt': AudioSignal.batch([item['target_gt'] for item in batch]),
            'metadata': [item['metadata'] for item in batch]
        }

