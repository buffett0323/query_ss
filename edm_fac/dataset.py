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


    def _midi_to_pitch_sequence(self, midi_path: Path, duration: float) -> torch.Tensor:
        """Convert MIDI file to pitch sequence tensor"""
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            n_frames = math.ceil(duration * self.sample_rate / self.hop_length)
            pitch_sequence = np.zeros((n_frames, self.n_notes))

            for instrument in pm.instruments:
                for note in instrument.notes:
                    start_frame = int(note.start * self.sample_rate / self.hop_length)
                    end_frame = int(note.end * self.sample_rate / self.hop_length)

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
        timbre_id, midi_id, wav_path, midi_path = self.file_index[idx]

        # 1. Sample a random offset
        midi_name = self.id_to_midi[midi_id]
        offset = np.random.choice(self.peak_records[midi_name])

        # 2. Load input audio with this offset
        target_gt = self._load_audio(wav_path, offset=offset)
        pitch_info = self._midi_to_pitch_sequence(midi_path, self.duration)

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

    def _get_fast_pitch(self, midi_path: str) -> torch.Tensor:
        """Fast pitch sequence retrieval using cache or direct computation"""
        if self.cache_midi:
            return self.pitch_cache[midi_path]
        else:
            return self._midi_to_pitch_sequence(midi_path, self.duration)

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

    def _midi_to_pitch_sequence(self, midi_path: Path, duration: float) -> torch.Tensor:
        """Convert MIDI file to pitch sequence tensor"""
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            n_frames = math.ceil(duration * self.sample_rate / self.hop_length)
            pitch_sequence = np.zeros((n_frames, self.n_notes))

            for instrument in pm.instruments:
                for note in instrument.notes:
                    start_frame = int(note.start * self.sample_rate / self.hop_length)
                    end_frame = int(note.end * self.sample_rate / self.hop_length)

                    start_frame = max(0, min(start_frame, n_frames-1))
                    end_frame = max(0, min(end_frame, n_frames-1))

                    note_idx = note.pitch - self.min_note

                    if 0 <= note_idx < self.n_notes:
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
        pitch_info = self._get_fast_pitch(midi_path)

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

    train_data = EDM_Simple_Dataset_Optimized(
        root_path=args.root_path,
        midi_path=args.midi_path,
        data_path=args.data_path,
        duration=args.duration,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        min_note=args.min_note,
        max_note=args.max_note,
        stems=args.stems,
        split="train"
    )

    val_data = EDM_Simple_Dataset_Optimized(
        root_path=args.root_path,
        midi_path=args.midi_path,
        data_path=args.data_path,
        duration=args.duration,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        min_note=args.min_note,
        max_note=args.max_note,
        stems=args.stems,
        split="evaluation"
    )
    print(len(train_data))
    print(len(val_data))


    os.makedirs("sample_audio", exist_ok=True)


    for i in range(5):
        data = train_data[i]
        data['target'].write(f"sample_audio/target_{i}.wav")
        data['content_match'].write(f"sample_audio/content_match_{i}.wav")
        # data['timbre_converted'].write(f"sample_audio/timbre_converted_{i}.wav")
        print(data['metadata'])

    train_loader = build_dataloader(train_data, batch_size=4, num_workers=8, prefetch_factor=8, split="train")
    # val_loader = build_dataloader(val_data, batch_size=4, num_workers=0, prefetch_factor=16, split="evaluation")

    for data in tqdm(train_loader):
        pass
