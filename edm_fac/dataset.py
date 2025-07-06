import os
import json
import random
import math
import numpy as np
import torch
import soundfile as sf
import pretty_midi
import argparse
import librosa

from audiotools import AudioSignal
from audiotools import transforms as tfm
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from utils import yaml_config_hook
import audiotools
print("Audiotools imported from", audiotools.__file__)


# Single Shot Dataset
class EDM_Single_Shot_Dataset(Dataset):
    """
    Use RMS Curve to analyze the ADSR
    """
    def __init__(
        self,
        root_path: str,
        duration: float = 1.0,
        sample_rate: int = 44100,
        hop_length: int = 512,
        perturb_prob: float = 0.5,
        n_notes: int = 21,
        split: str = "train",
    ):
        self.root_path = Path(os.path.join(root_path, split))
        self.duration = duration
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_notes = n_notes
        self.split = split
        self.perturb_prob = perturb_prob
        
        
        # Data
        with open(f'{self.root_path}/metadata.json', 'r') as f:
            self.metadata = json.load(f)["files"]    
        
        # Get envelopes metadata
        with open(f'info/envelopes_{split}_new.json', 'r') as f:
            envelopes = json.load(f)
            
        self.envelopes = {}
        for envelope in envelopes:
            self.envelopes[envelope['id']] = {
                "attack": envelope['attack'],
                'decay': envelope['decay'],
                'hold': envelope['hold'],
                'sustain': envelope['sustain'],
                'release': envelope['release'],
                'length': envelope['length'],
            }
        

        # Create ID mappings for all three levels
        self.timbre_to_files = {}
        self.content_to_files = {}
        self.adsr_to_files = {}
        
        # Storing names
        self.paired_data = []
        self._build_paired_index()
        


    def _build_paired_index(self):
        
        # Shuffle metadata
        random.shuffle(self.metadata)

        # Build paired index
        counter = 0
        for data in tqdm(self.metadata, desc="Building paired index"):
            wav_path = data['file_path']
            timbre_id = data['timbre_index']
            adsr_id = data['adsr_index']
            midi_id = data['note_index']
            
            if timbre_id not in self.timbre_to_files:
                self.timbre_to_files[timbre_id] = []
            self.timbre_to_files[timbre_id].append(counter)

            if midi_id not in self.content_to_files:
                self.content_to_files[midi_id] = []
            self.content_to_files[midi_id].append(counter)

            if adsr_id not in self.adsr_to_files:
                self.adsr_to_files[adsr_id] = []
            self.adsr_to_files[adsr_id].append(counter)
            
            self.paired_data.append((timbre_id, midi_id, adsr_id, wav_path))
            counter += 1
            

    def _load_audio(self, file_path: Path) -> AudioSignal:
        signal, _ = sf.read(
            file_path,
            start=0,
            frames=int(self.duration*self.sample_rate)
        )
        return AudioSignal(signal, self.sample_rate)

    
    
    def _get_random_match_content(self, timbre_id: int, content_id: int, adsr_id: int) -> int:
        if self.split == "train":
            return random.choice(self.content_to_files[content_id])

        else:
            possible_matches = [
                idx for idx in self.content_to_files[content_id]
                    if self.paired_data[idx][0] != timbre_id and \
                        self.paired_data[idx][2] != adsr_id
            ]
            return random.choice(possible_matches)


    def _get_random_match_timbre(self, timbre_id: int, content_id: int, adsr_id: int) -> int:
        if self.split == "train":
            return random.choice(self.timbre_to_files[timbre_id])

        else:
            possible_matches = [
                idx for idx in self.timbre_to_files[timbre_id]
                    if self.paired_data[idx][1] != content_id and \
                        self.paired_data[idx][2] != adsr_id
            ]
            return random.choice(possible_matches)


    def _get_random_match_content_adsr(self, timbre_id: int, content_id: int, adsr_id: int) -> int:
        possible_matches = [
            idx for idx in range(len(self.paired_data))
                if self.paired_data[idx][0] != timbre_id and \
                    self.paired_data[idx][1] == content_id and \
                    self.paired_data[idx][2] == adsr_id
        ]
        return random.choice(possible_matches)



    def _get_random_match_adsr(self, timbre_id: int, content_id: int, adsr_id: int) -> int:
        if self.split == "train":
            return random.choice(self.adsr_to_files[adsr_id])

        else:
            possible_matches = [
                idx for idx in self.adsr_to_files[adsr_id]
                    if self.paired_data[idx][0] != timbre_id and \
                        self.paired_data[idx][1] != content_id
            ]
            return random.choice(possible_matches)


    def _get_midi_from_adsr(self, midi_id: int, adsr_id: int) -> torch.Tensor:
        n_frames = math.ceil(self.duration * self.sample_rate / self.hop_length)
        pitch_sequence = np.zeros((n_frames, self.n_notes)) # Frames, Notes
        
        note_length = math.ceil(self.envelopes[adsr_id]["length"] * n_frames * 0.001 / self.duration)
        for i in range(note_length):
            pitch_sequence[i, midi_id] = 1
            
        return torch.FloatTensor(pitch_sequence)



    def __len__(self):
        return len(self.paired_data)
    
    
    def __getitem__(self, idx):
        # 0. Target: T1, ADSR1, C1
        timbre_id, midi_id, adsr_id, wav_path = self.paired_data[idx]

        # 1. Load input audio
        target_gt = self._load_audio(wav_path)

        # 2. Load timbre match
        if random.random() <= self.perturb_prob:
            perturb = True
            timbre_match_idx = self._get_random_match_timbre(timbre_id, midi_id, adsr_id)
            adsr_match_idx = self._get_random_match_adsr(timbre_id, midi_id, adsr_id)
            content_match_idx = self._get_random_match_content(timbre_id, midi_id, adsr_id)
        else:
            perturb = False
            timbre_match_idx = idx
            adsr_match_idx = idx
            content_match_idx = idx
        
        timbre_match = self._load_audio(self.paired_data[timbre_match_idx][3])
        adsr_match = self._load_audio(self.paired_data[adsr_match_idx][3])
        content_match = self._load_audio(self.paired_data[content_match_idx][3])
        
        # Pitch sequence
        pitch = self._get_midi_from_adsr(midi_id, adsr_id)
        
        return {
            'target': target_gt,
            
            'timbre_id': timbre_id,
            'adsr_id': adsr_id,
            'midi_id': midi_id,
            'pitch': pitch,
            
            'timbre_match': timbre_match,
            'adsr_match': adsr_match,
            'content_match': content_match,

            'metadata': {
                'perturb': perturb,
                'target': {
                    'timbre_id': timbre_id,
                    'content_id': midi_id,
                    'adsr_id': adsr_id,
                    'path': str(wav_path),
                },
                'content_match': {
                    'timbre_id': self.paired_data[content_match_idx][0],
                    'content_id': self.paired_data[content_match_idx][1],
                    'adsr_id': self.paired_data[content_match_idx][2],
                    'path': str(self.paired_data[content_match_idx][3])
                },
                'timbre_match': {
                    'timbre_id': self.paired_data[timbre_match_idx][0],
                    'content_id': self.paired_data[timbre_match_idx][1],
                    'adsr_id': self.paired_data[timbre_match_idx][2],
                    'path': str(self.paired_data[timbre_match_idx][3])
                },
                'adsr_match':{
                    'timbre_id': self.paired_data[adsr_match_idx][0],
                    'content_id': self.paired_data[adsr_match_idx][1],
                    'adsr_id': self.paired_data[adsr_match_idx][2],
                    'path': str(self.paired_data[adsr_match_idx][3])
                }
            }
        }
        
    @staticmethod
    def collate(batch: List[Dict]) -> Dict:
        return {
            'target': AudioSignal.batch([item['target'] for item in batch]),
            'timbre_id': torch.tensor([item['timbre_id'] for item in batch], dtype=torch.long),
            'adsr_id': torch.tensor([item['adsr_id'] for item in batch], dtype=torch.long),
            'midi_id': torch.tensor([item['midi_id'] for item in batch], dtype=torch.long),
            'pitch': torch.stack([item['pitch'] for item in batch]),
            'content_match': AudioSignal.batch([item['content_match'] for item in batch]),
            'timbre_match': AudioSignal.batch([item['timbre_match'] for item in batch]),
            'adsr_match': AudioSignal.batch([item['adsr_match'] for item in batch]),
            'metadata': [item['metadata'] for item in batch]
        }



class EDM_Single_Shot_Val_Dataset(Dataset):
    """
    Use RMS Curve to analyze the ADSR
    """
    def __init__(
        self,
        root_path: str,
        duration: float = 1.0,
        sample_rate: int = 44100,
        hop_length: int = 512,
        split: str = "evaluation",
        n_notes: int = 21,
    ):
        self.root_path = Path(os.path.join(root_path, split))
        self.duration = duration
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.split = split
        self.n_notes = n_notes
        
        
        # Data
        with open(f'{self.root_path}/metadata.json', 'r') as f:
            self.metadata = json.load(f)["files"]
        
        # Get envelopes metadata
        with open(f'info/envelopes_{split}_new.json', 'r') as f:
            envelopes = json.load(f)
            
        self.envelopes = {}
        for envelope in envelopes:
            self.envelopes[envelope['id']] = {
                "attack": envelope['attack'],
                'decay': envelope['decay'],
                'hold': envelope['hold'],
                'sustain': envelope['sustain'],
                'release': envelope['release'],
                'length': envelope['length'],
            }

        # Create ID mappings for all three levels
        self.timbre_to_files = {}
        self.content_to_files = {}
        self.adsr_to_files = {}
        self.ids_to_item_idx = {}
        
        # Storing names
        self.paired_data = []
        self._build_paired_index()
        


    def _build_paired_index(self):
        
        # Shuffle metadata
        random.shuffle(self.metadata)

        # Build paired index
        counter = 0
        for data in tqdm(self.metadata, desc="Building paired index"):
            wav_path = data['file_path']
            timbre_id = data['timbre_index']
            adsr_id = data['adsr_index']
            midi_id = data['note_index']
            
            if timbre_id not in self.timbre_to_files:
                self.timbre_to_files[timbre_id] = []
            self.timbre_to_files[timbre_id].append(counter)

            if midi_id not in self.content_to_files:
                self.content_to_files[midi_id] = []
            self.content_to_files[midi_id].append(counter)

            if adsr_id not in self.adsr_to_files:
                self.adsr_to_files[adsr_id] = []
            self.adsr_to_files[adsr_id].append(counter)
            
            self.paired_data.append((timbre_id, midi_id, adsr_id, wav_path))
            self.ids_to_item_idx[f"T{timbre_id:03d}_ADSR{adsr_id:03d}_C{midi_id:03d}"] = counter
            counter += 1
            

    def _load_audio(self, file_path: Path) -> AudioSignal:
        signal, _ = sf.read(
            file_path,
            start=0,
            frames=int(self.duration*self.sample_rate)
        )
        return AudioSignal(signal, self.sample_rate)

    
    
    def _get_random_match_total(self, timbre_id: int, content_id: int, adsr_id: int) -> int:
        possible_matches = [
            idx for idx in range(len(self.paired_data))
                if self.paired_data[idx][0] != timbre_id and \
                    self.paired_data[idx][1] != content_id and \
                    self.paired_data[idx][2] != adsr_id
        ]
        return random.choice(possible_matches)


    def _get_midi_from_adsr(self, midi_id: int, adsr_id: int) -> torch.Tensor:
        try:
            n_frames = math.ceil(self.duration * self.sample_rate / self.hop_length)
            pitch_sequence = np.zeros((n_frames, self.n_notes)) # Frames, Notes
            
            note_length = math.ceil(self.envelopes[adsr_id]["length"] * n_frames/ self.duration)
            for i in range(note_length):
                pitch_sequence[i, midi_id] = 1
                
            return torch.FloatTensor(pitch_sequence)

        except Exception as e:
            print("Error processing MIDI file")
            return torch.zeros((n_frames, self.n_notes))
        

    def __len__(self):
        return len(self.paired_data)
    
    
    def __getitem__(self, idx):
        # 1. Original: T5, C6, ADSR5
        timbre_id, midi_id, adsr_id, wav_path = self.paired_data[idx]
        orig_audio = self._load_audio(wav_path)

        # 2. Reference: T3, C7, ADSR8
        ref_idx = self._get_random_match_total(timbre_id, midi_id, adsr_id)
        ref_timbre_id, ref_midi_id, ref_adsr_id, ref_wav_path = self.paired_data[ref_idx]
        ref_audio = self._load_audio(ref_wav_path)
        
        
        # 3. Get target content, timbre, and adsr transfer ground truth
        """
        Target_content: T5, C7, ADSR5
        Target_timbre: T3, C6, ADSR5
        Target_adsr: T5, C6, ADSR8
        """
        target_content_idx = self.ids_to_item_idx[f"T{timbre_id:03d}_ADSR{adsr_id:03d}_C{ref_midi_id:03d}"]
        target_timbre_idx = self.ids_to_item_idx[f"T{ref_timbre_id:03d}_ADSR{adsr_id:03d}_C{midi_id:03d}"]
        target_adsr_idx = self.ids_to_item_idx[f"T{timbre_id:03d}_ADSR{ref_adsr_id:03d}_C{midi_id:03d}"]

        target_content = self._load_audio(self.paired_data[target_content_idx][3])
        target_timbre = self._load_audio(self.paired_data[target_timbre_idx][3])
        target_adsr = self._load_audio(self.paired_data[target_adsr_idx][3])
        
        # 4. Get target pitch
        orig_pitch = self._get_midi_from_adsr(midi_id, adsr_id)
        ref_pitch = self._get_midi_from_adsr(ref_midi_id, ref_adsr_id)
        
        return {
            'orig_audio': orig_audio,
            'ref_audio': ref_audio,

            'orig_midi': midi_id,
            'ref_midi': ref_midi_id,
            'orig_timbre': timbre_id,
            'ref_timbre': ref_timbre_id,
            'orig_adsr': adsr_id,
            'ref_adsr': ref_adsr_id,
            
            'orig_pitch': orig_pitch,
            'ref_pitch': ref_pitch,

            'target_content': target_content,
            'target_timbre': target_timbre,
            'target_adsr': target_adsr,

            'metadata': {
                'orig_audio': {
                    'timbre_id': timbre_id,
                    'content_id': midi_id,
                    'adsr_id': adsr_id,
                    'path': str(wav_path)
                },
                'ref_audio': {
                    'timbre_id': ref_timbre_id,
                    'content_id': ref_midi_id,
                    'adsr_id': ref_adsr_id,
                    'path': str(ref_wav_path),
                },
                'target_content': {
                    'timbre_id': self.paired_data[target_content_idx][0],
                    'content_id': self.paired_data[target_content_idx][1],
                    'adsr_id': self.paired_data[target_content_idx][2],
                    'path': str(self.paired_data[target_content_idx][3]),
                },
                'target_timbre': {
                    'timbre_id': self.paired_data[target_timbre_idx][0],
                    'content_id': self.paired_data[target_timbre_idx][1],
                    'adsr_id': self.paired_data[target_timbre_idx][2],
                    'path': str(self.paired_data[target_timbre_idx][3]),
                },
                'target_adsr': {
                    'timbre_id': self.paired_data[target_adsr_idx][0],
                    'content_id': self.paired_data[target_adsr_idx][1],
                    'adsr_id': self.paired_data[target_adsr_idx][2],
                    'path': str(self.paired_data[target_adsr_idx][3]),
                },
            }
        }

    @staticmethod
    def collate(batch: List[Dict]) -> Dict:
        """Custom collate function for batching"""
        return {
            'orig_audio': AudioSignal.batch([item['orig_audio'] for item in batch]),
            'ref_audio': AudioSignal.batch([item['ref_audio'] for item in batch]),
            'orig_pitch': torch.stack([item['orig_pitch'] for item in batch]),
            'ref_pitch': torch.stack([item['ref_pitch'] for item in batch]),
            
            'orig_midi': torch.tensor([item['orig_midi'] for item in batch], dtype=torch.long),
            'ref_midi': torch.tensor([item['ref_midi'] for item in batch], dtype=torch.long),
            'orig_timbre': torch.tensor([item['orig_timbre'] for item in batch], dtype=torch.long),
            'ref_timbre': torch.tensor([item['ref_timbre'] for item in batch], dtype=torch.long),
            'orig_adsr': torch.tensor([item['orig_adsr'] for item in batch], dtype=torch.long),
            'ref_adsr': torch.tensor([item['ref_adsr'] for item in batch], dtype=torch.long),
            'target_content': AudioSignal.batch([item['target_content'] for item in batch]),
            'target_timbre': AudioSignal.batch([item['target_timbre'] for item in batch]),
            'target_adsr': AudioSignal.batch([item['target_adsr'] for item in batch]),
            'metadata': [item['metadata'] for item in batch]
        }











# Mixed Dataset
class EDM_ADSR_Paired_Dataset(Dataset):
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
        perturb_content: bool = True,
        perturb_timbre: bool = True,
        perturb_adsr: bool = True,
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
        self.perturb_content = perturb_content
        self.perturb_timbre = perturb_timbre
        self.perturb_adsr = perturb_adsr

        # Create ID mappings for all three levels
        self.timbre_to_files = {}
        self.content_to_files = {}
        self.adsr_to_files = {}

        # Storing names
        self._preload_metadata()
        self.paired_data = []  # Rendered data
        self._build_paired_index()
        print("Paired data:", len(self.paired_data))
        # print(self.metadata[0])


    def _preload_metadata(self):
        # Timbre, MIDI, ADSR info
        # with open(f'{self.data_path}/timbres.json', 'r') as f:
        #     self.timbre_metadata = json.load(f)
        with open(f'{self.data_path}/midi_files.json', 'r') as f:
            self.midi_metadata = json.load(f)
        # with open(f'{self.data_path}/adsr_envelopes.json', 'r') as f:
        #     self.adsr_metadata = json.load(f)

        # Metadata
        with open(f'{self.data_path}/dataset_info.json', 'r') as f:
            self.dataset_info = json.load(f)
        with open(f'{self.data_path}/metadata.json', 'r') as f:
            self.metadata = json.load(f)

        # Process MIDI metadata
        self.total_seconds = self.dataset_info['total_seconds']
        thres = self.total_seconds - self.duration

        for _, midi_data in self.midi_metadata.items():
            midi_data['onset_seconds'] = [x for x in midi_data['onset_seconds'] if x <= thres]



    def _build_paired_index(self):

        # Shuffle metadata
        random.shuffle(self.metadata)

        # Render Dataset
        counter = 0
        for data in tqdm(self.metadata, desc="Building paired index"):
            wav_path = data['file_path']

            # Update mapping dictionaries
            timbre_id = data['timbre_index']
            adsr_id = data['adsr_index']
            midi_id = data['midi_index']
            midi_key = data['midi_id']


            if timbre_id not in self.timbre_to_files:
                self.timbre_to_files[timbre_id] = []
            self.timbre_to_files[timbre_id].append(counter)

            if midi_id not in self.content_to_files:
                self.content_to_files[midi_id] = []
            self.content_to_files[midi_id].append(counter)

            if adsr_id not in self.adsr_to_files:
                self.adsr_to_files[adsr_id] = []
            self.adsr_to_files[adsr_id].append(counter)


            midi_path = os.path.join(
                self.midi_path,
                self.midi_metadata[midi_key]['filename']
            )
            self.paired_data.append((timbre_id, midi_id, adsr_id, wav_path, midi_path))
            counter += 1


    def _get_random_match_content(self, timbre_id: int, content_id: int, adsr_id: int) -> int:
        if self.split == "train":
            return random.choice(self.content_to_files[content_id])

        else:
            possible_matches = [
                idx for idx in self.content_to_files[content_id]
                    if self.paired_data[idx][0] != timbre_id and \
                        self.paired_data[idx][2] != adsr_id
            ]
            return random.choice(possible_matches)


    def _get_random_match_timbre(self, timbre_id: int, content_id: int, adsr_id: int) -> int:
        if self.split == "train":
            return random.choice(self.timbre_to_files[timbre_id])

        else:
            possible_matches = [
                idx for idx in self.timbre_to_files[timbre_id]
                    if self.paired_data[idx][1] != content_id and \
                        self.paired_data[idx][2] != adsr_id
            ]
            return random.choice(possible_matches)

    # Timbre has to be different from target, content and adsr has to be different from target
    def _get_random_match_content_adsr(self, timbre_id: int, content_id: int, adsr_id: int) -> int:
        possible_matches = [
            idx for idx in range(len(self.paired_data))
                if self.paired_data[idx][0] != timbre_id and \
                    self.paired_data[idx][1] == content_id and \
                    self.paired_data[idx][2] == adsr_id
        ]
        return random.choice(possible_matches)



    def _get_random_match_adsr(self, timbre_id: int, content_id: int, adsr_id: int) -> int:
        if self.split == "train":
            return random.choice(self.adsr_to_files[adsr_id])

        else:
            possible_matches = [
                idx for idx in self.adsr_to_files[adsr_id]
                    if self.paired_data[idx][0] != timbre_id and \
                        self.paired_data[idx][1] != content_id
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
        # signal = signal.mean(axis=1, keepdims=False)
        return AudioSignal(signal, self.sample_rate)


    def __len__(self) -> int:
        return len(self.paired_data) # 255000


    def __getitem__(self, idx: int) -> Dict:
        timbre_id, midi_id, adsr_id, wav_path, midi_path = self.paired_data[idx]

        # 1. Sample a random offset
        offset = np.random.choice(self.midi_metadata[f"C{midi_id:03d}"]['onset_seconds'])

        # 2. Load input audio with this offset
        target_gt = self._load_audio(wav_path, offset=offset)
        pitch_info = self._midi_to_pitch_sequence(midi_path, self.duration, offset)

        # 3. Find matches for content and timbre
        content_match_idx = self._get_random_match_content_adsr(timbre_id, midi_id, adsr_id) # self._get_random_match_content(timbre_id, midi_id, adsr_id)
        timbre_match_idx = self._get_random_match_timbre(timbre_id, midi_id, adsr_id)
        adsr_match_idx = content_match_idx #self._get_random_match_adsr(timbre_id, midi_id, adsr_id)

        """ Perturbation """
        # 1). Load content match
        if self.perturb_content:
            content_match = self._load_audio(self.paired_data[content_match_idx][3], offset=offset)
        else:
            content_match = target_gt
        # content_pitch = self._midi_to_pitch_sequence(self.paired_data[content_match_idx][3], self.duration) # Actually the same as input_pitch -> no need

        # 2). Load timbre match
        if self.perturb_timbre:
            timbre_match = self._load_audio(self.paired_data[timbre_match_idx][3], offset=offset)
        else:
            timbre_match = target_gt
        # timbre_pitch = self._midi_to_pitch_sequence(self.paired_data[timbre_match_idx][3], self.duration) # No need

        # 3). Load adsr match
        if self.perturb_adsr:
            adsr_match = self._load_audio(self.paired_data[adsr_match_idx][3], offset=offset)
        else:
            adsr_match = target_gt

        # # Validation use for conversion
        # if self.split == "evaluation":
        #     orig_audio_input = self._get_random_match_content(timbre_id, midi_id)

        return {
            'target': target_gt,
            'pitch': pitch_info,

            'timbre_id': timbre_id,
            'adsr_id': adsr_id,

            'content_match': content_match,
            'timbre_match': timbre_match,
            'adsr_match': adsr_match,

            'metadata': {
                'target': {
                    'timbre_id': timbre_id,
                    'content_id': midi_id,
                    'adsr_id': adsr_id,
                    'path': str(wav_path),
                    'offset': offset
                },
                'content_match': {
                    'timbre_id': self.paired_data[content_match_idx][0],
                    'content_id': self.paired_data[content_match_idx][1],
                    'adsr_id': self.paired_data[content_match_idx][2],
                    'path': str(self.paired_data[content_match_idx][3])
                },
                'timbre_match': {
                    'timbre_id': self.paired_data[timbre_match_idx][0],
                    'content_id': self.paired_data[timbre_match_idx][1],
                    'adsr_id': self.paired_data[timbre_match_idx][2],
                    'path': str(self.paired_data[timbre_match_idx][3])
                },
                'adsr_match':{
                    'timbre_id': self.paired_data[adsr_match_idx][0],
                    'content_id': self.paired_data[adsr_match_idx][1],
                    'adsr_id': self.paired_data[adsr_match_idx][2],
                    'path': str(self.paired_data[adsr_match_idx][3])
                }
            }
        }

    @staticmethod
    def collate(batch: List[Dict]) -> Dict:
        """Custom collate function for batching"""
        return {
            'target': AudioSignal.batch([item['target'] for item in batch]),
            'pitch': torch.stack([item['pitch'] for item in batch]),
            'timbre_id': torch.tensor([item['timbre_id'] for item in batch], dtype=torch.long),
            'adsr_id': torch.tensor([item['adsr_id'] for item in batch], dtype=torch.long),
            'content_match': AudioSignal.batch([item['content_match'] for item in batch]),
            'timbre_match': AudioSignal.batch([item['timbre_match'] for item in batch]),
            'adsr_match': AudioSignal.batch([item['adsr_match'] for item in batch]),
            'metadata': [item['metadata'] for item in batch]
        }



class EDM_ADSR_Val_Paired_Dataset(Dataset):
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
        split: str = "evaluation",
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
        self.timbre_to_files = {}
        self.content_to_files = {}
        self.adsr_to_files = {}
        self.ids_to_item_idx = {}

        # Storing names
        self._preload_metadata()
        self.paired_data = []  # Rendered data
        self._build_paired_index()
        print("Paired data:", len(self.paired_data))


    def _preload_metadata(self):
        # Timbre, MIDI, ADSR info
        # with open(f'{self.data_path}/timbres.json', 'r') as f:
        #     self.timbre_metadata = json.load(f)
        with open(f'{self.data_path}/midi_files.json', 'r') as f:
            self.midi_metadata = json.load(f)
        # with open(f'{self.data_path}/adsr_envelopes.json', 'r') as f:
        #     self.adsr_metadata = json.load(f)

        # Metadata
        with open(f'{self.data_path}/dataset_info.json', 'r') as f:
            self.dataset_info = json.load(f)
        with open(f'{self.data_path}/metadata.json', 'r') as f:
            self.metadata = json.load(f)

        # Process MIDI metadata
        self.total_seconds = self.dataset_info['total_seconds']
        thres = self.total_seconds - self.duration

        for _, midi_data in self.midi_metadata.items():
            midi_data['onset_seconds'] = [x for x in midi_data['onset_seconds'] if x <= thres]



    def _build_paired_index(self):

        # Shuffle metadata
        random.shuffle(self.metadata)

        # Render Dataset
        counter = 0
        for data in tqdm(self.metadata, desc="Building paired index"):
            wav_path = data['file_path']

            # Update mapping dictionaries
            timbre_id = data['timbre_index']
            adsr_id = data['adsr_index']
            midi_id = data['midi_index']
            midi_key = data['midi_id']


            if timbre_id not in self.timbre_to_files:
                self.timbre_to_files[timbre_id] = []
            self.timbre_to_files[timbre_id].append(counter)

            if midi_id not in self.content_to_files:
                self.content_to_files[midi_id] = []
            self.content_to_files[midi_id].append(counter)

            if adsr_id not in self.adsr_to_files:
                self.adsr_to_files[adsr_id] = []
            self.adsr_to_files[adsr_id].append(counter)


            midi_path = os.path.join(
                self.midi_path,
                self.midi_metadata[midi_key]['filename']
            )
            self.paired_data.append((timbre_id, midi_id, adsr_id, wav_path, midi_path))
            self.ids_to_item_idx[f"T{timbre_id:03d}_ADSR{adsr_id:03d}_C{midi_id:03d}"] = counter
            counter += 1




    def _get_random_match_total(self, timbre_id: int, content_id: int, adsr_id: int) -> int:
        possible_matches = [
            idx for idx in range(len(self.paired_data))
                if self.paired_data[idx][0] != timbre_id and \
                    self.paired_data[idx][1] != content_id and \
                    self.paired_data[idx][2] != adsr_id
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
        # signal = signal.mean(axis=1, keepdims=False)
        return AudioSignal(signal, self.sample_rate)


    def __len__(self) -> int:
        return len(self.paired_data) # 255000


    def __getitem__(self, idx: int) -> Dict:
        # 1. Original: T5, C6, ADSR5
        timbre_id, midi_id, adsr_id, wav_path, midi_path = self.paired_data[idx]
        offset = np.random.choice(self.midi_metadata[f"C{midi_id:03d}"]['onset_seconds'])
        orig_audio = self._load_audio(wav_path, offset=offset)

        # 2. Reference: T3, C7, ADSR8
        ref_idx = self._get_random_match_total(timbre_id, midi_id, adsr_id)
        ref_timbre_id, ref_midi_id, ref_adsr_id, ref_wav_path, ref_midi_path = self.paired_data[ref_idx]
        ref_audio = self._load_audio(ref_wav_path, offset=offset)

        # 3. Get target content, timbre, and adsr transfer ground truth
        """
        Target_content: T5, C7, ADSR5
        Target_timbre: T3, C6, ADSR5
        Target_adsr: T5, C6, ADSR8
        """
        target_content_idx = self.ids_to_item_idx[f"T{timbre_id:03d}_ADSR{adsr_id:03d}_C{ref_midi_id:03d}"]
        target_timbre_idx = self.ids_to_item_idx[f"T{ref_timbre_id:03d}_ADSR{adsr_id:03d}_C{midi_id:03d}"]
        target_adsr_idx = self.ids_to_item_idx[f"T{timbre_id:03d}_ADSR{ref_adsr_id:03d}_C{midi_id:03d}"]

        target_content = self._load_audio(self.paired_data[target_content_idx][3], offset=offset)
        target_timbre = self._load_audio(self.paired_data[target_timbre_idx][3], offset=offset)
        target_adsr = self._load_audio(self.paired_data[target_adsr_idx][3], offset=offset)


        # 4. Load pitch info
        orig_pitch = self._midi_to_pitch_sequence(midi_path, self.duration, offset)
        ref_pitch = self._midi_to_pitch_sequence(ref_midi_path, self.duration, offset)


        return {
            'orig_audio': orig_audio,
            'ref_audio': ref_audio,

            'orig_pitch': orig_pitch,
            'ref_pitch': ref_pitch,
            'orig_timbre': timbre_id,
            'ref_timbre': ref_timbre_id,
            'orig_adsr': adsr_id,
            'ref_adsr': ref_adsr_id,

            'target_content': target_content,
            'target_timbre': target_timbre,
            'target_adsr': target_adsr,

            'metadata': {
                'orig_audio': {
                    'timbre_id': timbre_id,
                    'content_id': midi_id,
                    'adsr_id': adsr_id,
                    'path': str(wav_path),
                    'offset': offset
                },
                'ref_audio': {
                    'timbre_id': ref_timbre_id,
                    'content_id': ref_midi_id,
                    'adsr_id': ref_adsr_id,
                    'path': str(ref_wav_path),
                },
                'target_content': {
                    'timbre_id': self.paired_data[target_content_idx][0],
                    'content_id': self.paired_data[target_content_idx][1],
                    'adsr_id': self.paired_data[target_content_idx][2],
                    'path': str(self.paired_data[target_content_idx][3]),
                },
                'target_timbre': {
                    'timbre_id': self.paired_data[target_timbre_idx][0],
                    'content_id': self.paired_data[target_timbre_idx][1],
                    'adsr_id': self.paired_data[target_timbre_idx][2],
                    'path': str(self.paired_data[target_timbre_idx][3]),
                },
                'target_adsr': {
                    'timbre_id': self.paired_data[target_adsr_idx][0],
                    'content_id': self.paired_data[target_adsr_idx][1],
                    'adsr_id': self.paired_data[target_adsr_idx][2],
                    'path': str(self.paired_data[target_adsr_idx][3]),
                },
            }
        }

    @staticmethod
    def collate(batch: List[Dict]) -> Dict:
        """Custom collate function for batching"""
        return {
            'orig_audio': AudioSignal.batch([item['orig_audio'] for item in batch]),
            'ref_audio': AudioSignal.batch([item['ref_audio'] for item in batch]),
            'orig_pitch': torch.stack([item['orig_pitch'] for item in batch]),
            'ref_pitch': torch.stack([item['ref_pitch'] for item in batch]),
            'orig_timbre': torch.tensor([item['orig_timbre'] for item in batch], dtype=torch.long),
            'ref_timbre': torch.tensor([item['ref_timbre'] for item in batch], dtype=torch.long),
            'orig_adsr': torch.tensor([item['orig_adsr'] for item in batch], dtype=torch.long),
            'ref_adsr': torch.tensor([item['ref_adsr'] for item in batch], dtype=torch.long),
            'target_content': AudioSignal.batch([item['target_content'] for item in batch]),
            'target_timbre': AudioSignal.batch([item['target_timbre'] for item in batch]),
            'target_adsr': AudioSignal.batch([item['target_adsr'] for item in batch]),
            'metadata': [item['metadata'] for item in batch]
        }



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

    config = yaml_config_hook("configs/config_ss.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    train_paired_data = EDM_Single_Shot_Dataset(
        root_path=args.root_path,
        duration=args.duration,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        perturb_prob=args.perturb_prob,
        split="train",
        n_notes=args.n_notes,
    )
    dt = train_paired_data[0]
    print(dt['pitch'].shape)
    print(json.dumps(dt['metadata'], indent=2))

    val_paired_data = EDM_Single_Shot_Val_Dataset(
        root_path=args.root_path,
        duration=args.duration,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        split="evaluation",
        n_notes=args.n_notes,
    )
    dt = val_paired_data[0]
    print(dt['orig_pitch'].shape)
    print(dt['ref_pitch'].shape)
    print(json.dumps(dt['metadata'], indent=2))
    