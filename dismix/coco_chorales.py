import os
import yaml
import torchaudio
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram

class CocoChoraleDataset(Dataset):
    def __init__(
        self, 
        N_s = 4,
        file_dir='/home/buffett/NAS_189/cocochorales_full_v1_output/main_dataset',
        split='train',
        transform=None, 
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
            
        return mix_melspec, stems_melspec, pitch_annotation



# Example usage
if __name__ == "__main__":
    file_dir = "/home/buffett/NAS_189/cocochorales_full_v1_output/main_dataset/"

    for split in ["train", "valid", "test"]:
        dataset = CocoChoraleDataset(file_dir, split=split)
        print(len(dataset))
