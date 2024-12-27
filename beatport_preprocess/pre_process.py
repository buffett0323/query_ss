import os
import allin1.analyze
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
np.int = int
np.float = float
import allin1
import warnings
from pydub import AudioSegment


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    # message="You're calling NATTEN op `natten.functional.natten2dqkrpb`, which is deprecated",
)

device = torch.device('cuda:2')
sep_model_name = 'htdemucs'
sources = ['bass', 'drums', 'other', 'vocals']


def segment_audio(song_name, segments, sep_path, output_path, target='chorus'):

    chorus_counter = 1
    for segment in segments:
        if segment.label == target:
            start_ms = int(segment.start * 1000)  # Convert seconds to milliseconds
            end_ms = int(segment.end * 1000)     # Convert seconds to milliseconds
            
            # Extract the segment for the sources
            os.makedirs(os.path.join(output_path, target, f"{song_name}_{chorus_counter}"), exist_ok=True)
            for source in sources:
                audio = AudioSegment.from_file(os.path.join(sep_path, f"{source}.wav"))
                seg_audio = audio[start_ms:end_ms]
                
                # Export the segment
                output_file = os.path.join(output_path, target, f"{song_name}_{chorus_counter}", f"{source}_{target}_{chorus_counter}.mp3")
                seg_audio.export(output_file, format="mp3")                
            
            chorus_counter += 1
            
            
            

def load_data_and_process(input_path, output_path, target='chorus', sep_model_name='htdemucs'):
    # Open folders
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, target), exist_ok=True)
    os.makedirs(os.path.join(output_path, sep_model_name), exist_ok=True)
    os.makedirs(os.path.join(output_path, "json"), exist_ok=True)
    
    # Iterate through each folder in the given path
    for folder_name in tqdm(os.listdir(input_path)):
        folder_path = os.path.join(input_path, folder_name)
        
        if os.path.isdir(folder_path):
            # TODO: Pre-process and also store in metadata
            audio_files = [
                os.path.join(folder_path, file_name)
                for file_name in os.listdir(folder_path)
                if file_name.endswith(('.wav', '.mp3'))
            ]
            
            # Analyze by allin1
            results = allin1.analyze(
                audio_files[:5],
                out_dir=os.path.join(output_path, "json"),
                demix_dir=output_path, 
                spec_dir=output_path,
                device=device, 
                keep_byproducts=True
            )
            
            
            # Load audio files
            for audio_path, result in zip(audio_files[:5], results[:5]):
                song_name = audio_path.split('/')[-1].split('.mp3')[0]

                # Segment audio to get chorus
                segment_audio(
                    song_name=song_name, 
                    segments=result.segments, 
                    sep_path=os.path.join(output_path, sep_model_name, song_name), 
                    output_path=output_path, 
                    target=target
                )
            
    


if __name__ == "__main__":
    input_path = "/mnt/gestalt/database/beatport/audio/audio"
    output_path = "/mnt/gestalt/home/ddmanddman/beatport_preprocess"
    target='chorus'
    load_data_and_process(input_path, output_path, target)
