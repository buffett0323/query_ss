import os
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


device = torch.device('cuda:1')

def load_data_and_process(input_path, output_path):

    # Iterate through each folder in the given path
    for folder_name in tqdm(os.listdir(input_path)):
        folder_path = os.path.join(input_path, folder_name)
        os.makedirs(os.path.join(output_path, folder_name), exist_ok=True)
        
        if os.path.isdir(folder_path):
            # TODO: Pre-process
            audio_files = [
                os.path.join(folder_path, file_name)
                for file_name in os.listdir(folder_path)
                if file_name.endswith(('.wav', '.mp3'))
            ]
            
            # Load audio files
            for file_path in tqdm(audio_files):
                song_name = file_path.split('/')[-1].split('.mp3')[0]
                
                # Load the audio file
                audio = AudioSegment.from_file(file_path)
                
                # All in one 
                result = allin1.analyze(file_path, device=device)
                segments = result.segments

                
                chorus_counter = 1
                for segment in segments:
                    # Check if the segment is labeled as 'chorus'
                    if segment.label == 'chorus':
                        start_ms = int(segment.start * 1000)  # Convert seconds to milliseconds
                        end_ms = int(segment.end * 1000)     # Convert seconds to milliseconds
                        
                        # Extract the segment
                        chorus_audio = audio[start_ms:end_ms]
                        os.makedirs(os.path.join(output_path, folder_name, song_name), exist_ok=True)
                        
                        
                        # Source Separation Model
                        
                        
                        
                        # Construct output file name
                        output_file = os.path.join(output_path, f"{song_name}_chorus_{chorus_counter}.mp3")
                        
                        # Export the segment
                        chorus_audio.export(output_file, format="mp3")
                        print(f"Exported: {output_file}")
                        
                        chorus_counter += 1
                    
                    break
                break
            
            break
            
    


if __name__ == "__main__":
    input_path = "/mnt/gestalt/database/beatport/audio/audio"
    output_path = "/mnt/gestalt/home/ddmanddman/beatport_preprocess"
    load_data_and_process(input_path, output_path)
