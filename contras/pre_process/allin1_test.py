# Use conda env: allin1
import os
import allin1
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
from pydub import AudioSegment
import warnings
import traceback 

np.int = int
np.float = float

torch.multiprocessing.set_start_method('spawn', force=True)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Global Vars
sep_model_name = 'htdemucs'
sources = ['bass', 'drums', 'other', 'vocals']
error_log_path = "error_log.txt"  # Define error log file path

audio_files = ["psy-trance/b36be413-daa5-484b-b3cc-78f3a6acfe85.mp3", "progressive-house/c5f8d324-2bdb-4456-b76d-0c12fc8682e1.mp3", 
                "electro-big-room/7ac465df-dbee-4e51-9102-6d1c3afb66bc.mp3", "house/b8a01c12-a84c-4c06-8afa-8158145bacfd.mp3"]


input_path = "/mnt/gestalt/database/beatport/audio/audio"
output_path = "/mnt/gestalt/home/ddmanddman/beatport_analyze"
target = 'chorus'
device = torch.device("cuda:2")
audio_files = [os.path.join(input_path, f) for f in audio_files]

try:
    # Analyze using allin1
    results = allin1.analyze(
        audio_files,
        out_dir=os.path.join(output_path, "json"),
        demix_dir=output_path, 
        spec_dir=os.path.join(output_path, "spec"),
        device=device, 
        keep_byproducts=True
    )

except Exception as e:
    error_message = f"Error in folder: {audio_files} on GPU \n"
    error_message += traceback.format_exc()  # Capture full error traceback
    print(error_message)

    # Append error message to a log file
    with open(error_log_path, "a") as log_file:
        log_file.write(error_message + "\n")

    




print("---All Well Done---")