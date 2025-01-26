"""
FMA + Beatport Music Dataset
"""
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

np.int = int
np.float = float

torch.multiprocessing.set_start_method('spawn', force=True)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Global Vars
sep_model_name = 'htdemucs'
sources = ['bass', 'drums', 'other', 'vocals']


def process_folders(file_names, input_path, output_path, device_id):
    device = torch.device(f'cuda:{device_id}')
    
    for file_name in tqdm(file_names, desc=f"Device {device_id}"):
        file_path = os.path.join(input_path, file_name)
        if os.path.exists(file_path):

            # Analyze by allin1
            allin1.analyze(
                file_path,
                out_dir=os.path.join(output_path, "json"),
                demix_dir=output_path, 
                spec_dir=os.path.join(output_path, "spec"),
                device=device, 
                keep_byproducts=True
            )
            

def load_data_and_process(input_path, output_path, devices=[1, 2, 3]):
    # Split folder names for each GPU
    folder_names = os.listdir(input_path)
    num_devices = len(devices)
    num_folders = len(folder_names)
    folders_per_device = (num_folders + num_devices - 1) // num_devices  # Divide evenly

    processes = []
    for i, device_id in enumerate(devices):
        files_device = folder_names[i * folders_per_device : (i + 1) * folders_per_device]
        process = Process(target=process_folders, args=(files_device, input_path, output_path, device_id))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()
    


if __name__ == "__main__":
    input_path = "/mnt/gestalt/database/FMA/fma_track/audio" #"/mnt/gestalt/database/beatport/audio/audio"
    output_path = "/mnt/gestalt/home/ddmanddman/fma_analyze" #"/mnt/gestalt/home/ddmanddman/beatport_analyze"
    target = 'chorus'
    devices = [2, 3]
    
    # Open folders
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, sep_model_name), exist_ok=True)
    os.makedirs(os.path.join(output_path, "json"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "spec"), exist_ok=True)
    

    load_data_and_process(input_path, output_path, devices)
    print("---All Well Done---")