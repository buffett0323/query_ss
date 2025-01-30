"""
FMA + Beatport Music Dataset
Use conda env: allin1
"""
import os
import allin1
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Process
from pydub import AudioSegment
import warnings

np.int = int
np.float = float

torch.multiprocessing.set_start_method('spawn', force=True)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Global Vars
sep_model_name = 'htdemucs'


def process_batch(file_names, input_path, output_path, device_id):
    """
    Process a batch of files on a specific GPU to prevent exceeding memory limits.
    """
    device = torch.device(f'cuda:{device_id}')
    
    for file_name in tqdm(file_names, desc=f"Device {device_id}"):
        file_path = [
            os.path.join(input_path, f) 
            for f in file_name
            if f.endswith(('.wav', '.mp3'))
        ]
      
        # Process in small chunks to prevent overloading GPU memory
        results = allin1.analyze(
            file_path,
            out_dir=os.path.join(output_path, "json"),
            demix_dir=output_path, 
            spec_dir=os.path.join(output_path, "spec"),
            device=device, 
            keep_byproducts=True
        )
        print(f"Result Done at {device}")
        
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()



def load_data_and_process(input_path, output_path, devices=[2, 3], chunk_size=10):
    """
    Distribute file processing across multiple GPUs while ensuring all processes run to completion.
    """
    folder_names = os.listdir(input_path)  # List of files in input_path
    new_folder_names = [folder_names[i:i + chunk_size] for i in range(0, len(folder_names), chunk_size)]

    num_devices = len(devices)
    processes = []

    # Create processes instead of using Pool to avoid daemon-related issues
    for i, device_id in enumerate(devices):
        folder_subset = new_folder_names[i::num_devices]  # Distribute evenly
        process = Process(target=process_batch, args=(folder_subset, input_path, output_path, device_id))
        process.start()
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.join()


if __name__ == "__main__":
    input_path = "/mnt/gestalt/database/FMA/fma_track/audio"
    output_path = "/mnt/gestalt/home/ddmanddman/fma_analyze"
    target = 'chorus'
    devices = [2, 3]  # Available GPUs

    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, sep_model_name), exist_ok=True)
    os.makedirs(os.path.join(output_path, "json"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "spec"), exist_ok=True)

    # Pre-process with limited GPU usage
    load_data_and_process(input_path, output_path, devices)
    print("---All Well Done---")
