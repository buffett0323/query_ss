"""
FMA + Beatport Music Dataset
Use conda env: allin1
"""
import os
import allin1
import shutil
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Process
from pydub import AudioSegment
import warnings

from utils import detect_large_file

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
            if f.endswith(('.wav', '.mp3')) and \
                detect_large_file(os.path.join(input_path, f)) == True
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
        
        # torch.cuda.synchronize(device)
        # torch.cuda.empty_cache()


def load_data_and_process(input_path, output_path, devices, chunk_size=5):
    # Distribute file processing across multiple GPUs while ensuring all processes run to completion.
    
    htdemucs_folder = os.listdir("/mnt/gestalt/home/ddmanddman/fma_analyze/htdemucs")
    json_folder = [js.split('.json')[0] for js in os.listdir("/mnt/gestalt/home/ddmanddman/fma_analyze/json")]
    spec_folder = [sp.split('.npy')[0] for sp in os.listdir("/mnt/gestalt/home/ddmanddman/fma_analyze/spec")]
    
    # Remove htdemucs folder if json file not exists
    for ht in htdemucs_folder:
        if ht not in json_folder:
            ht_pth = os.path.join("/mnt/gestalt/home/ddmanddman/fma_analyze/htdemucs", ht)#f"'{ht}'")
            if os.path.exists(ht_pth):
                shutil.rmtree(ht_pth)
                print("Remove HT")
                
    
    for sp in spec_folder:
        if sp not in json_folder:
            sp_pth = os.path.join("/mnt/gestalt/home/ddmanddman/fma_analyze/spec", f"{sp}.npy")
            if os.path.exists(sp_pth):
                os.remove(sp_pth)
                print("Remove Spec")
                
                
    # Remove already done tracks
    mp3_files = [mp.split('.mp3')[0] for mp in os.listdir(input_path)]  # List of files in input_path
    print("MP3:", len(mp3_files))
    
    folder_names = [f"{mp3}.mp3" for mp3 in mp3_files if mp3 not in json_folder]
    print("Filter:", len(folder_names))
    
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
    devices = [1, 3]  # Available GPUs

    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, sep_model_name), exist_ok=True)
    os.makedirs(os.path.join(output_path, "json"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "spec"), exist_ok=True)

    # Pre-process with limited GPU usage
    load_data_and_process(input_path, output_path, devices)
    print("---All Well Done---")
