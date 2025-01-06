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


def process_folders(folder_names, input_path, output_path, device_id):
    device = torch.device(f'cuda:{device_id}')
    
    for folder_name in tqdm(folder_names, desc=f"Device {device_id}"):
        folder_path = os.path.join(input_path, folder_name)
        if os.path.isdir(folder_path):
            # Pre-process and store metadata
            audio_files = [
                os.path.join(folder_path, file_name)
                for file_name in os.listdir(folder_path)
                    if file_name.endswith(('.wav', '.mp3'))
            ]

            # Analyze by allin1
            results = allin1.analyze(
                audio_files,
                out_dir=os.path.join(output_path, "json"),
                demix_dir=output_path, 
                spec_dir=os.path.join(output_path, "spec"),
                device=device, 
                keep_byproducts=True
            )
            

def load_data_and_process(sel_list, input_path, output_path, devices=[1, 2, 3]):
    # Split folder names for each GPU
    folder_names = sel_list #os.listdir(input_path) # sel_list
    num_folders = len(folder_names)
    num_devices = len(devices)
    folders_per_device = (num_folders + num_devices - 1) // num_devices  # Divide evenly

    processes = []
    for i, device_id in enumerate(devices):
        folder_subset = folder_names[i * folders_per_device : (i + 1) * folders_per_device]
        process = Process(target=process_folders, args=(folder_subset, input_path, output_path, device_id))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()
    


if __name__ == "__main__":
    input_path = "/mnt/gestalt/database/beatport/audio/audio"
    output_path = "/mnt/gestalt/home/ddmanddman/beatport_analyze"
    target = 'chorus'
    devices = [2, 3]
    
    # Open folders
    os.makedirs(output_path, exist_ok=True)
    # os.makedirs(os.path.join(output_path, target), exist_ok=True)
    os.makedirs(os.path.join(output_path, sep_model_name), exist_ok=True)
    os.makedirs(os.path.join(output_path, "json"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "spec"), exist_ok=True)
    
    # given_list = [g.split('.json')[0] for g in os.listdir("/mnt/gestalt/home/ddmanddman/beatport_preprocess/json")]
    # sel_list = []
    # for i in os.listdir(input_path):
    #     cnt = 0
    #     for j in os.listdir(os.path.join(input_path, i)):
    #         if j.split('.mp3')[0] not in given_list:
    #             cnt += 1
    #             break
    #     if cnt != 0 : #and i != "electro-big-room" and i != "leftfield-house-and-techno":
    #         sel_list.append(i)
    # print("Selected List:", sel_list)
    sel_list = ['future-house', 'garage-baseline-grime', 
                'glitch-hop', 'hard-dance', 'hardcore-hard-techno', 'hip-hop-r-and-b']
    
    load_data_and_process(sel_list, input_path, output_path, devices)
    print("---All Well Done---")