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

problem_list = ["b36be413-daa5-484b-b3cc-78f3a6acfe85.mp3", "c5f8d324-2bdb-4456-b76d-0c12fc8682e1.mp3",
                "7ac465df-dbee-4e51-9102-6d1c3afb66bc.mp3", "b8a01c12-a84c-4c06-8afa-8158145bacfd.mp3"]

def process_folders(folder_names, input_path, output_path, device_id):
    device = torch.device(f'cuda:{device_id}')

    for folder_name in tqdm(folder_names, desc=f"Device {device_id}"):
        folder_path = os.path.join(input_path, folder_name)
        if os.path.isdir(folder_path):
            # Pre-process and store metadata
            audio_files = [
                os.path.join(folder_path, file_name)
                for file_name in os.listdir(folder_path)
                    if file_name.endswith(('.wav', '.mp3')) and file_name not in problem_list
            ]

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

            except Exception:
                error_message = f"Error in folder: {folder_name} on GPU {device_id}\n"
                error_message += traceback.format_exc()  # Capture full error traceback
                print(error_message)

                # Append error message to a log file
                with open(error_log_path, "a") as log_file:
                    log_file.write(error_message + "\n")


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
    devices = [2]

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

    ### TODO: ['electro-big-room', 'house', 'progressive-house', 'psy-trance']
    # 'electro-big-room' --> corrupt
    # 'house' --> corrupt
    sel_list = ['house'] #['progressive-house'] #['psy-trance']

    load_data_and_process(sel_list, input_path, output_path, devices)
    print("---All Well Done---")
