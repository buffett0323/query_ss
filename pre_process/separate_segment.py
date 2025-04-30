# Use conda env: allin1
import os
import json
import random
import yaml
import h5py
import torch
import torchaudio
import argparse
import warnings
import numpy as np
import torchaudio.transforms as T

from tqdm import tqdm
from multiprocessing import Process, cpu_count
from pydub import AudioSegment

def yaml_config_hook(config_file):
    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg

np.int = int
np.float = float

torch.multiprocessing.set_start_method('spawn', force=True)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Global Vars
sources = ['bass', 'drums', 'other', 'vocals']

# given_list = [g.split('.json')[0] for g in os.listdir("/mnt/gestalt/home/ddmanddman/beatport_preprocess/json")]
# path = "/mnt/gestalt/database/beatport/audio/audio"
# sel_list = []
# for i in os.listdir(path):
#     cnt = 0
#     for j in os.listdir(os.path.join(path, i)):
#         if j.split('.mp3')[0] not in given_list:
#             cnt += 1
#     if cnt != 0 :
#         sel_list.append(i)
# print("Selected list is:", sel_list)
                
                
def segment_audio(args, song_name, segments, sep_path, output_path, target='chorus'):
    chorus_counter = 1
    for segment in segments:
        if segment['label'] == target:
            start_sr = int(segment['start'] * args.sample_rate)
            end_sr = int(segment['end'] * args.sample_rate)
            
            # Threshold for 3 seconds
            if (segment['end'] - segment['start']) > args.segment_thres:
                # Extract the segment for the sources
                os.makedirs(os.path.join(output_path, target, f"{song_name}_{chorus_counter}"), exist_ok=True)
                os.makedirs(os.path.join(output_path, f"{target}_npy", f"{song_name}_{chorus_counter}"), exist_ok=True)
                # os.makedirs(os.path.join(output_path, f"{target}_h5", f"{song_name}_{chorus_counter}"), exist_ok=True)
                
                if len(os.listdir(os.path.join(output_path, f"{target}_npy", f"{song_name}_{chorus_counter}"))) >= 4:
                    chorus_counter += 1
                    continue
                
                for source in sources:
                    audio, _ = torchaudio.load(os.path.join(sep_path, f"{source}.wav"))
                    seg_audio = audio[:, start_sr:end_sr]
                    
                    # Pre-process for unified segment length
                    target_length = (args.segment_second * args.sample_rate)
                    if seg_audio.size(-1) < target_length:
                        repeat_count = (target_length // seg_audio.size(-1)) + 1
                        seg_audio = seg_audio.repeat(1, repeat_count)
                    
                    if seg_audio.size(-1) > target_length:
                        # Randomly crop to 5 seconds
                        start_point = random.randint(0, seg_audio.size(-1) - target_length)
                        seg_audio = seg_audio[:, start_point:start_point + target_length]
                    
                    if seg_audio.shape[-1] != 220500:
                        print(audio.shape, seg_audio.shape)
                        
                    
                    # Transform to mel-spectrogram
                    mel_transform = T.MelSpectrogram(
                        sample_rate=args.sample_rate,
                        n_fft=args.n_fft,
                        hop_length=args.hop_length,
                        n_mels=args.n_mels
                    )
                    mel_spec = mel_transform(seg_audio)
                    
                    # Store files in multiple ways
                    # 1. Save waveform as MP3
                    mp3_output_file = os.path.join(output_path, target, f"{song_name}_{chorus_counter}", f"{source}_{chorus_counter}.mp3")
                    torchaudio.save(mp3_output_file, seg_audio, args.sample_rate, format="mp3")

                    # 2. Save mel-spectrogram as .npy
                    mel_npy_output_file = os.path.join(output_path, f"{target}_npy", f"{song_name}_{chorus_counter}", f"{source}_{chorus_counter}_mel.npy")
                    np.save(mel_npy_output_file, mel_spec.numpy())
                    
                    
                    # # 3. Save to HDF5
                    # h5_output_file = os.path.join(output_path, f"{target}_h5", f"{song_name}_{chorus_counter}.h5")
                    # with h5py.File(h5_output_file, "a") as h5f:
                    #     h5f.create_dataset(f"{source}_waveform", data=waveform.numpy())
                    #     h5f.create_dataset(f"{source}_mel_spec", data=mel_spec.numpy())              
                
                chorus_counter += 1

        
def process_folders(my_args, folder_names, output_path, target):
    for js in tqdm(folder_names):
        json_path = os.path.join(output_path, "json", js)
        result = json.load(open(json_path))
        song_name = js.split('.json')[0]
        
        segment_audio(
            args=my_args,
            song_name=song_name, 
            segments=result['segments'], 
            sep_path=os.path.join(output_path, my_args.sep_model_name, song_name), 
            output_path=output_path, 
            target=target,
        )
        

def load_data_and_process(my_args, output_path, target='chorus'):
    folder_names = os.listdir(os.path.join(output_path, "json"))
    num_processes = min(cpu_count(), len(folder_names))  # Limit to available CPU cores
    folders_per_process = (len(folder_names) + num_processes - 1) // num_processes

    processes = []
    for i in range(num_processes):
        folder_subset = folder_names[i * folders_per_process : (i + 1) * folders_per_process]
        process = Process(target=process_folders, args=(my_args, folder_subset, output_path, target))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")

    config = yaml_config_hook("../config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    output_path = "/mnt/gestalt/home/ddmanddman/beatport_analyze"
    target = args.target
    
    # Open folders
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, target), exist_ok=True)
    os.makedirs(os.path.join(output_path, args.sep_model_name), exist_ok=True)
    os.makedirs(os.path.join(output_path, "json"), exist_ok=True)
    os.makedirs(os.path.join(output_path, f"{target}_npy"), exist_ok=True) # Mel-spec
    # os.makedirs(os.path.join(output_path, f"{target}_h5"), exist_ok=True) # waveform for h5
    
    load_data_and_process(args, output_path, target)
    print("---All Well Done---")
    
    
    
    


# def load_data_and_process(my_args, input_path, output_path, target='chorus', devices=[1, 2, 3]):
#     # Split folder names for each GPU
#     folder_names =  os.listdir(input_path) # sel_list
#     num_folders = len(folder_names)
#     num_devices = len(devices)
#     folders_per_device = (num_folders + num_devices - 1) // num_devices  # Divide evenly

#     processes = []
#     for i, device_id in enumerate(devices):
#         folder_subset = folder_names[i * folders_per_device : (i + 1) * folders_per_device]
#         process = Process(target=process_folders, args=(my_args, folder_subset, input_path, output_path, target))
#         processes.append(process)
#         process.start()

#     # Wait for all processes to complete
#     for process in processes:
#         process.join()


# def process_folders(my_args, folder_names, output_path, target):
#     for js in tqdm(os.listdir(os.path.join(output_path, "json"))):
#         # Segment audio
#         result = json.load(js)
#         song_name = js.split('.json')[0]
        
#         # Segment audio to get chorus
#         segment_audio(
#             args=my_args,
#             song_name=song_name, 
#             segments=result.segments, 
#             sep_path=os.path.join(output_path, my_args.sep_model_name, song_name), 
#             output_path=output_path, 
#             target=target,
#         )



# def segment_audio(args, song_name, segments, sep_path, output_path, target='chorus'):
#     chorus_counter = 1
#     for segment in segments:
#         if segment.label == target:
#             start_ms = int(segment.start * 1000)  # Convert seconds to milliseconds
#             end_ms = int(segment.end * 1000)     # Convert seconds to milliseconds
            
#             # Threshold for 3 seconds
#             if (start_ms - end_ms) > (args.segment_thres * 1000):
#                 # Extract the segment for the sources
#                 os.makedirs(os.path.join(output_path, target, f"{song_name}_{chorus_counter}"), exist_ok=True)
#                 os.makedirs(os.path.join(output_path, f"{target}_h5", f"{song_name}_{chorus_counter}"), exist_ok=True)
#                 os.makedirs(os.path.join(output_path, f"{target}_npy", f"{song_name}_{chorus_counter}"), exist_ok=True)
#                 for source in sources:
#                     audio = AudioSegment.from_file(os.path.join(sep_path, f"{source}.wav"))
#                     seg_audio = audio[start_ms:end_ms]
                    
#                     """ Pre-process for unify all the segment length """
#                     target_length = (args.segment_second * 1000)
#                     if seg_audio.size(-1) < target_length:
#                         repeat_count = (args.max_length // seg_audio.size(-1)) + 1
#                         seg_audio = seg_audio.repeat(1, 1, repeat_count)
                    
#                     if seg_audio.size(-1) > target_length:
#                         # Randomly crop to 5 seconds
#                         start_point = random.randint(0, len(seg_audio) - target_length)
#                         seg_audio = seg_audio[start_point:start_point + target_length]
                    
                    
#                     # Convert to waveform tensor
#                     waveform = torch.tensor(seg_audio.get_array_of_samples()).float().unsqueeze(0)

#                     # Transform to mel-spectrogram
#                     mel_transform = T.MelSpectrogram(
#                         sample_rate=args.sample_rate,
#                         n_fft=args.n_fft,
#                         hop_length=args.hop_length,
#                         n_mels=args.n_mels
#                     )
#                     mel_spec = mel_transform(waveform)
                    
#                     """ Store files in multiple ways """
#                     # 1. Export MP3
#                     mp3_output_file = os.path.join(output_path, target, f"{song_name}_{chorus_counter}", f"{source}_{chorus_counter}.mp3")
#                     os.makedirs(os.path.dirname(mp3_output_file), exist_ok=True)
#                     seg_audio.export(mp3_output_file, format="mp3")
                                
#                     # 2. Save mel-spectrogram as .npy
#                     mel_npy_output_file = os.path.join(output_path, f"{target}_npy", f"{song_name}_{chorus_counter}", f"{source}_{chorus_counter}_mel.npy")
#                     np.save(mel_npy_output_file, mel_spec.numpy())
                    
#                     # 3. Save to HDF5
#                     h5_output_file = os.path.join(output_path, f"{target}_h5", f"{song_name}_{chorus_counter}.h5")
#                     with h5py.File(h5_output_file, "a") as h5f:
#                         h5f.create_dataset(f"{source}_waveform", data=waveform.numpy())
#                         h5f.create_dataset(f"{source}_mel_spec", data=mel_spec.numpy())              
                
#                 chorus_counter += 1