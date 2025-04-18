import os
import yaml
import random
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from collections import defaultdict
from tqdm import tqdm



def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

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


def define_param_groups(model, weight_decay, optimizer_name):
   def exclude_from_wd_and_adaptation(name):
       if 'bn' in name:
           return True
       if optimizer_name == 'lars' and 'bias' in name:
           return True

   param_groups = [
       {
           'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
           'weight_decay': weight_decay,
           'layer_adaptation': True,
       },
       {
           'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
           'weight_decay': 0.,
           'layer_adaptation': False,
       },
   ]
   return param_groups


# Save to text files
def save_to_txt(filename, data):
    with open(filename, "w") as f:
        for item in data:
            f.write(f"{item}\n")
            
def load_from_txt(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines()]


def train_test_split_BPDataset(path="/home/buffett/NAS_NTU/beatport_analyze/verse_audio_16000_npy"):
    path_file = os.listdir(path)
    random.shuffle(path_file)

    # Compute split sizes
    total_files = len(path_file)
    train_size = int(total_files * 9 / 10)
    valid_size = int(total_files * 0.5 / 10)
    test_size = total_files - train_size - valid_size  # Ensure all files are allocated

    # Split dataset
    train_files = path_file[:train_size]
    valid_files = path_file[train_size:train_size + valid_size]
    test_files = path_file[train_size + valid_size:]

    save_to_txt("info/train_bp_verse_8secs.txt", train_files)
    save_to_txt("info/valid_bp_verse_8secs.txt", valid_files)
    save_to_txt("info/test_bp_verse_8secs.txt", test_files)
    print(f"Dataset split complete: {train_size} train, {valid_size} valid, {test_size} test")


def split_dataset_by_song(path="/home/buffett/NAS_NTU/beatport_analyze/verse_audio_16000_npy"):
    path_file = os.listdir(path)
    random.shuffle(path_file)
    
    song_dict = defaultdict(lambda: "train")
    song_counter_dict = defaultdict(lambda: [])
    unique_songs = set()
    
    for file in path_file:
        song_name, song_num = file[:-2], int(file[-1])
        unique_songs.add(song_name)
        song_counter_dict[song_name].append(song_num)
        
    
    # Compute split sizes
    total_files = len(unique_songs)
    train_size = int(total_files * 9 / 10)
    valid_size = int(total_files * 0.5 / 10)
    test_size = total_files - train_size - valid_size  # Ensure all files are allocated
    print("Size check: ", "train", train_size, "valid", valid_size, "test", test_size)
    
    
    # Split dataset by song name
    for i, song_name in tqdm(enumerate(unique_songs)):
        if i < test_size:
            song_dict[song_name] = "test"
        elif i >= test_size and i < test_size + valid_size:
            song_dict[song_name] = "valid"
        else:
            song_dict[song_name] = "train"
            

    # Add song number to the song name
    train_files, valid_files, test_files = [], [], []
    
    for song_name, song_split in tqdm(song_dict.items()):
        if song_split == "train":
            for i in song_counter_dict[song_name]:
                train_files.append(f"{song_name}_{i}")
        elif song_split == "valid":
            for i in song_counter_dict[song_name]:
                valid_files.append(f"{song_name}_{i}")
        else:
            for i in song_counter_dict[song_name]:
                test_files.append(f"{song_name}_{i}")
    

    random.shuffle(train_files)
    random.shuffle(valid_files)
    random.shuffle(test_files)

    save_to_txt("info/split_by_song_name_4secs_train.txt", train_files)
    save_to_txt("info/split_by_song_name_4secs_valid.txt", valid_files)
    save_to_txt("info/split_by_song_name_4secs_test.txt", test_files)
    print(f"Dataset split complete: {len(train_files)} train, {len(valid_files)} valid, {len(test_files)} test")

    

    
    
def plot_spec_and_save(spectrogram, savefig_name, sr=16000):
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.squeeze().numpy()
                
    plt.figure(figsize=(6, 6))
    librosa.display.specshow(
        spectrogram,
        x_axis='time', 
        y_axis='mel', 
        sr=sr,
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel-Spectrogram")
    plt.savefig(savefig_name)


def resize_spec(spectrogram, target_size=(256, 256)):
    """ Resize spectrogram using interpolation to fit Swin Transformer input """
    resized_spec = zoom(spectrogram, (target_size[0] / spectrogram.shape[0], target_size[1] / spectrogram.shape[1]), order=3)
    return resized_spec

    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
    
if __name__ == "__main__":
    # train_test_split_BPDataset()
    split_dataset_by_song(path="/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_4secs_npy")