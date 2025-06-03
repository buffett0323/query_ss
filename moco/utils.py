import yaml
import os
import torch
import multiprocessing
import json
import lmdb
import numpy as np
import pickle
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


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix: str = "") -> None:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_and_serialize(song_file_pair):
    song, file_path = song_file_pair
    try:
        arr = np.load(file_path)
        key = song.encode()
        val = pickle.dumps(arr, protocol=4)
        return key, val
    except Exception as e:
        print(f"[ERROR] Failed on {file_path}: {e}")
        return None


def create_lmdb(data_dir, lmdb_path):
    map_size = 10 * 1024 ** 3  # 10GB
    env = lmdb.open(
        lmdb_path,
        map_size=map_size,
        writemap=True,        # ✅ performance boost
        map_async=True,       # ✅ performance boost
        readahead=False
    )

    with open("info/train_seg_counter_amp_05.json", "r") as f:
        seg_counter = json.load(f)

    # Pre-collect all file paths
    npy_files = [
        (song, os.path.join(data_dir, song, "bass_other_seg_0.npy"))
        for song in seg_counter
    ]
    print("Got", len(npy_files), "files")


    # Use multiprocessing to load and serialize data
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(load_and_serialize, npy_files), total=len(npy_files)))

    # Filter out failed ones
    results = [r for r in results if r is not None]

    # Write to LMDB
    batch_size = 5000
    txn = env.begin(write=True)
    for i, (key, val) in enumerate(tqdm(results, desc="Writing to LMDB")):
        txn.put(key, val)
        if (i + 1) % batch_size == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.sync()
    env.close()

    print(f"✅ LMDB created at {lmdb_path}")
