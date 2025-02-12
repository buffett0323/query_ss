import os
import yaml
import random

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


def train_test_split_BPDataset(path="/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_6secs_npy"):
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

    # Save to text files
    def save_to_txt(filename, data):
        with open(filename, "w") as f:
            for item in data:
                f.write(f"{item}\n")

    save_to_txt("info/train_bp.txt", train_files)
    save_to_txt("info/valid_bp.txt", valid_files)
    save_to_txt("info/test_bp.txt", test_files)
    print(f"Dataset split complete: {train_size} train, {valid_size} valid, {test_size} test")


def load_from_txt(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines()]
    
    

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
    train_test_split_BPDataset()