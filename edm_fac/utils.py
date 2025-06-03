from omegaconf import OmegaConf
from tqdm import tqdm
import os
import torch

def yaml_config_hook(config_file):
    """
    Load YAML with OmegaConf to support ${variable} interpolation.
    Also supports nested includes via a 'defaults' section.
    """
    # Load main config
    cfg = OmegaConf.load(config_file)

    # Load nested defaults if any (like Hydra-style)
    if "defaults" in cfg:
        for d in cfg.defaults:
            config_dir, cf = d.popitem()
            cf_path = os.path.join(os.path.dirname(config_file), config_dir, f"{cf}.yaml")
            nested_cfg = OmegaConf.load(cf_path)
            cfg = OmegaConf.merge(cfg, nested_cfg)

        del cfg.defaults

    return cfg



def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch



def save_checkpoint(args, iter, wrapper):
    """Save model checkpoint and optimizer state"""
    checkpoint_path = os.path.join(args.ckpt_path, f'checkpoint_{iter}.pt')

    # Save generator
    torch.save({
        'generator_state_dict': wrapper.generator.state_dict(),
        'optimizer_g_state_dict': wrapper.optimizer_g.state_dict(),
        'scheduler_g_state_dict': wrapper.scheduler_g.state_dict(),
        'discriminator_state_dict': wrapper.discriminator.state_dict(),
        'optimizer_d_state_dict': wrapper.optimizer_d.state_dict(),
        'scheduler_d_state_dict': wrapper.scheduler_d.state_dict(),
        'iter': iter
    }, checkpoint_path)

    # Save latest checkpoint by creating a symlink
    latest_path = os.path.join(args.ckpt_path, 'checkpoint_latest.pt')
    if os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(checkpoint_path, latest_path)

    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(args, device, iter, wrapper):
    """Load model checkpoint and optimizer state"""
    if iter == -1:
        # Load latest checkpoint
        checkpoint_path = os.path.join(args.ckpt_path, 'checkpoint_latest.pt')
    else:
        # Load specific checkpoint
        checkpoint_path = os.path.join(args.ckpt_path, f'checkpoint_{iter}.pt')

    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load generator
    wrapper.generator.load_state_dict(checkpoint['generator_state_dict'])
    wrapper.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    wrapper.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])

    # Load discriminator
    wrapper.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    wrapper.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    wrapper.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])

    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint['iter']



def get_timbre_names(path):
    timbres = set()
    for file in tqdm(os.listdir(path)):
        if file.endswith(".wav"):
            timbres.add(file.split("_")[0])

    with open(f"info/timbre_names_{path.split('/')[-1]}.txt", "w") as f:
        for timbre in timbres:
            f.write(timbre + "\n")

    print(len(timbres))
