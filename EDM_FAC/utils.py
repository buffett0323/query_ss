from omegaconf import OmegaConf
import os

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