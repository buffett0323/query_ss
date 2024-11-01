from omegaconf import OmegaConf

def _load_config(config_path: str) -> OmegaConf:
    config = OmegaConf.load(config_path)

    config_dict = {}

    for k, v in config.items():
        if isinstance(v, str) and v.endswith(".yml"):
            config_dict[k] = OmegaConf.load(v)
        else:
            config_dict[k] = v

    config = OmegaConf.merge(config_dict)

    return config