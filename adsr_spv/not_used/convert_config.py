import yaml
from adsr_spv.not_used.config import TrainConfig

def convert_config_to_yaml():
    """Convert the old Python config class to YAML format."""
    config = TrainConfig()

    # Convert config to dictionary
    config_dict = {
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'device': config.device,
        'data_dir': str(config.data_dir),
        'wandb_use': config.wandb_use,
        'wandb_dir': str(config.wandb_dir),
        'wandb_project': config.wandb_project,
        'wandb_name': config.wandb_name,
        'd_model': config.d_model,
        'd_out': config.d_out,
        'n_heads': config.n_heads,
        'n_layers': config.n_layers,
        'patch_size': config.patch_size,
        'patch_stride': config.patch_stride,
        'input_channels': config.input_channels,
        'spec_shape': list(config.spec_shape),
        'lr': config.lr,
        'epochs': config.epochs,
        'save_interval': config.save_interval,
        'param_weight': config.param_weight,
        'spectral_weight': config.spectral_weight,
        'sr': config.sr,
        'n_fft': config.n_fft,
        'n_mels': config.n_mels,
        'fmin': config.fmin,
        'fmax': config.fmax,
        # Lightning specific settings
        'accelerator': 'gpu',
        'devices': 1,
        'precision': 16,
        'gradient_clip_val': 1.0,
        'accumulate_grad_batches': 1,
        'log_every_n_steps': 50,
    }

    # Save to YAML file
    with open('config.yaml', 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    print("Config converted to config.yaml")

if __name__ == "__main__":
    convert_config_to_yaml()
