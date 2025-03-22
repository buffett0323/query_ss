import torch
import utils
from model.new_dddm_mixup import DDDM
from data_loader import MelSpectrogramFixed
from torch.nn import functional as F


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
hps = utils.get_hparams()
model = DDDM(
    hps.data.n_mel_channels, hps.diffusion.spk_dim,
    hps.diffusion.dec_dim, hps.diffusion.beta_min, 
    hps.diffusion.beta_max, hps
).to(device)


# Check if model is on cuda:3
if torch.cuda.is_available():
    # Get the device of the first parameter
    first_param_device = next(model.parameters()).device
    print(f"Model parameters device: {first_param_device}")
    assert first_param_device.index == 3, f"Model not on cuda:3, but on {first_param_device}"
    
    # Check if all parameters are on cuda:3
    all_on_cuda3 = all(param.device.index == 3 for param in model.parameters())
    if all_on_cuda3:
        print("All model parameters are on cuda:3")
    else:
        # Find any parameters not on cuda:3
        for name, param in model.named_parameters():
            if param.device.index != 3:
                print(f"Warning: Parameter {name} is on {param.device}, not cuda:3")
else:
    print("CUDA not available, model is on CPU")


mel_fn = MelSpectrogramFixed(
    sample_rate=hps.data.sampling_rate,
    n_fft=hps.data.filter_length,
    win_length=hps.data.win_length,
    hop_length=hps.data.hop_length,
    f_min=hps.data.mel_fmin,
    f_max=hps.data.mel_fmax,
    n_mels=hps.data.n_mel_channels,
    window_fn=torch.hann_window
).to(device)


 
x = torch.randn(4, 16000*4).to(device)
mel_y = torch.randn(4, 1, 256, 256).to(device)
mel_fn_x = mel_fn(x).to(device)
length = torch.LongTensor([mel_fn_x.size(2)]).to(device)# length = torch.randn(4).to(device)


# Run models
enc_output, mel_rec = model(x, mel_y, mel_fn_x, length, n_timesteps=6, mode='ml')
print("Result:", enc_output.shape, mel_rec.shape)

mel_loss = F.l1_loss(mel_fn_x, mel_rec).item()
enc_loss = F.l1_loss(mel_fn_x, enc_output).item()
print(mel_loss, enc_loss)