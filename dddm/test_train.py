import torch
import utils
from model.te_dddm_mixup import DDDM
from data_loader import MelSpectrogramFixed
from torch.nn import functional as F


hps = utils.get_hparams(
    config_path="./ckpt/config.json",
    model_dir="/home/buffett/nas_home/buffett/timbre_transfer_logs/",
)
device = torch.device(f"cuda:{hps.train.device}")
model = DDDM(
    hps.data.n_mel_channels, hps.diffusion.spk_dim,
    hps.diffusion.dec_dim, hps.diffusion.beta_min,
    hps.diffusion.beta_max, hps
).to(device)


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
mel_x = torch.randn(4, 1, 256, 256).to(device)
mel_fn_x = mel_fn(x).to(device)
length = torch.LongTensor([mel_fn_x.size(2)] * 4).to(device)# length = torch.randn(4).to(device)

# Run models
loss_diff, loss_mel = model.compute_loss(x, mel_x, mel_fn_x, length)
print(loss_diff, loss_mel)


# Run evaluation
print("Run evaluation:")

enc_output, mel_rec = model(x, mel_x, mel_fn_x, length)

mel_loss = F.l1_loss(mel_fn_x, mel_rec).item()
enc_loss = F.l1_loss(mel_fn_x, enc_output).item()
print(mel_loss, enc_loss)
