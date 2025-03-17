import torch
from model.simple_dddm_mixup import DDDM
import utils

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
hps = utils.get_hparams()
model = DDDM(
    hps.data.n_mel_channels, hps.diffusion.spk_dim,
    hps.diffusion.dec_dim, hps.diffusion.beta_min, 
    hps.diffusion.beta_max, hps
).to(device)


model.eval()

x = torch.randn(4, 80, 200).to(device)