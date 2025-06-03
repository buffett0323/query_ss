import os
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from scipy.io.wavfile import write
import torchaudio
import utils
import librosa
from torchvision import transforms

from data_loader import MelSpectrogramFixed
from model.new_dddm_mixup import DDDM
from vocoder.hifigan import HiFi

h = None
device = None
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def load_audio(path):
    if path.endswith('.wav'):
        audio, sr = torchaudio.load(path)
        audio = audio[:1]
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000, resampling_method="kaiser_window")

        p = (audio.shape[-1] // 1280 + 1) * 1280 - audio.shape[-1]
        audio = torch.nn.functional.pad(audio, (0, p))

        return audio
    elif path.endswith('.npy'):
        audio = np.load(path)
        audio = torch.from_numpy(audio)
        return audio
    else:
        raise ValueError(f"Unsupported file format: {path}")


def mel_timbre(x):
    # Infos:
    img_mean = -1.100174903869629
    img_std = 14.353998184204102

    # To numpy
    x = x.numpy()
    x = librosa.feature.melspectrogram(
        y=x,
        sr=16000,
        n_fft=1024,
        hop_length=256,
    )
    x = librosa.power_to_db(np.abs(x))
    x = torch.from_numpy(x).unsqueeze(0)

    # Resize to 256x256
    resizer = transforms.Resize((256, 256))
    x = resizer(x)
    return (x - img_mean) / img_std



def save_audio(wav, out_file, syn_sr=16000):
    wav = (wav.squeeze() / wav.abs().max() * 0.999 * 32767.0).detach().cpu().numpy().astype('int16') #.cpu().numpy().astype('int16')
    write(out_file, syn_sr, wav)


def inference(a):
    os.makedirs(a.output_dir, exist_ok=True)
    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    ).to(device)#.cuda()

    # Load pre-trained w2v (XLS-R)
    # w2v = Wav2vec2().cuda()

    # Load model
    # f0_quantizer = Quantizer(hps).cuda()
    # utils.load_checkpoint(a.ckpt_f0_vqvae, f0_quantizer)
    # f0_quantizer.eval()

    model = DDDM(
        hps.data.n_mel_channels, hps.diffusion.spk_dim,
        hps.diffusion.dec_dim, hps.diffusion.beta_min,
        hps.diffusion.beta_max, hps
    ).to(device)#.cuda()
    utils.load_checkpoint(a.ckpt_model, model, None)
    model.eval()

    # Load vocoder
    net_v = HiFi(hps.data.n_mel_channels, hps.train.segment_size // hps.data.hop_length, **hps.model).to(device)#.cuda()
    utils.load_checkpoint(a.ckpt_voc, net_v, None)
    net_v.eval().dec.remove_weight_norm()

    # Synthesis for original audio
    src_name = os.path.splitext(os.path.basename(a.src_path))[0]
    audio = load_audio(a.src_path)[:64000].unsqueeze(0)#.to(device)
    mel_audio = mel_timbre(audio).to(device)

    audio = audio.to(device)
    src_mel = mel_fn(audio).to(device)#.cuda())
    src_length = torch.LongTensor([src_mel.size(-1)]).to(device)#.cuda()
    _, mel_rec = model(audio, mel_audio, src_mel, src_length, n_timesteps=6, mode='ml')
    y_hat = net_v(mel_rec).squeeze(0)#.squeeze(0)

    # Save original audio
    save_audio(audio, os.path.join(a.output_dir, f'{a.output_name}_orig.wav'))
    save_audio(y_hat, os.path.join(a.output_dir, f'{a.output_name}_orig_synth.wav'))

    # Synthesis for target audio
    trg_name = os.path.splitext(os.path.basename(a.trg_path))[0]
    trg_audio = load_audio(a.trg_path)[:64000].unsqueeze(0)#.to(device)
    mel_trg_audio = mel_timbre(trg_audio).to(device)

    trg_audio = trg_audio.to(device)
    trg_mel = mel_fn(trg_audio).to(device)#.cuda())
    trg_length = torch.LongTensor([trg_mel.size(-1)]).to(device)

    _, mel_rec = model(trg_audio, mel_trg_audio, trg_mel, trg_length, n_timesteps=6, mode='ml')
    y_hat1 = net_v(mel_rec).squeeze(0)#.squeeze(0)

    # Save target audio
    save_audio(trg_audio, os.path.join(a.output_dir, f'{a.output_name}_trg.wav'))
    save_audio(y_hat1, os.path.join(a.output_dir, f'{a.output_name}_trg_synth.wav'))

    # Convert audio
    with torch.no_grad():
        c = model.vc(audio, mel_audio, src_mel, src_length,
                     trg_audio, mel_trg_audio, trg_mel, trg_length,
                     n_timesteps=a.time_step, mode='ml')
        converted_audio = net_v(c).squeeze(0).squeeze(0)

    # Save converted audio
    save_audio(converted_audio, os.path.join(a.output_dir, f'{a.output_name}_converted.wav' ))
    print(">> Done.")


def main():
    print('>> Initializing Inference Process...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_8secs_npy/')
    parser.add_argument('--src_path', type=str, default='/workspace/ha0/data/src.wav')
    parser.add_argument('--trg_path', type=str, default='/workspace/ha0/data/tar.wav')
    parser.add_argument('--ckpt_model', type=str, default='/mnt/gestalt/home/buffett/tt_training/timbre_transfer_te_train_dict/G_200000.pth')
    parser.add_argument('--ckpt_voc', type=str, default='/mnt/gestalt/home/buffett/hifigan_ckpt/voc_ckpt.pth')
    parser.add_argument('--logs_path', type=str, default='/mnt/gestalt/home/buffett/timbre_transfer_logs')
    parser.add_argument('--output_dir', '-o', type=str, default='./converted')
    parser.add_argument('--output_name', type=str, default='1')
    parser.add_argument('--time_step', '-t', type=int, default=6)
    parser.add_argument('--device', '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                    help="Choose device: 'cpu', 'cuda', or 'cuda:0' for specific GPU")

    global hps, device, a

    a = parser.parse_args()
    a.src_path = os.path.join(a.base_path, a.src_path, 'other.npy')
    a.trg_path = os.path.join(a.base_path, a.trg_path, 'other.npy')
    config = os.path.join(a.logs_path, 'config.json')
    hps = utils.get_hparams_from_file(config)
    device = torch.device(a.device)

    inference(a)

if __name__ == '__main__':
    main()
