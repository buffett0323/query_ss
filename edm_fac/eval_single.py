from evaluate_metrics import MultiScaleSTFTLoss
from audiotools import AudioSignal
import soundfile as sf

path = "/home/buffett/nas_data/EDM_FAC_LOG/0804_proposed/sample_audio/iter_548000/conv_both"
path1 = f"{path}/01_recon.wav"
path2 = f"{path}/01_gt.wav"

signal1, _ = sf.read(path1)
signal2, _ = sf.read(path2)
signal1 = AudioSignal(signal1, 44100)
signal2 = AudioSignal(signal2, 44100)

stft_loss = MultiScaleSTFTLoss()
stft_val = stft_loss(signal1, signal2)
print(stft_val)
