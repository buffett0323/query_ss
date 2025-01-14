from torch_models import Wavegram_Logmel128_Cnn14, Wavegram_Logmel_Cnn14
import torch

# Parameters for the model
sample_rate = 16000      # 16 kHz audio sampling rate
window_size = 1024       # FFT window size
hop_size = 320           # Hop length
mel_bins = 64            # Number of mel filter banks
fmin = 20                # Minimum frequency (Hz)
fmax = 8000              # Maximum frequency (Hz)
classes_num = 512       # Number of output classes (e.g., 10 sound categories)

# Initialize the model
model = Wavegram_Logmel_Cnn14(
    sample_rate=sample_rate,
    window_size=window_size,
    hop_size=hop_size,
    mel_bins=mel_bins,
    fmin=fmin,
    fmax=fmax,
    classes_num=classes_num
)

sample_input = torch.randn(16, 32000)
output = model(sample_input)
print(output.shape)