import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
# Load the Wavegram_Logmel_Cnn model from torch.hub
model = torch.hub.load('qiuqiangkong/panns_inference', 'Wavegram_Logmel_Cnn14', pretrained=True)

# Remove the classification head for feature extraction
model.fc_audioset = nn.Identity()

# Send to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print(model)

# Generate a dummy mel-spectrogram (Batch Size=16, Channels=1, Height=128, Width=128)
batch_size = 16
n_mels = 128
time_frames = 128
dummy_input = torch.randn(batch_size, 1, n_mels, time_frames).to(device)  # Log-mel spectrogram

# Alternatively, preprocess real audio
# waveform: [Batch Size, Samples] - Raw audio waveform
sample_rate = 16000
mel_transform = MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=2048,
    hop_length=512,
    n_mels=n_mels
)
amplitude_to_db = AmplitudeToDB()

waveform = torch.randn(batch_size, sample_rate * 4)  # Example 4-second audio
mel_spectrogram = amplitude_to_db(mel_transform(waveform)).unsqueeze(1).to(device)  # Add channel dimension

# Pass the input through the model
output_embeddings = model(mel_spectrogram)  # Shape: [Batch Size, Feature Dimension]

print(f"Input Shape: {mel_spectrogram.shape}")  # torch.Size([16, 1, 128, 128])
print(f"Output Shape: {output_embeddings.shape}")  # torch.Size([16, 2048])
