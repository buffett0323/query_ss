import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def load_audio(file_path):
    wav, sr = librosa.load(file_path)
    return wav, sr


# Compute Mel-Spectrogram
def get_mel_spec(wav, sr=22050, n_mels=128):
    mel_spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels)
    return mel_spec

# Compute Log-Mel Spectrogram
def get_log_mel_spec(wav, sr=22050, n_mels=128):
    mel_spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

def get_cqt(wav, sr=22050, n_bins=84, bins_per_octave=12):
    cqt = librosa.cqt(wav, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave)
    return np.abs(cqt)



