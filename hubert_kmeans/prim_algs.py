import librosa
import essentia

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import essentia.standard as es

from essentia.standard import *
from tempfile import TemporaryDirectory

# print(dir(essentia.standard))


def onset_detection(file_path, mono=True):
    
    if mono:
        audio = es.MonoLoader(filename=file_path)()
    else:
        audio = es.AudioLoader(filename=file_path)()[0]
    
    # 1. Compute the onset detection function (ODF).
    # The OnsetDetection algorithm provides various ODFs.
    od_hfc = OnsetDetection(method='hfc')
    od_complex = OnsetDetection(method='complex')

    # We need the auxilary algorithms to compute magnitude and phase.
    w = Windowing(type='hann')
    fft = FFT() # Outputs a complex FFT vector.
    c2p = CartesianToPolar() # Converts it into a pair of magnitude and phase vectors.

    # Compute both ODF frame by frame. Store results to a Pool.
    pool = essentia.Pool()
    for frame in FrameGenerator(audio, frameSize=1024, hopSize=512):
        magnitude, phase = c2p(fft(w(frame)))
        pool.add('odf.hfc', od_hfc(magnitude, phase))
        pool.add('odf.complex', od_complex(magnitude, phase))

    # 2. Detect onset locations.
    onsets = Onsets()

    onsets_hfc = onsets(essentia.array([pool['odf.hfc']]), [1])
    onsets_complex = onsets(essentia.array([pool['odf.complex']]), [1])
    
    return onsets_hfc, onsets_complex
    
    
    
def envelop_detection(file_path, sr=44100, plotting=False):
    y, sr = librosa.load(file_path, sr=sr)

    # Compute Hilbert Transform Envelope
    analytic_signal = signal.hilbert(y)
    envelope = np.abs(analytic_signal)
    
    # Time axis
    time_axis = np.linspace(0, len(y)/sr, num=len(y))

    # Plot envelope extraction
    if plotting:
        plt.figure(figsize=(12, 5))
        plt.plot(time_axis, y, alpha=0.5, label="Original Waveform")
        plt.plot(time_axis, envelope, label="Hilbert Envelope", color="red")
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Envelope Extraction (Hilbert Transform & RMS)")
        plt.show()
    
    return envelope
    
if __name__ == "__main__":
    file_path = '/mnt/gestalt/database/beatport/audio/audio/electro-house/ff527b1b-305f-4893-b8aa-0dd84b03c2ac.mp3'
    