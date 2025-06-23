import multiprocessing

# Audio-specific constants
SAMPLE_RATE = 44100
MAX_SECONDS = 2.97
MAX_AUDIO_LENGTH = int(MAX_SECONDS * SAMPLE_RATE)
N_MELS = 128
HOP_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 1000
LR = 3e-4
NUM_GPUS = 1
NUM_WORKERS = multiprocessing.cpu_count()
