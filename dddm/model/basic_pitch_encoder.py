import pathlib
import numpy as np
import numpy.typing as npt
from typing import Any, Dict, Iterable, Tuple, Union

from basic_pitch.inference import Model
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.constants import (
    AUDIO_SAMPLE_RATE, # 22050
    AUDIO_N_SAMPLES, # 43844
    ANNOTATIONS_FPS, # 86
    FFT_HOP, # 256
)
AUDIO_SAMPLE_RATE = 44100




def window_audio_file(
    audio_original: npt.NDArray[np.float32], hop_size: int
) -> Iterable[Tuple[npt.NDArray[np.float32], Dict[str, float]]]:
    """
    Pad appropriately an audio file, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns:
        audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
        window_times: list of {'start':.., 'end':...} objects (times in seconds)

    """
    for i in range(0, audio_original.shape[0], hop_size):
        window = audio_original[i : i + AUDIO_N_SAMPLES]
        if len(window) < AUDIO_N_SAMPLES:
            window = np.pad(
                window,
                pad_width=[[0, AUDIO_N_SAMPLES - len(window)]],
            )
        t_start = float(i) / AUDIO_SAMPLE_RATE
        window_time = {
            "start": t_start,
            "end": t_start + (AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE),
        }
        yield np.expand_dims(window, axis=-1), window_time





def unwrap_output(
    output: npt.NDArray[np.float32],
    audio_original_length: int,
    n_overlapping_frames: int,
) -> np.array:
    """Unwrap batched model predictions to a single matrix.

    Args:
        output: array (n_batches, n_times_short, n_freqs)
        audio_original_length: length of original audio signal (in samples)
        n_overlapping_frames: number of overlapping frames in the output

    Returns:
        array (n_times, n_freqs)
    """
    if len(output.shape) != 3:
        return None

    n_olap = int(0.5 * n_overlapping_frames)
    if n_olap > 0:
        # remove half of the overlapping frames from beginning and end
        output = output[:, n_olap:-n_olap, :]

    output_shape = output.shape
    n_output_frames_original = int(np.floor(audio_original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
    unwrapped_output = output.reshape(output_shape[0] * output_shape[1], output_shape[2])
    return unwrapped_output[:n_output_frames_original, :]  # trim to original audio length



def run_inference(
    input_array, #: npt.NDArray[np.float32],  # Ensure this is np.float32
    model_or_model_path=Model(ICASSP_2022_MODEL_PATH),
): # -> Dict[str, np.array]:
    """Run the model on the input audio array.

    Args:
        audio_array: The audio array to run inference on.
        model_or_model_path: A loaded Model or path to a serialized model to load.
        debug_file: An optional path to output debug data to. Useful for testing/verification.

    Returns:
       A dictionary with the notes, onsets and contours from model inference.
    """
    if isinstance(model_or_model_path, Model):
        model = model_or_model_path
    else:
        model = Model(model_or_model_path)

    # overlap 30 frames
    n_overlapping_frames = 30
    overlap_len = n_overlapping_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    
    output_note = []
    
    # Ensure the audio array is of type float32
    output: Dict[str, Any] = {"note": [], "onset": [], "contour": []}
    
    for i in range(input_array.shape[0]):
        audio_array = input_array[i]
        audio_original_length = audio_array.shape[-1]
        audio_array = np.concatenate([np.zeros((1, int(overlap_len / 2)), dtype=np.float32), audio_array], axis=1).astype(np.float32)
        for window, _ in window_audio_file(audio_array[0], hop_size):
            for k, v in model.predict(window).items():
                output[k].append(v)

        unwrapped_output = {
            k: unwrap_output(np.concatenate(output[k]), audio_original_length, n_overlapping_frames) for k in output
        }
        output_note.append(unwrapped_output['note'])

    return np.stack(output_note, axis=0)


if __name__ == "__main__":
    basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)
    audio_array = np.random.randn(8, 1, 44100*8) #np.load(audio_path)
    output = run_inference(audio_array, basic_pitch_model)
    
    print(output.shape)