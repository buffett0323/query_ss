import torch
import torchaudio
import os
import numpy as np
import soundfile as sf
import subprocess
from collections import Counter
from yourmt3_utils import (
    load_model_checkpoint,
    transcribe_get_notes,
)

from yourmt3_gradio import prepare_media
from yourmt3_html import (
    to_data_url,
    create_html_from_midi,
    create_html_youtube_player,
)


if __name__ == "__main__":
    # Step 1: Choose checkpoint
    checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
    args = [
        checkpoint, '-p', '2024', '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
        '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
        '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
        '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', 'bf16-mixed'
    ]
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # Step 2: Load model
    model = load_model_checkpoint(args=args, device=device).to(device)

    # Step 3: Load and process your audio file
    audio_path = "/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_npy/222dbfce-0c68-4d3f-b67a-edc5b62b9810_3/other.npy"
    audio_info = prepare_media(audio_path, source_type='audio_filepath')

    # Load and process npy file
    waveform = np.load(audio_path)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    if waveform.shape[0] == 2:
        waveform = np.mean(waveform, axis=0, keepdims=True)

    sample_rate = 44100
    output_path = "output/temp.wav"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert to torch tensor
    waveform_tensor = torch.tensor(waveform, dtype=torch.float32)

    # Save
    torchaudio.save(output_path, waveform_tensor, sample_rate)
    print(f"Saved waveform to: {output_path}")


    # Step 4: Run transcription and get notes
    notes, midifile = transcribe_get_notes(model, audio_info)
    midifile = to_data_url(midifile)
    html_script = create_html_from_midi(midifile) # html midiplayer


    # Step 5: Print onset/offsets
    note_dicts = [
        {
            "program": note.program,
            "pitch": note.pitch,
            "onset": note.onset,
            "offset": note.offset,
            "velocity": note.velocity,
            "is_drum": note.is_drum
        }
        for note in notes
    ]

    # Print
    for n in note_dicts:
        print(f"Pitch: {n['pitch']}, Onset: {n['onset']:.2f}s, \
              Offset: {n['offset']:.2f}s, Velocity: {n['velocity']}, Program: {n['program']}")
