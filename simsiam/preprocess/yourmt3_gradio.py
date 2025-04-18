# @title GradIO helper
import os
import subprocess
import glob
import numpy as np
from typing import Tuple, Dict, Literal
from ctypes import ArgumentError

from pytube import YouTube
import gradio as gr
import torchaudio
from yourmt3_utils import (
    load_model_checkpoint,
    transcribe_get_notes,
    transcribe,
)
from yourmt3_html import (
    to_data_url,
    create_html_from_midi,
    create_html_youtube_player,
)

def prepare_media(source_path_or_url: os.PathLike,
                  source_type: Literal['audio_filepath', 'youtube_url'],
                  delete_video: bool = True) -> Dict:
    """prepare media from source path or youtube, and return audio info"""
    # Get audio_file
    if source_type == 'audio_filepath':
        audio_file = source_path_or_url
    elif source_type == 'youtube_url':
        # Download from youtube
        try:
            # Try PyTube first
            proxy_handler = {"http": "http://127.0.0.1:1087", "https":"http://127.0.0.1:1087"}
            yt = YouTube(source_path_or_url, proxies=proxy_handler)
            # yt = YouTube(source_path_or_url)
            audio_stream = min(yt.streams.filter(only_audio=True), key=lambda s: s.bitrate)
            mp4_file = audio_stream.download(output_path='downloaded') # ./downloaded
            audio_file = mp4_file[:-3] + 'mp3'
            subprocess.run(['ffmpeg', '-i', mp4_file, '-ac', '1', audio_file])
            os.remove(mp4_file)
        except Exception as e:
            try:
                # Try alternative
                print(f"Failed with PyTube, error: {e}. Trying yt-dlp...")
                audio_file = './downloaded/yt_audio'
                subprocess.run(['yt-dlp', '-x', source_path_or_url, '-f', 'bestaudio',
                    '-o', audio_file, '--audio-format', 'mp3', '--restrict-filenames',
                    '--force-overwrites'])
                # subprocess.run(['yt-dlp', '-x', source_path_or_url, '-f', 'bestaudio',
                #     '-o', audio_file, '--audio-format', 'mp3', '--restrict-filenames',
                #     '--force-overwrites', '--cookiefile', '/content/cookies.txt'])
                audio_file += '.mp3'
            except Exception as e:
                print(f"Alternative downloader failed, error: {e}. Please try again later!")
                return None
    else:
        raise ValueError(source_type)

    # Create info
    if audio_file.endswith('.npy'):
        # Load waveform from .npy
        waveform = np.load(audio_file)
        print("Waveform shape:", waveform.shape)
        
        # If shape is (T,), make it (1, T) as mono
        if waveform.shape[0] == 2:
            waveform = np.mean(waveform, axis=0, keepdims=True)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]

        sample_rate = 44100

        return {
            "filepath": audio_file,
            "track_name": os.path.basename(audio_file).split('.')[0],
            "sample_rate": sample_rate,
            "bits_per_sample": 32,  # assuming float32
            "num_channels": waveform.shape[0],
            "num_frames": waveform.shape[1],
            "duration": waveform.shape[1] / sample_rate,
            "encoding": "float32",
            "waveform": waveform  # optional: attach waveform for later
        }

    else:
        # Handle real audio files
        info = torchaudio.info(audio_file)
        return {
            "filepath": audio_file,
            "track_name": os.path.basename(audio_file).split('.')[0],
            "sample_rate": int(info.sample_rate),
            "bits_per_sample": int(info.bits_per_sample),
            "num_channels": int(info.num_channels),
            "num_frames": int(info.num_frames),
            "duration": int(info.num_frames / info.sample_rate),
            "encoding": str.lower(info.encoding),
        }

def process_audio(audio_filepath, model):
    if audio_filepath is None:
        return None
    audio_info = prepare_media(audio_filepath, source_type='audio_filepath')
    midifile = transcribe(model, audio_info)
    midifile = to_data_url(midifile)
    return create_html_from_midi(midifile) # html midiplayer

def process_video(youtube_url, model):
    if 'youtu' not in youtube_url:
        return None
    audio_info = prepare_media(youtube_url, source_type='youtube_url')
    midifile = transcribe(model, audio_info)
    midifile = to_data_url(midifile)
    return create_html_from_midi(midifile) # html midiplayer

def play_video(youtube_url):
    if 'youtu' not in youtube_url:
        return None
    return create_html_youtube_player(youtube_url)
