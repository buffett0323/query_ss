from pydub import AudioSegment
import os

def detect_large_file(file_path, minutes=5):
    audio = AudioSegment.from_mp3(file_path)

    # Get duration in seconds
    duration_seconds = len(audio) / 1000
    if duration_seconds < (minutes * 60):
        return True
    return False

def get_size(file_path, siz=20):
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert bytes to MB
    if file_size_mb > 20:
        return True
    return False
