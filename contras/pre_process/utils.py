from pydub import AudioSegment


def detect_large_file(file_path, minutes=5):
    audio = AudioSegment.from_mp3(file_path)
    
    # Get duration in seconds
    duration_seconds = len(audio) / 1000
    if duration_seconds < (minutes * 60): 
        return True
    return False