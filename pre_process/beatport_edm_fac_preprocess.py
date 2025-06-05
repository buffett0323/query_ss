import os
import json
from tqdm import tqdm
from audiotools.core import AudioSignal
import multiprocessing as mp
from functools import partial

# chorus_path = "/mnt/gestalt/home/buffett/beatport_original/chorus" # 136800
htdemucs_path = "/mnt/gestalt/home/buffett/beatport_original/htdemucs" # 74979
json_path = "/mnt/gestalt/home/buffett/beatport_original/json" # 74979
output_path = "/mnt/gestalt/home/buffett/beatport_analyze"
new_folder_name = "chorus_audio_44100_wav_other_new"
PROCESS_TIME = 10
SAMPLE_RATE = 44100


def extract_chorus(song_path, chorus_start_list, chorus_end_list, song_name):
    # Mix the bass and other tracks together
    for i, (chorus_start, chorus_end) in enumerate(zip(chorus_start_list, chorus_end_list)):
        # bass_path = os.path.join(song_path, "bass.wav")
        other_path = os.path.join(song_path, "other.wav")
        
        # bass_audio = AudioSignal(
        #     bass_path, 
        #     offset=chorus_start, 
        #     duration=chorus_end - chorus_start,
        #     sample_rate=SAMPLE_RATE
        # )
        other_audio = AudioSignal(
            other_path, 
            offset=chorus_start, 
            duration=chorus_end - chorus_start,
            sample_rate=SAMPLE_RATE
        )
        
        # # Scale each track more aggressively before mixing to prevent clipping
        # bass_audio = bass_audio * 0.4
        # other_audio = other_audio * 0.4
        # mixed_audio = bass_audio + other_audio
        
        # # Normalize with significant headroom to prevent clipping
        # mixed_audio = mixed_audio.normalize(-6.0)
        
        # # Additional safety check: ensure no sample exceeds 0.95 to prevent clipping
        # peak_amplitude = mixed_audio.audio_data.abs().max()
        # if peak_amplitude > 0.95:
        #     safety_scale = 0.95 / peak_amplitude
        #     mixed_audio = mixed_audio * safety_scale
        
        other_audio.write(
            os.path.join(
                output_path, 
                new_folder_name, 
                f"{song_name}_chorus_other_{i}.wav"
            )
        )


def process_single_song(song_name):
    """Process a single song and return the number of chorus segments processed and individual segment durations"""
    try:
        song_path = os.path.join(htdemucs_path, song_name)
        json_file = os.path.join(json_path, f"{song_name}.json")
        
        # Check if files exist
        if not os.path.exists(json_file):
            return 0, []
        if not os.path.exists(song_path):
            return 0, []
            
        with open(json_file, "r") as f:
            data = json.load(f)
        
        chorus_start_list, chorus_end_list = [], []
        segment_durations = []
        
        for segment in data["segments"]:
            if segment["label"] == "chorus":
                chorus_start = segment["start"]
                chorus_end = segment["end"]
                segment_duration = chorus_end - chorus_start
                if segment_duration > PROCESS_TIME:
                    chorus_start_list.append(chorus_start)
                    chorus_end_list.append(chorus_end)
                    segment_durations.append(segment_duration)
        
        if chorus_start_list:
            extract_chorus(song_path, chorus_start_list, chorus_end_list, song_name)
        
        return len(chorus_start_list), segment_durations
    
    except Exception as e:
        print(f"Error processing {song_name}: {e}")
        return 0, []



if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(output_path, new_folder_name), exist_ok=True)
    
    # Get all song names
    song_names = [name for name in os.listdir(htdemucs_path)]
    total_songs = len(song_names)
    
    # Use multiprocessing to process songs in parallel
    num_processes = mp.cpu_count()-1  # Limit to 16 processes to avoid overwhelming the system
    print(f"Using {num_processes} processes for parallel processing")
    print(f"Total songs to process: {total_songs}")
    
    record_dict = {}
    total_counter = 0
    songs_processed = 0
    
    
    
    with mp.Pool(processes=num_processes) as pool:
        # Process songs in batches to allow for early stopping when counter >= 20
        batch_size = num_processes * 2
        
        for i in range(0, len(song_names), batch_size):
            batch = song_names[i:i + batch_size]
            
            # Process batch in parallel
            results = []
            for song_name in batch:
                result = pool.apply_async(process_single_song, (song_name,))
                results.append((song_name, result))
            
            # Collect results
            for song_name, result in results:
                try:
                    chorus_count, segment_durations = result.get(timeout=300)  # 5 minute timeout per song
                    total_counter += chorus_count
                    songs_processed += 1
                    
                    if chorus_count > 0:
                        record_dict[song_name] = segment_durations
                        durations_str = ", ".join([f"{d:.2f}s" for d in segment_durations])
                        print(f"✓ Progress: {songs_processed}/{total_songs}, This song: {chorus_count} chorus segments, Total segments: {total_counter}")
                    else:
                        print(f"- Progress: {songs_processed}/{total_songs}, This song: 0 chorus segments, Total segments: {total_counter}")
                        
                except Exception as e:
                    songs_processed += 1
                    print(f"✗ Failed to process {song_name}: {e} (Progress: {songs_processed}/{total_songs})")
    
    
    
    print(f"\n=== Processing Complete ===")
    print(f"Total chorus segments processed: {total_counter}")
    print(f"Total songs processed: {songs_processed}/{total_songs}")
    print(f"Songs with chorus segments: {len(record_dict)}")
    
    with open(f"{output_path}/segment_chorus_duration.json", "w") as f:
        json.dump(record_dict, f, indent=4)
