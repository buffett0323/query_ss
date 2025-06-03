import os
import librosa
import numpy as np
import scipy.signal
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count
from functools import partial
import pretty_midi


SAMPLE_RATE = 44100
STEMS = ['lead'] #['lead', 'pad', 'bass', 'keys', 'pluck']
FILTER_TIME = 8.0
path = "/mnt/gestalt/home/buffett/EDM_FAC_DATA/rendered_audio_new"


def process_file(file_info, path):
    stem, file = file_info
    file_name = file.split(".wav")[0]
    if file.endswith(".wav"):
        audio, _ = librosa.load(os.path.join(path, stem, file), sr=None)
        audio = audio[:int(FILTER_TIME * SAMPLE_RATE)]

        # Find indices where amplitude exceeds threshold
        envelope = np.abs(audio)
        peaks, _ = scipy.signal.find_peaks(envelope, distance=SAMPLE_RATE // 20)
        if len(peaks) == 0: return (file_name, [])

        # Get peak amplitudes
        peak_amplitudes = envelope[peaks]

        # Get top 10 peaks by amplitude
        if len(peaks) > 10:
            top_indices = np.argsort(peak_amplitudes)[-10:]
            peaks = peaks[top_indices]
            peak_amplitudes = peak_amplitudes[top_indices]

        # Convert peak indices to time and filter peaks > 8 seconds
        peak_info = [peak_idx / SAMPLE_RATE for peak_idx in peaks]

        return (file_name, peak_info) if peak_info else (file_name, [])



def get_onset_from_midi(midi_file):
    """
    Extract onset times from a MIDI file, filtering to only include onsets before 8 seconds.

    Args:
        midi_file: Path to the MIDI file

    Returns:
        onset_times: List of onset times in seconds before 8 seconds
    """
    try:
        # Load MIDI file
        midi_data = pretty_midi.PrettyMIDI(midi_file)

        # Collect all note onset times
        onset_times = []
        n_count = 0
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                n_count += 1
                if note.start <= FILTER_TIME:
                    onset_times.append(note.start)

        return sorted(list(set(onset_times)))

    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return []



if __name__ == "__main__":
    train_path = "/home/buffett/dataset/EDM_FAC_DATA/single_note_midi/train/midi"
    eval_path = "/home/buffett/dataset/EDM_FAC_DATA/single_note_midi/evaluation/midi"
    train_midis = []
    eval_midis = []

    with open("info/train_midi_names.txt", "r") as f:
        for line in f:
            train_midis.append(os.path.join(train_path, line.strip() + ".mid"))

    with open("info/evaluation_midi_names.txt", "r") as f:
        for line in f:
            eval_midis.append(os.path.join(eval_path, line.strip() + ".mid"))

    onset_records_train = {}
    onset_records_eval = {}

    for midi_file in tqdm(train_midis):
        m = midi_file.split("/")[-1].split(".mid")[0]
        onset_times = get_onset_from_midi(midi_file)
        onset_records_train[m] = onset_times

    for midi_file in tqdm(eval_midis):
        m = midi_file.split("/")[-1].split(".mid")[0]
        onset_times = get_onset_from_midi(midi_file)
        onset_records_eval[m] = onset_times

    with open("/home/buffett/dataset/EDM_FAC_DATA/onset_records_lead_train.json", "w") as f:
        json.dump(onset_records_train, f, indent=4)

    with open("/home/buffett/dataset/EDM_FAC_DATA/onset_records_lead_evaluation.json", "w") as f:
        json.dump(onset_records_eval, f, indent=4)



    # # Create list of all files to process
    # files_to_process = []
    # for stem in STEMS:
    #     list_files = os.listdir(os.path.join(path, stem))
    #     files_to_process.extend([(stem, file) for file in list_files])

    # # files_to_process = files_to_process[:10]
    # print(len(files_to_process))

    # # Process files in parallel using 24 processes
    # process_with_path = partial(process_file, path=path)
    # num_processes = cpu_count() - 1
    # with Pool(num_processes) as pool:
    #     results = list(tqdm(
    #         pool.imap(process_with_path, files_to_process),
    #         total=len(files_to_process),
    #         desc="Processing files"
    #     ))

    # # Filter out None results and separate files with no peaks
    # no_peak_files = []
    # peak_records = {}
    # for result in results:
    #     if len(result[1]) == 0:
    #         no_peak_files.append(result[0])
    #     else:
    #         file_name, peak_info = result
    #         peak_records[file_name] = peak_info

    # print(f"Files with no peaks: {len(no_peak_files)}")
    # if no_peak_files:
    #     with open("info/no_peak_files.txt", "w") as f:
    #         for file in no_peak_files:
    #             f.write(file + "\n")


    # # Save results
    # with open("/mnt/gestalt/home/buffett/EDM_FAC_DATA/peak_records_lead.json", "w") as f:
    #     json.dump(peak_records, f, indent=4)
