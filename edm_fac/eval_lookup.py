import json
import os

path = "/home/buffett/nas_data/EDM_FAC_LOG/demo_website"
amount = 20


with open(os.path.join(path, "metadata.json"), "r") as f:
    metadata = json.load(f)

print(f"Total entries: {len(metadata)}")

# Sort by STFT loss (ascending) and get the 20 lowest
sorted_metadata = sorted(metadata, key=lambda x: x['stft_loss'])
lowest = sorted_metadata[:amount]

print(f"\nTop {amount} lowest STFT loss entries:")
print("=" * 80)

for i, entry in enumerate(lowest, 1):
    print(f"{i:2d}. STFT Loss: {entry['stft_loss']:.4f}, Envelope Loss: {entry['envelope_loss']:.4f}")
    print(f"    Original: T{entry['name']['orig_audio']['timbre_id']:03d}_ADSR{entry['name']['orig_audio']['adsr_id']:03d}_C{entry['name']['orig_audio']['content_id']:03d}")
    print(f"    Reference: T{entry['name']['ref_audio']['timbre_id']:03d}_ADSR{entry['name']['ref_audio']['adsr_id']:03d}_C{entry['name']['ref_audio']['content_id']:03d}")
    print(f"    Target: T{entry['name']['target_both']['timbre_id']:03d}_ADSR{entry['name']['target_both']['adsr_id']:03d}_C{entry['name']['target_both']['content_id']:03d}")
    print()