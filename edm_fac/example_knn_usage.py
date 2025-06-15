#!/usr/bin/env python3
"""
Example usage of KNN inference for EDM-FAC model
"""

from knn_infer import EDMFACInference

def main():
    # Initialize the model
    model = EDMFACInference(
        checkpoint_path="path/to/your/checkpoint.pth",
        config_path="configs/config.yaml",
        audio_length=1.0,
        device="cuda"
    )
    
    # Build KNN database from training/validation data
    model.build_knn_database(
        audio_dir="path/to/training/audio",
        midi_dir="path/to/training/midi", 
        max_samples=1000
    )
    
    # Single prediction example
    result = model.knn_predict(
        query_audio_path="path/to/query/audio.wav",
        k=3,
        metric="cosine"
    )
    
    print("=== Top-3 Timbre Predictions ===")
    for pred in result['timbre_predictions']:
        print(f"{pred['rank']}. {pred['timbre_name']} - Confidence: {pred['confidence']:.3f}")
    
    print("\n=== Top-3 Content Predictions ===")
    for pred in result['content_predictions']:
        print(f"{pred['rank']}. MIDI Note {pred['midi_note']} - Confidence: {pred['confidence']:.3f}")
    
    # Batch prediction example
    batch_results = model.knn_batch_predict(
        test_audio_dir="path/to/test/audio",
        output_file="knn_batch_results.json",
        k=3,
        metric="cosine"
    )
    
    print(f"\nProcessed {len(batch_results)} test files")

if __name__ == "__main__":
    main() 