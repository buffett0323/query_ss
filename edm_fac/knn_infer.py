import argparse
import json
import torch
import os
import librosa
import random
import numpy as np
import soundfile as sf
import torch.nn as nn
import pretty_midi
import math
import dac
import matplotlib.pyplot as plt
import warnings

from audiotools import AudioSignal
from audiotools.core import util
from pathlib import Path
from utils import yaml_config_hook
from tqdm import tqdm
from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss, GANLoss, L1Loss

# Filter out specific warnings
warnings.filterwarnings("ignore", message="stft_data changed shape")
warnings.filterwarnings("ignore", message="Audio amplitude > 1 clipped when saving")

class EDMFACInference:
    def __init__(
        self,
        checkpoint_path,
        config_path="configs/config.yaml",
        audio_length=1.0,
        device="cuda",
    ):
        """
        Initialize the EDM-FAC inference model

        Args:
            checkpoint_path: Path to the trained model checkpoint
            config_path: Path to the configuration file
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load configuration
        self.config = yaml_config_hook(config_path)
        self.args = argparse.Namespace(**self.config)
        self.audio_length = audio_length

        # Get parameters
        self.sample_rate = self.args.sample_rate
        self.hop_length = self.args.hop_length
        self.min_note = self.args.min_note
        self.max_note = self.args.max_note
        self.n_notes = self.max_note - self.min_note + 1

        # Initialize model
        self.generator = dac.model.MyDAC(
            encoder_dim=self.args.encoder_dim,
            encoder_rates=self.args.encoder_rates,
            latent_dim=self.args.latent_dim,
            decoder_dim=self.args.decoder_dim,
            decoder_rates=self.args.decoder_rates,
            sample_rate=self.args.sample_rate,
            timbre_classes=self.args.timbre_classes,
            pitch_nums=self.args.max_note - self.args.min_note + 1,
        ).to(self.device)

        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        self.generator.eval()

        # Load losses
        self.stft_loss = MultiScaleSTFTLoss().to(self.device)
        self.mel_loss = MelSpectrogramLoss(
            n_mels=[5, 10, 20, 40, 80, 160, 320],
            window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
            mel_fmin=[0, 0, 0, 0, 0, 0, 0],
            mel_fmax=[None, None, None, None, None, None, None],
            pow=1.0,
            mag_weight=0.0,
        ).to(self.device)
        self.l1_loss = L1Loss().to(self.device)
        # self.gan_loss = GANLoss(discriminator=self.discriminator).to(self.device)

        self.timbre_loss = nn.CrossEntropyLoss().to(self.device)
        self.content_loss = nn.CrossEntropyLoss().to(self.device)

        print(f"EDM-FAC model loaded on {self.device}")



    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'generator_state_dict' in checkpoint:
            # Training checkpoint format with separate state dicts
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            print(f"Loaded checkpoint from step {checkpoint.get('iter', 'unknown')}")
        elif 'generator' in checkpoint:
            # Alternative format
            self.generator.load_state_dict(checkpoint['generator'])
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        elif 'model_state_dict' in checkpoint:
            # Simple format
            self.generator.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the checkpoint contains the model state dict directly
            self.generator.load_state_dict(checkpoint)

        print("Model weights loaded successfully")


    def load_audio(self, audio_path, target_length=None):
        """
        Load and preprocess audio file

        Args:
            audio_path: Path to audio file
            target_length: Target length in samples (optional)

        Returns:
            AudioSignal object
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.args.sample_rate, mono=True)

        # Pad or trim to target length
        if target_length is not None:
            if len(audio) < target_length:
                # Pad with zeros
                audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
            elif len(audio) > target_length:
                # Trim to target length
                audio = audio[:target_length]

        # Convert to AudioSignal
        audio_signal = AudioSignal(torch.tensor(audio).unsqueeze(0).unsqueeze(0), self.args.sample_rate)
        return audio_signal


    def get_midi_to_pitch_sequence(self, midi_path: str, duration: float) -> torch.Tensor:
        """Convert MIDI file to pitch sequence tensor"""
        n_frames = math.ceil(duration * self.sample_rate / self.hop_length)

        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            pitch_sequence = np.zeros((n_frames, self.n_notes))

            for instrument in pm.instruments:
                for note in instrument.notes:
                    start_frame = int(note.start * self.sample_rate / self.hop_length)
                    end_frame = int(note.end * self.sample_rate / self.hop_length)

                    start_frame = max(0, min(start_frame, n_frames-1))
                    end_frame = max(0, min(end_frame, n_frames-1))

                    note_idx = note.pitch - self.min_note

                    if 0 <= note_idx < self.n_notes:
                        pitch_sequence[start_frame:end_frame+1, note_idx] = 1

            return torch.FloatTensor(pitch_sequence)

        except Exception as e:
            print(f"Error processing MIDI file {midi_path}: {e}")
            return torch.zeros((n_frames, self.n_notes))


    def build_knn_database(self, audio_dir, midi_dir, max_samples=1000):
        """
        Build a database of latent representations for KNN search
        
        Args:
            audio_dir: Directory containing audio files
            midi_dir: Directory containing MIDI files
            max_samples: Maximum number of samples to include in database
        """
        print("Building KNN database...")
        
        # Get validation data
        with open("info/evaluation_midi_names_lead_out.txt", "r") as f:
            validation_midi_names = f.read().splitlines()
        with open("info/timbre_names_lead_out.txt", "r") as f:
            timbre_names = f.read().splitlines()
        
        # Collect all audio files
        all_files = []
        for timbre_name in timbre_names:
            for midi_name in validation_midi_names:
                audio_path = os.path.join(audio_dir, f"{timbre_name}_{midi_name}.wav")
                midi_path = os.path.join(midi_dir, f"{midi_name}.mid")
                if os.path.exists(audio_path) and os.path.exists(midi_path):
                    all_files.append((audio_path, midi_path, timbre_name, midi_name))
        
        # Randomly sample if too many files
        if len(all_files) > max_samples:
            all_files = random.sample(all_files, max_samples)
        
        print(f"Processing {len(all_files)} files for KNN database...")
        
        timbre_features = []
        content_features = []
        timbre_labels = []
        content_labels = []
        
        # Map timbre names to indices
        self.timbre_to_idx = {name: idx for idx, name in enumerate(timbre_names)}
        self.idx_to_timbre = {idx: name for name, idx in self.timbre_to_idx.items()}
        
        with torch.no_grad():
            for audio_path, midi_path, timbre_name, midi_name in tqdm(all_files, desc="Extracting features"):
                try:
                    # Load audio
                    target_length = int(self.audio_length * self.args.sample_rate) if self.audio_length > 0 else None
                    audio_signal = self.load_audio(audio_path, target_length)
                    audio_data = audio_signal.to(self.device).audio_data
                    
                    # Extract latent features
                    z = self.generator.encoder(audio_data)
                    
                    # Content features (for pitch prediction)
                    content_z, _, _, _, _ = self.generator.quantizer(z)
                    content_features.append(content_z.cpu().flatten())
                    
                    # Timbre features (processed through transformer)
                    timbre_z = z.transpose(1, 2)
                    timbre_z = self.generator.transformer(timbre_z, None, None)
                    timbre_z = timbre_z.transpose(1, 2)
                    timbre_z_global = torch.mean(timbre_z, dim=2)  # Global mean pooling
                    timbre_features.append(timbre_z_global.cpu().flatten())
                    
                    # Labels
                    timbre_labels.append(self.timbre_to_idx[timbre_name])
                    
                    # Get MIDI content labels (simplified: use first note onset frame)
                    duration = audio_data.shape[-1] / self.args.sample_rate
                    midi_seq = self.get_midi_to_pitch_sequence(midi_path, duration)
                    # Get the most common active pitch as label
                    active_pitches = torch.sum(midi_seq, dim=0)
                    content_label = torch.argmax(active_pitches).item()
                    content_labels.append(content_label)
                    
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
                    continue
        
        # Convert to tensors
        self.timbre_db = torch.stack(timbre_features)
        self.content_db = torch.stack(content_features)
        self.timbre_labels_db = torch.tensor(timbre_labels)
        self.content_labels_db = torch.tensor(content_labels)
        
        print(f"KNN database built with {len(self.timbre_db)} samples")
        print(f"Timbre features shape: {self.timbre_db.shape}")
        print(f"Content features shape: {self.content_db.shape}")
        
    def knn_predict(self, query_audio_path, k=3, metric='cosine'):
        """
        Perform KNN prediction for top-k results
        
        Args:
            query_audio_path: Path to query audio file
            k: Number of nearest neighbors (default: 3)
            metric: Distance metric ('cosine' or 'euclidean')
            
        Returns:
            Dictionary with top-k predictions for timbre and content
        """
        if not hasattr(self, 'timbre_db'):
            raise ValueError("KNN database not built. Call build_knn_database() first.")
        
        print(f"Performing KNN prediction for {query_audio_path}")
        
        with torch.no_grad():
            # Load and process query audio
            target_length = int(self.audio_length * self.args.sample_rate) if self.audio_length > 0 else None
            audio_signal = self.load_audio(query_audio_path, target_length)
            audio_data = audio_signal.to(self.device).audio_data
            
            # Extract query features
            z = self.generator.encoder(audio_data)
            
            # Query content features
            content_z, _, _, _, _ = self.generator.quantizer(z)
            query_content = content_z.cpu().flatten().unsqueeze(0)
            
            # Query timbre features
            timbre_z = z.transpose(1, 2)
            timbre_z = self.generator.transformer(timbre_z, None, None)
            timbre_z = timbre_z.transpose(1, 2)
            timbre_z_global = torch.mean(timbre_z, dim=2)
            query_timbre = timbre_z_global.cpu().flatten().unsqueeze(0)
            
            # Compute distances
            if metric == 'cosine':
                # Cosine similarity
                timbre_sim = torch.nn.functional.cosine_similarity(query_timbre, self.timbre_db, dim=1)
                content_sim = torch.nn.functional.cosine_similarity(query_content, self.content_db, dim=1)
                
                # Convert to distances (higher similarity = lower distance)
                timbre_distances = 1 - timbre_sim
                content_distances = 1 - content_sim
                
            elif metric == 'euclidean':
                # Euclidean distance
                timbre_distances = torch.norm(self.timbre_db - query_timbre, dim=1)
                content_distances = torch.norm(self.content_db - query_content, dim=1)
            
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            # Find top-k nearest neighbors
            timbre_top_k = torch.topk(timbre_distances, k, largest=False)
            content_top_k = torch.topk(content_distances, k, largest=False)
            
            # Get predictions
            timbre_predictions = []
            for i in range(k):
                idx = timbre_top_k.indices[i].item()
                distance = timbre_top_k.values[i].item()
                label = self.timbre_labels_db[idx].item()
                timbre_name = self.idx_to_timbre[label]
                confidence = 1 / (1 + distance)  # Simple confidence score
                
                timbre_predictions.append({
                    'rank': i + 1,
                    'timbre_id': label,
                    'timbre_name': timbre_name,
                    'distance': distance,
                    'confidence': confidence
                })
            
            content_predictions = []
            for i in range(k):
                idx = content_top_k.indices[i].item()
                distance = content_top_k.values[i].item()
                label = self.content_labels_db[idx].item()
                pitch_note = label + self.min_note  # Convert back to MIDI note
                confidence = 1 / (1 + distance)
                
                content_predictions.append({
                    'rank': i + 1,
                    'pitch_id': label,
                    'midi_note': pitch_note,
                    'distance': distance,
                    'confidence': confidence
                })
            
            return {
                'timbre_predictions': timbre_predictions,
                'content_predictions': content_predictions,
                'query_file': query_audio_path
            }
    
    def knn_batch_predict(self, test_audio_dir, output_file, k=3, metric='cosine'):
        """
        Perform KNN prediction on a batch of test files
        
        Args:
            test_audio_dir: Directory containing test audio files
            output_file: Path to save results JSON
            k: Number of nearest neighbors
            metric: Distance metric
        """
        if not hasattr(self, 'timbre_db'):
            raise ValueError("KNN database not built. Call build_knn_database() first.")
            
        # Get test files
        test_files = [f for f in os.listdir(test_audio_dir) if f.endswith('.wav')]
        results = {}
        
        print(f"Running KNN prediction on {len(test_files)} test files...")
        
        for test_file in tqdm(test_files, desc="KNN prediction"):
            test_path = os.path.join(test_audio_dir, test_file)
            try:
                result = self.knn_predict(test_path, k=k, metric=metric)
                results[test_file] = result
            except Exception as e:
                print(f"Error processing {test_file}: {e}")
                continue
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"KNN results saved to {output_file}")
        return results


def main():
    parser = argparse.ArgumentParser(description="EDM-FAC KNN Inference")

    # Required arguments
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--audio_length", default=1.0, type=float, help="Audio length for processing")
    parser.add_argument("--device", default="cuda", help="Device to use for inference")

    # Database arguments
    parser.add_argument("--db_audio_dir", required=True, help="Audio directory for building KNN database")
    parser.add_argument("--db_midi_dir", required=True, help="MIDI directory for building KNN database")
    parser.add_argument("--max_samples", default=1000, type=int, help="Maximum samples in KNN database")

    # Prediction arguments
    parser.add_argument("--mode", default="single", choices=["single", "batch"], help="Prediction mode")
    parser.add_argument("--query_audio", help="Path to query audio file (single mode)")
    parser.add_argument("--test_audio_dir", help="Test audio directory (batch mode)")
    parser.add_argument("--output_file", default="knn_results.json", help="Output file for results")
    parser.add_argument("--k", default=3, type=int, help="Number of nearest neighbors")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"], help="Distance metric")

    args = parser.parse_args()

    # Initialize inference model
    model = EDMFACInference(
        args.checkpoint,
        args.config,
        args.audio_length,
        args.device
    )

    # Build KNN database
    print("Building KNN database...")
    model.build_knn_database(
        args.db_audio_dir,
        args.db_midi_dir,
        args.max_samples
    )
    
    return

    # Perform predictions
    if args.mode == "single":
        if not args.query_audio:
            raise ValueError("--query_audio is required for single mode")
        
        result = model.knn_predict(args.query_audio, k=args.k, metric=args.metric)
        
        print("\n=== KNN Prediction Results ===")
        print(f"Query: {result['query_file']}")
        
        print(f"\nTop-{args.k} Timbre Predictions:")
        for pred in result['timbre_predictions']:
            print(f"  {pred['rank']}. {pred['timbre_name']} (ID: {pred['timbre_id']}) - "
                  f"Distance: {pred['distance']:.4f}, Confidence: {pred['confidence']:.4f}")
        
        print(f"\nTop-{args.k} Content Predictions:")
        for pred in result['content_predictions']:
            print(f"  {pred['rank']}. MIDI Note {pred['midi_note']} (ID: {pred['pitch_id']}) - "
                  f"Distance: {pred['distance']:.4f}, Confidence: {pred['confidence']:.4f}")
        
        # Save single result
        with open(args.output_file, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\nResults saved to {args.output_file}")

    elif args.mode == "batch":
        if not args.test_audio_dir:
            raise ValueError("--test_audio_dir is required for batch mode")
        
        results = model.knn_batch_predict(
            args.test_audio_dir,
            args.output_file,
            k=args.k,
            metric=args.metric
        )
        
        print(f"\nBatch prediction completed on {len(results)} files.")
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
    