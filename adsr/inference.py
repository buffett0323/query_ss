#!/usr/bin/env python3
"""
BYOL-A Inference Script for ADSR Audio Representation Learning

This script loads a trained BYOL-A checkpoint and extracts audio representations
from input audio files or mel spectrograms, with KNN inference capabilities.

Usage:
    python inference.py --ckpt_path /path/to/checkpoint.ckpt --input_audio /path/to/audio.wav
    python inference.py --ckpt_path /path/to/checkpoint.ckpt --input_mel /path/to/mel.npy
    python inference.py --ckpt_path /path/to/checkpoint.ckpt --input_dir /path/to/audio/directory
    python inference.py --ckpt_path /path/to/checkpoint.ckpt --knn_inference --query_audio /path/to/query.wav --reference_dir /path/to/reference/audio
"""

import argparse
import torch
import torchaudio
import numpy as np
import h5py
from pathlib import Path
from typing import Union, List, Tuple
import nnAudio.features
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import json

# Import BYOL-A modules
from byol_a2.common import load_yaml_config
from byol_a2.models import AudioNTT2022, load_pretrained_weights
from byol_a2.augmentations import PrecomputedNorm, NormalizeBatch
from byol_a2.byol_pytorch import BYOL

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BYOLAInference:
    """BYOL-A inference class for extracting audio representations with KNN inference."""

    def __init__(
        self,
        ckpt_path: str,
        config_path: str = "config_v2.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mel_mean: float = -12.621428,
        mel_std: float = 5.243587,
    ):
        """
        Initialize BYOL-A inference model.

        Args:
            ckpt_path: Path to the trained checkpoint file
            config_path: Path to the configuration file
            device: Device to run inference on
            mel_mean: Pre-computed mel spectrogram mean for normalization
            mel_std: Pre-computed mel spectrogram std for normalization
        """
        self.device = torch.device(device)
        self.ckpt_path = ckpt_path
        self.config_path = config_path

        # Load configuration
        self.cfg = load_yaml_config(config_path)

        # Initialize mel spectrogram converter
        self.mel_converter = nnAudio.features.MelSpectrogram(
            sr=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            win_length=self.cfg.win_length,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
            fmin=self.cfg.f_min,
            fmax=self.cfg.f_max,
            center=True,
            power=2,
            verbose=False,
        ).to(self.device)

        # Initialize normalization
        self.pre_norm = PrecomputedNorm(np.array([mel_mean, mel_std]))
        self.post_norm = NormalizeBatch()

        # Load model
        self.model, self.learner = self._load_model()

        logger.info(f"BYOL-A inference model loaded successfully on {self.device}")
        logger.info(f"Model feature dimension: {self.cfg.feature_d}")

    def _load_model(self) -> Tuple[AudioNTT2022, BYOL]:
        """Load the trained BYOL-A model from checkpoint."""
        # Initialize base model
        model = AudioNTT2022(
            n_mels=self.cfg.n_mels,
            d=self.cfg.feature_d
        )

        # Initialize BYOL learner
        learner = BYOL(
            model,
            image_size=self.cfg.shape,
            hidden_layer=-1,
            projection_size=self.cfg.proj_size,
            projection_hidden_size=self.cfg.proj_dim,
            moving_average_decay=self.cfg.ema_decay,
        )

        # Load checkpoint
        if self.ckpt_path.endswith('.ckpt'):
            # PyTorch Lightning checkpoint
            checkpoint = torch.load(self.ckpt_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                # Load the full state dict into the learner
                learner.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                learner.load_state_dict(checkpoint, strict=False)
        else:
            # Direct model weights
            load_pretrained_weights(model, self.ckpt_path)

        model.to(self.device)
        learner.to(self.device)
        model.eval()
        learner.eval()

        logger.info(f"Model loaded from {self.ckpt_path}")
        return model, learner

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Preprocess audio file to mel spectrogram.

        Args:
            audio_path: Path to audio file

        Returns:
            Preprocessed mel spectrogram tensor
        """
        # Load audio
        wav, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.cfg.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.cfg.sample_rate)
            wav = resampler(wav)

        # Pad or truncate to unit length
        unit_samples = int(self.cfg.unit_sec * self.cfg.sample_rate)
        if wav.shape[1] < unit_samples:
            # Pad with zeros
            padding = unit_samples - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, padding))
        elif wav.shape[1] > unit_samples:
            # Truncate
            wav = wav[:, :unit_samples]

        # Convert to mel spectrogram
        mel = self.mel_converter(wav.to(self.device))

        # Apply log and normalization
        eps = torch.finfo(mel.dtype).eps
        lms = (mel + eps).log()
        lms = self.pre_norm(lms)
        lms = self.post_norm(lms)

        return lms

    def preprocess_mel(self, mel_path: str) -> torch.Tensor:
        """
        Preprocess pre-computed mel spectrogram.

        Args:
            mel_path: Path to mel spectrogram file (.npy)

        Returns:
            Preprocessed mel spectrogram tensor
        """
        # Load mel spectrogram
        mel = np.load(mel_path)
        mel = torch.from_numpy(mel).float()

        # Add channel dimension if needed
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)  # [F, T] -> [1, F, T]

        # Ensure correct shape
        if mel.shape[1] != self.cfg.n_mels:
            raise ValueError(f"Expected {self.cfg.n_mels} mel bins, got {mel.shape[1]}")

        # Apply log and normalization
        eps = torch.finfo(mel.dtype).eps
        lms = (mel + eps).log()
        lms = self.pre_norm(lms)
        lms = self.post_norm(lms)

        return lms.to(self.device)

    def extract_features(self, input_data: Union[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from input audio or mel spectrogram using BYOL online encoder.

        Args:
            input_data: Path to audio file, mel file, or preprocessed tensor

        Returns:
            Extracted features tensor [feature_dim]
        """
        with torch.no_grad():
            if isinstance(input_data, str):
                if input_data.endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    # Audio file
                    lms = self.preprocess_audio(input_data)
                elif input_data.endswith('.npy'):
                    # Mel spectrogram file
                    lms = self.preprocess_mel(input_data)
                else:
                    raise ValueError(f"Unsupported file format: {input_data}")
            else:
                # Preprocessed tensor
                lms = input_data.to(self.device)

            # Extract features using BYOL online encoder
            features = self.learner.online_encoder(lms)

            return features

    def extract_features_batch(
        self,
        input_paths: List[str],
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Extract features from multiple files in batches.

        Args:
            input_paths: List of input file paths
            batch_size: Batch size for processing

        Returns:
            Extracted features tensor [num_files, feature_dim]
        """
        all_features = []

        for i in range(0, len(input_paths), batch_size):
            batch_paths = input_paths[i:i + batch_size]
            batch_features = []

            for path in batch_paths:
                features = self.extract_features(path)
                batch_features.append(features)

            batch_features = torch.stack(batch_features)
            all_features.append(batch_features)

            logger.info(f"Processed batch {i//batch_size + 1}/{(len(input_paths) + batch_size - 1)//batch_size}")

        return torch.cat(all_features, dim=0)

    def build_reference_database(
        self,
        reference_dir: str,
        file_extensions: List[str] = [".wav", ".mp3", ".flac", ".m4a"],
        batch_size: int = 32
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build a reference database from a directory of audio files.

        Args:
            reference_dir: Directory containing reference audio files
            file_extensions: Audio file extensions to process
            batch_size: Batch size for processing

        Returns:
            Tuple of (features_array, file_paths_list)
        """
        reference_path = Path(reference_dir)
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference directory not found: {reference_dir}")

        # Find all audio files
        audio_files = []
        for ext in file_extensions:
            audio_files.extend(reference_path.glob(f"*{ext}"))

        if not audio_files:
            raise ValueError(f"No audio files found in {reference_dir}")

        logger.info(f"Found {len(audio_files)} reference audio files")

        # Extract features
        features = self.extract_features_batch(
            [str(f) for f in audio_files],
            batch_size=batch_size
        )

        return features.cpu().numpy(), [str(f) for f in audio_files]

    def knn_inference(
        self,
        query_features: torch.Tensor,
        reference_features: np.ndarray,
        reference_paths: List[str],
        k: int = 5,
        metric: str = 'cosine'
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Perform KNN inference to find similar audio samples.

        Args:
            query_features: Query features tensor [feature_dim]
            reference_features: Reference features array [num_references, feature_dim]
            reference_paths: List of reference file paths
            k: Number of nearest neighbors to return
            metric: Distance metric ('cosine', 'euclidean', etc.)

        Returns:
            Tuple of (distances, indices, similar_paths)
        """
        # Convert query features to numpy
        query_np = query_features.cpu().numpy().reshape(1, -1)

        # Initialize KNN
        if metric == 'cosine':
            # For cosine similarity, we need to normalize features
            query_norm = query_np / np.linalg.norm(query_np, axis=1, keepdims=True)
            ref_norm = reference_features / np.linalg.norm(reference_features, axis=1, keepdims=True)

            # Compute cosine similarities
            similarities = cosine_similarity(query_norm, ref_norm)[0]

            # Get top-k indices (highest similarities)
            indices = np.argsort(similarities)[::-1][:k]
            distances = similarities[indices]

        else:
            # Use sklearn KNN for other metrics
            knn = NearestNeighbors(n_neighbors=k, metric=metric)
            knn.fit(reference_features)

            distances, indices = knn.kneighbors(query_np)
            distances = distances[0]
            indices = indices[0]

        # Get corresponding file paths
        similar_paths = [reference_paths[i] for i in indices]

        return distances, indices, similar_paths

    def compute_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between two feature vectors.

        Args:
            features1: First feature vector [feature_dim] or [batch_size, feature_dim]
            features2: Second feature vector [feature_dim] or [batch_size, feature_dim]

        Returns:
            Cosine similarity score(s)
        """
        # Normalize features
        features1_norm = torch.nn.functional.normalize(features1, dim=-1)
        features2_norm = torch.nn.functional.normalize(features2, dim=-1)

        # Compute cosine similarity
        similarity = torch.sum(features1_norm * features2_norm, dim=-1)

        return similarity

    def save_features(self, features: torch.Tensor, output_path: str):
        """Save extracted features to file."""
        features_np = features.cpu().numpy()
        np.save(output_path, features_np)
        logger.info(f"Features saved to {output_path}")

    def save_reference_database(
        self,
        features: np.ndarray,
        file_paths: List[str],
        output_path: str
    ):
        """Save reference database to file."""
        database = {
            'features': features,
            'file_paths': file_paths,
            'num_samples': len(features),
            'feature_dim': features.shape[1]
        }
        np.savez(output_path, **database)
        logger.info(f"Reference database saved to {output_path}")

    def load_reference_database(self, database_path: str) -> Tuple[np.ndarray, List[str]]:
        """Load reference database from file."""
        data = np.load(database_path, allow_pickle=True)
        features = data['features']
        file_paths = data['file_paths'].tolist()
        logger.info(f"Loaded reference database with {len(features)} samples")
        return features, file_paths


def main():
    parser = argparse.ArgumentParser(description="BYOL-A Inference for Audio Representation with KNN")
    parser.add_argument("--ckpt_path", type=str, required=True,
                       help="Path to the trained checkpoint file")
    parser.add_argument("--config_path", type=str, default="config_v2.yaml",
                       help="Path to the configuration file")
    parser.add_argument("--input_audio", type=str,
                       help="Path to input audio file")
    parser.add_argument("--input_mel", type=str,
                       help="Path to input mel spectrogram file (.npy)")
    parser.add_argument("--input_dir", type=str,
                       help="Directory containing audio files to process")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory for extracted features")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run inference on (auto, cuda, cpu)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing multiple files")
    parser.add_argument("--file_extensions", nargs="+",
                       default=[".wav", ".mp3", ".flac", ".m4a"],
                       help="Audio file extensions to process")

    # KNN inference arguments
    parser.add_argument("--knn_inference", action="store_true",
                       help="Perform KNN inference")
    parser.add_argument("--query_audio", type=str,
                       help="Query audio file for KNN inference")
    parser.add_argument("--reference_dir", type=str,
                       help="Directory containing reference audio files")
    parser.add_argument("--reference_database", type=str,
                       help="Path to pre-built reference database (.npz)")
    parser.add_argument("--k", type=int, default=5,
                       help="Number of nearest neighbors to return")
    parser.add_argument("--metric", type=str, default="cosine",
                       choices=["cosine", "euclidean", "manhattan"],
                       help="Distance metric for KNN")
    parser.add_argument("--save_reference_db", type=str,
                       help="Save reference database to specified path")

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Initialize inference model
    inference = BYOLAInference(
        ckpt_path=args.ckpt_path,
        config_path=args.config_path,
        device=device
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # KNN inference
    if args.knn_inference:
        if not args.query_audio:
            logger.error("--query_audio is required for KNN inference")
            return

        if not args.reference_dir and not args.reference_database:
            logger.error("Either --reference_dir or --reference_database is required for KNN inference")
            return

        # Load or build reference database
        if args.reference_database:
            reference_features, reference_paths = inference.load_reference_database(args.reference_database)
        else:
            logger.info("Building reference database...")
            reference_features, reference_paths = inference.build_reference_database(
                args.reference_dir,
                args.file_extensions,
                args.batch_size
            )

            # Save reference database if requested
            if args.save_reference_db:
                inference.save_reference_database(
                    reference_features,
                    reference_paths,
                    args.save_reference_db
                )

        # Extract query features
        logger.info(f"Extracting features from query audio: {args.query_audio}")
        query_features = inference.extract_features(args.query_audio)

        # Perform KNN inference
        logger.info(f"Performing KNN inference with k={args.k}, metric={args.metric}")
        distances, indices, similar_paths = inference.knn_inference(
            query_features,
            reference_features,
            reference_paths,
            k=args.k,
            metric=args.metric
        )

        # Save results
        results = {
            'query_audio': args.query_audio,
            'k': args.k,
            'metric': args.metric,
            'similar_audio_files': similar_paths,
            'similarities': distances.tolist() if args.metric == 'cosine' else distances.tolist(),
            'indices': indices.tolist()
        }

        results_path = output_dir / "knn_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Print results
        logger.info("KNN Inference Results:")
        logger.info(f"Query: {args.query_audio}")
        for i, (path, sim) in enumerate(zip(similar_paths, distances)):
            logger.info(f"  {i+1}. {Path(path).name} (similarity: {sim:.4f})")

        logger.info(f"Results saved to: {results_path}")
        return

    # Process single file
    if args.input_audio:
        logger.info(f"Processing single audio file: {args.input_audio}")
        features = inference.extract_features(args.input_audio)
        output_path = output_dir / f"{Path(args.input_audio).stem}_features.npy"
        inference.save_features(features, str(output_path))
        logger.info(f"Feature dimension: {features.shape}")
        logger.info(f"Features saved to: {output_path}")

    elif args.input_mel:
        logger.info(f"Processing single mel file: {args.input_mel}")
        features = inference.extract_features(args.input_mel)
        output_path = output_dir / f"{Path(args.input_mel).stem}_features.npy"
        inference.save_features(features, str(output_path))
        logger.info(f"Feature dimension: {features.shape}")
        logger.info(f"Features saved to: {output_path}")

    # Process directory
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find all audio files
        audio_files = []
        for ext in args.file_extensions:
            audio_files.extend(input_dir.glob(f"*{ext}"))

        if not audio_files:
            logger.warning(f"No audio files found in {input_dir}")
            return

        logger.info(f"Found {len(audio_files)} audio files to process")

        # Extract features in batches
        features = inference.extract_features_batch(
            [str(f) for f in audio_files],
            batch_size=args.batch_size
        )

        # Save features
        output_path = output_dir / "batch_features.npy"
        inference.save_features(features, str(output_path))

        # Save file mapping
        file_mapping = {i: str(f) for i, f in enumerate(audio_files)}
        mapping_path = output_dir / "file_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(file_mapping, f, indent=2)

        logger.info(f"Processed {len(audio_files)} files")
        logger.info(f"Feature dimensions: {features.shape}")
        logger.info(f"Features saved to: {output_path}")
        logger.info(f"File mapping saved to: {mapping_path}")

    else:
        logger.error("Please provide either --input_audio, --input_mel, --input_dir, or --knn_inference")
        return


if __name__ == "__main__":
    main()
