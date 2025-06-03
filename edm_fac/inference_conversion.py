import argparse
import torch
import os
import numpy as np
from pathlib import Path
import librosa
import soundfile as sf

from audiotools import AudioSignal
from audiotools.core import util
import dac
from utils import yaml_config_hook


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

        print(f"EDM-FAC model loaded on {self.device}")
        print(f"Sample rate: {self.args.sample_rate}")
        print(f"Audio duration original set: {self.audio_length}s")



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

    @torch.no_grad()
    def convert_audio(self, target_audio_path, content_ref_path, timbre_ref_path, output_path):
        """
        Perform audio conversion

        Args:
            target_audio_path: Path to target audio (for reconstruction)
            content_ref_path: Path to content reference audio
            timbre_ref_path: Path to timbre reference audio
            output_path: Path to save converted audio
        """
        # Calculate target length based on duration
        if self.audio_length != 0.0:
            target_length = int(self.audio_length * self.args.sample_rate)
        else:
            leng = librosa.get_duration(path=target_audio_path)
            print(f"Audio Duration: {leng}s")
            target_length = int(leng * self.args.sample_rate)

        # Load audio files
        target_audio = self.load_audio(target_audio_path, target_length)
        content_ref = self.load_audio(content_ref_path, target_length)
        timbre_ref = self.load_audio(timbre_ref_path, target_length)

        # Move to device
        target_audio = target_audio.to(self.device)
        content_ref = content_ref.to(self.device)
        timbre_ref = timbre_ref.to(self.device)


        # Forward pass
        with torch.no_grad():
            out = self.generator(
                audio_data=target_audio.audio_data,
                content_match=content_ref.audio_data,
                timbre_match=timbre_ref.audio_data,
            )

        # Get converted audio
        gt_audio = AudioSignal(target_audio.audio_data.cpu(), self.args.sample_rate)
        converted_audio = AudioSignal(out["audio"].cpu(), self.args.sample_rate)

        # Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gt_audio.write(output_path.replace(".wav", "_gt.wav"))
        converted_audio.write(output_path.replace(".wav", "_recon.wav"))

        print(f"Converted audio saved to: {output_path}")

        # Return additional information
        results = {
            'converted_audio_path': output_path,
            'predicted_timbre_logits': out["pred_timbre_id"].cpu().numpy(),
            'predicted_pitch_logits': out["pred_pitch"].cpu().numpy(),
            'vq_commitment_loss': out["vq/commitment_loss"].item(),
            'vq_codebook_loss': out["vq/codebook_loss"].item(),
        }

        return results



    # TODO: add batch conversion
    @torch.no_grad()
    def batch_convert(self, input_dir, content_ref_path, timbre_ref_path, output_dir):
        """
        Convert multiple audio files in batch

        Args:
            input_dir: Directory containing input audio files
            content_ref_path: Path to content reference audio
            timbre_ref_path: Path to timbre reference audio
            output_dir: Directory to save converted audio files
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Supported audio formats
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac'}

        audio_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in audio_extensions
        ]

        print(f"Found {len(audio_files)} audio files to convert")

        results = []
        for audio_file in audio_files:
            print(f"Converting: {audio_file.name}")

            output_file = output_path / f"converted_{audio_file.stem}.wav"

            try:
                result = self.convert_audio(
                    target_audio_path=str(audio_file),
                    content_ref_path=content_ref_path,
                    timbre_ref_path=timbre_ref_path,
                    output_path=str(output_file)
                )
                result['input_file'] = str(audio_file)
                results.append(result)

            except Exception as e:
                print(f"Error converting {audio_file.name}: {e}")
                continue

        print(f"Batch conversion completed. {len(results)} files converted successfully.")
        return results


def main():
    parser = argparse.ArgumentParser(description="EDM-FAC Inference")

    # Required arguments
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--audio_length", default=0.0, type=float, help="Conversion length")

    # Audio paths
    parser.add_argument("--target_audio", help="Path to target audio")
    parser.add_argument("--content_ref", help="Path to content reference audio")
    parser.add_argument("--timbre_ref", help="Path to timbre reference audio")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--input_dir", help="Input directory for batch mode")
    parser.add_argument("--device", default="cuda", help="Device to use for inference")

    args = parser.parse_args()

    # Initialize inference model
    model = EDMFACInference(
        args.checkpoint,
        args.config,
        args.audio_length,
        args.device
    )


    # Conversion
    results = model.convert_audio(
        args.target_audio,
        args.content_ref,
        args.timbre_ref,
        args.output,
    )

    print("Inference completed successfully!")
    # print(f"Results: {results}")


if __name__ == "__main__":
    main()
