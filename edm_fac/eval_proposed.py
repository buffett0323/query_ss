import argparse
import json
import torch
import torch.nn as nn
import os
import librosa
import dac
import warnings

from tqdm import tqdm
from torch.utils.data import DataLoader
from audiotools import AudioSignal
from utils import yaml_config_hook
from evaluate_metrics import MultiScaleSTFTLoss, LogRMSEnvelopeLoss
from dataset import EDM_MN_Val_Dataset
from dac.nn.loss import MelSpectrogramLoss, L1Loss

# Filter out specific warnings
warnings.filterwarnings("ignore", message="stft_data changed shape")
warnings.filterwarnings("ignore", message="Audio amplitude > 1 clipped when saving")
LENGTH = 44100*3

class EDMFACInference:
    def __init__(
        self,
        checkpoint_path,
        config_path="configs/config.yaml",
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

        # Get parameters
        self.sample_rate = self.args.sample_rate
        self.hop_length = self.args.hop_length

        # Initialize model
        self.generator = dac.model.MyDAC(
            encoder_dim=self.args.encoder_dim,
            encoder_rates=self.args.encoder_rates,
            latent_dim=self.args.latent_dim,
            decoder_dim=self.args.decoder_dim,
            decoder_rates=self.args.decoder_rates,
            adsr_enc_dim=self.args.adsr_enc_dim,
            adsr_enc_ver=self.args.adsr_enc_ver,
            sample_rate=self.args.sample_rate,
            timbre_classes=self.args.timbre_classes,
            adsr_classes=self.args.adsr_classes,
            pitch_nums=self.args.max_note - self.args.min_note + 1, # 88
            use_gr_content=self.args.use_gr_content,
            use_gr_adsr=self.args.use_gr_adsr,
            use_gr_timbre=self.args.use_gr_timbre,
            use_FiLM=self.args.use_FiLM,
            rule_based_adsr_folding=self.args.rule_based_adsr_folding,
            use_cross_attn=self.args.use_cross_attn,
        ).to(self.device)

        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        self.generator.eval()

        # Load losses for evaluation
        # 1) Multi-scale STFT Loss
        self.stft_loss = MultiScaleSTFTLoss().to(self.device)
        
        # 2) Envelope L1 Loss (Log-RMS envelope)
        self.envelope_loss = LogRMSEnvelopeLoss().to(self.device)
        
        # 3) Mel-Spectrogram Loss (match training settings)
        self.mel_loss = MelSpectrogramLoss(
            n_mels=[5, 10, 20, 40, 80, 160, 320],
            window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
            mel_fmin=[0, 0, 0, 0, 0, 0, 0],
            mel_fmax=[None, None, None, None, None, None, None],
            pow=1.0,
            mag_weight=0.0,
        ).to(self.device)
        
        # 4) L1 waveform loss
        self.l1_eval_loss = L1Loss().to(self.device)

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

    def load_audio(self, audio_path):
        """
        Load and preprocess audio file

        Args:
            audio_path: Path to audio file

        Returns:
            AudioSignal object
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        audio, _ = librosa.load(audio_path, sr=self.args.sample_rate, mono=True)
        audio = audio[:LENGTH]

        # Convert to AudioSignal
        audio_signal = AudioSignal(torch.tensor(audio).unsqueeze(0).unsqueeze(0), self.args.sample_rate)
        return audio_signal


    @torch.no_grad()
    def evaluate_loader(self, data_loader: DataLoader, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        # Aggregators
        conv_types = list(getattr(self.args, "convert_type", ["both", "adsr", "timbre"]))
        sums = {ct: {"stft": 0.0, "l1": 0.0, "mel": 0.0, "env": 0.0, "num": 0} for ct in conv_types}
        overall = {"stft": 0.0, "l1": 0.0, "mel": 0.0, "env": 0.0, "num": 0}

        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move to device
            orig_audio = batch['orig_audio'].to(self.device)
            ref_audio = batch['ref_audio'].to(self.device)

            for ct in conv_types:
                target_audio = batch[f'target_{ct}'].to(self.device)

                out = self.generator.conversion(
                    orig_audio=orig_audio.audio_data,
                    ref_audio=ref_audio.audio_data,
                    convert_type=ct,
                )

                recons = AudioSignal(out["audio"], self.args.sample_rate)

                # Losses
                stft_val = self.stft_loss(recons, target_audio)
                l1_val = self.l1_eval_loss(recons, target_audio)
                mel_val = self.mel_loss(recons, target_audio)
                env_val = self.envelope_loss(recons, target_audio)

                bs = int(out["audio"].shape[0])
                sums[ct]["stft"] += float(stft_val.item()) * bs
                sums[ct]["l1"] += float(l1_val.item()) * bs
                sums[ct]["mel"] += float(mel_val.item()) * bs
                sums[ct]["env"] += float(env_val.item()) * bs
                sums[ct]["num"] += bs

                overall["stft"] += float(stft_val.item()) * bs
                overall["l1"] += float(l1_val.item()) * bs
                overall["mel"] += float(mel_val.item()) * bs
                overall["env"] += float(env_val.item()) * bs
                overall["num"] += bs
                
        # Compute means
        per_type = {}
        for ct, met in sums.items():
            n = max(1, met["num"])
            per_type[ct] = {
                "stft_loss": met["stft"] / n,
                "l1_loss": met["l1"] / n,
                "mel_loss": met["mel"] / n,
                "envelope_loss": met["env"] / n,
                "num_samples": met["num"],
            }

        n_all = max(1, overall["num"])
        results = {
            "num_total_samples": overall["num"],
            "per_convert_type": per_type,
            "overall": {
                "stft_loss": overall["stft"] / n_all,
                "l1_loss": overall["l1"] / n_all,
                "mel_loss": overall["mel"] / n_all,
                "envelope_loss": overall["env"] / n_all,
            },
        }

        # Save metadata immediately here as well
        metadata_path = os.path.join(output_dir, f"metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(results, f, indent=4)

        return results



def main():
    parser = argparse.ArgumentParser(description="EDM-FAC Evaluation on Validation/Test Loader")

    # Arguments
    parser.add_argument("--device", default="cuda", help="Device to use for inference")
    parser.add_argument("--bs", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--output_dir", required=True, help="Output directory for results/metadata")

    # Parse initial arguments to get config path
    initial_args, _ = parser.parse_known_args()
    
    # Load config and add config parameters as arguments
    config = yaml_config_hook(initial_args.config)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    
    # Parse all arguments including config parameters
    args = parser.parse_args()

    # Ibnference Model
    model = EDMFACInference(args.checkpoint, args.config, args.device)

    # Build Evaluation Dataset/Loader from Model Config
    test_dataset = EDM_MN_Val_Dataset(
        root_path=args.root_path,
        midi_path=args.midi_path,
        duration=args.duration,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        split="evaluation",
        perturb_content=args.perturb_content,
        perturb_adsr=args.perturb_adsr,
        perturb_timbre=args.perturb_timbre,
        get_midi_only_from_onset=args.get_midi_only_from_onset,
        mask_delay_frames=args.mask_delay_frames,
        disentanglement_mode=args.disentanglement,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.bs, # args.batch_size
        num_workers=16, # args.num_workers
        collate_fn=test_dataset.collate,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Evaluation batch size: {args.bs}")

    # Perform evaluation over loader and save metadata
    results = model.evaluate_loader(test_loader, args.output_dir)
    print("Evaluation completed!")
    print(f"Output directory: {args.output_dir}")
    print(f"Metadata saved to: {os.path.join(args.output_dir, f'metadata.json')}")
    print(f"Results: {json.dumps(results, indent=4)}")


if __name__ == "__main__":
    main()
