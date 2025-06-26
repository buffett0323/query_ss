import argparse
import os
import random
import warnings
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
import nnAudio.features

from byol_a2.common import (np, Path, torch,
     get_logger, load_yaml_config, seed_everything, get_timestamp, hash_text)
from byol_a2.byol_pytorch import BYOL
from byol_a2.models import AudioNTT2022, load_pretrained_weights
from byol_a2.augmentations import NormalizeBatch, PrecomputedNorm
from byol_a2.dataset import ADSRDataset, ADSR_h5_Dataset


torch.set_float32_matmul_precision('high')
warnings.filterwarnings(
    "ignore",
    message="`training_step` returned `None`. If this was on purpose, ignore this warning...",
    category=UserWarning,
)


def knn_predict(feature, feature_bank, knn_k=5, knn_t=1.0):
    """
    Perform KNN prediction using cosine similarity.
    
    Args:
        feature: Query features [batch_size, feature_dim]
        feature_bank: Reference feature bank [feature_dim, num_references]
        knn_k: Number of nearest neighbors
        knn_t: Temperature parameter for softmax
        
    Returns:
        top_k_indices: Indices of top-k neighbors [batch_size, k]
        top_k_similarities: Similarities of top-k neighbors [batch_size, k]
    """
    # Compute cosine similarity
    sim_matrix = torch.mm(feature, feature_bank)  # [batch_size, num_references]
    
    # Get top-k neighbors
    top_k_similarities, top_k_indices = sim_matrix.topk(k=knn_k, dim=-1)
    
    # Apply temperature scaling
    top_k_similarities = (top_k_similarities / knn_t).exp()
    
    return top_k_indices, top_k_similarities


def main():
    cfg = load_yaml_config("config_v2.yaml")
    device = torch.device(f'cuda:{cfg.device_id[0]}')
    
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.device_id[0])
    else:
        raise RuntimeError("CUDA device requested but not available")

    cfg.unit_samples = int(cfg.sample_rate * cfg.unit_sec)
    
    # Build Model
    model = AudioNTT2022(
        n_mels=cfg.n_mels,
        d=cfg.feature_d
    )
        
    # Initialize BYOL learner
    learner = BYOL(
        model,
        image_size=cfg.shape,
        hidden_layer=-1,
        projection_size=cfg.proj_size,
        projection_hidden_size=cfg.proj_dim,
        moving_average_decay=cfg.ema_decay,
    )
    
    # Load checkpoint
    if cfg.ckpt_path.endswith('.ckpt'):
        # PyTorch Lightning checkpoint
        checkpoint = torch.load(cfg.ckpt_path, map_location=f'cuda:{cfg.device_id[0]}')
        if 'state_dict' in checkpoint:
            # Load the full state dict into the learner
            learner.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            learner.load_state_dict(checkpoint, strict=False)
    else:
        # Direct model weights
        load_pretrained_weights(model, cfg.ckpt_path)
    
    encoder = learner.online_encoder
    encoder.eval()
    encoder.to(device)

    # Load memory dataset
    memory_dataset = ADSR_h5_Dataset(h5_path=cfg.h5_path, cache_size=getattr(cfg, 'cache_size', 1000))
    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=64, shuffle=False,  # Don't shuffle for consistent results
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    # Normalization
    norm_stats = np.array([cfg.mel_mean, cfg.mel_std])
    pre_norm = PrecomputedNorm(norm_stats).to(device)
    post_norm = NormalizeBatch()

    # Build feature bank from memory dataset
    feature_bank, feature_paths = [], []
    print("Building feature bank from memory dataset...")
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(memory_loader, desc='Building Feature Bank')):
            spec1, _, file_path1, _ = batch
            spec1 = spec1.to(device)

            # Preprocess mel spectrograms
            eps = torch.finfo(spec1.dtype).eps
            lms1 = (spec1 + eps).log()
            lms1 = pre_norm(lms1)
            lms1 = post_norm(lms1)

            # Extract features
            feature, _ = encoder(lms1)
            feature = F.normalize(feature, dim=1)  # L2 normalize: 64, 256

            feature_bank.append(feature)
            feature_paths.extend(file_path1)
            
            if idx > 2000:
                break

        # Concatenate all features
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()  # [feature_dim, num_samples]
        print(f"Feature bank shape: {feature_bank.shape}")
        print(f"Number of reference samples: {len(feature_paths)}")

    # Create output directory
    os.makedirs('info', exist_ok=True)
    
    # KNN Evaluation loop
    print("Starting KNN evaluation...")
    with open('info/test_knn.txt', 'w') as f:
        f.write("=== KNN Evaluation Results ===\n\n")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(memory_loader, desc='KNN Evaluation')):
                spec1, _, file_path1, _ = batch
                spec1 = spec1.to(device)

                # Preprocess mel spectrograms
                eps = torch.finfo(spec1.dtype).eps
                lms1 = (spec1 + eps).log()
                lms1 = pre_norm(lms1)
                lms1 = post_norm(lms1)

                # Extract features
                feature, _ = encoder(lms1)
                feature = F.normalize(feature, dim=1)  # L2 normalize

                # Perform KNN prediction
                top_k_indices, top_k_similarities = knn_predict(
                    feature, feature_bank, knn_k=5, knn_t=1.0
                )

                # Write results for each sample in the batch
                for i in range(len(file_path1)):
                    query_path = file_path1[i]
                    # query_feature = feature[i]
                    
                    f.write(f"=== Sample {batch_idx * memory_loader.batch_size + i + 1} ===\n")
                    f.write(f"Query file: {query_path}\n")
                    f.write("Top 5 nearest neighbors:\n")
                    
                    for rank in range(5):
                        neighbor_idx = top_k_indices[i, rank].item()
                        similarity = top_k_similarities[i, rank].item()
                        neighbor_path = feature_paths[neighbor_idx]
                        
                        f.write(f"  {rank + 1}. {neighbor_path} (similarity: {similarity:.4f})\n")
                    
                    f.write("\n")
                    
                    # # Write commands to convert npy to mp3 (if needed)
                    # f.write("Commands to convert to audio:\n")
                    # f.write(f"# Query: python npy2mp3.py --output_mp3 query_{batch_idx}_{i}.mp3 {query_path}\n")
                    
                    # for rank in range(5):
                    #     neighbor_idx = top_k_indices[i, rank].item()
                    #     neighbor_path = feature_paths[neighbor_idx]
                    #     f.write(f"# Top {rank + 1}: python npy2mp3.py --output_mp3 neighbor_{batch_idx}_{i}_{rank+1}.mp3 {neighbor_path}\n")
                    
                    # f.write("\n" + "="*50 + "\n\n")
                    
                    # Only process first 10 samples for demonstration
                    if batch_idx * memory_loader.batch_size + i >= 9:
                        break
                
                # Stop after first few batches
                if batch_idx >= 5:
                    break

    print("KNN evaluation completed!")
    print("Results saved to: info/test_knn.txt")
    
    # Print some statistics
    print(f"\nFeature bank statistics:")
    print(f"  - Total reference samples: {len(feature_paths)}")
    print(f"  - Feature dimension: {feature_bank.shape[0]}")
    print(f"  - Feature bank shape: {feature_bank.shape}")


if __name__ == "__main__":
    main()
