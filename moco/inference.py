#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random
import nnAudio
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from collections import OrderedDict
from tqdm import tqdm
from model import MoCo
from utils import yaml_config_hook, AverageMeter, ProgressMeter
from dataset import SegmentBPDataset
from augmentation import PrecomputedNorm, NormalizeBatch

EPOCH='0179'
GPU_ID=1


def knn_predict(feature, feature_bank, knn_k=3, knn_t=1.0):
    sim_matrix = torch.mm(feature, feature_bank)
    top_k_similarities, top_k_indices = sim_matrix.topk(k=knn_k, dim=-1)
    top_k_similarities = (top_k_similarities / knn_t).exp()
    return top_k_indices, top_k_similarities


def main() -> None:
    parser = argparse.ArgumentParser(description="MoCoV2_BP")
    config = yaml_config_hook("config/moco_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    
    args = parser.parse_args()
    
    # Initial settings
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    
    # build model
    print("=> Creating model '{}'".format(args.arch))
    model = MoCo(
        args, 
        dim=args.moco_dim, 
        K=args.moco_K, 
        m=args.moco_m, 
        T=args.moco_T, 
        mlp=args.moco_mlp
    ).to(device)
    
    checkpoint = torch.load(f'/mnt/gestalt/home/buffett/moco_model_dict/bass_other_new_amp05/checkpoint_{EPOCH}.pth.tar')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)

    cudnn.benchmark = True
    inference_encoder = model.encoder_q
    inference_encoder.eval()
    

    # Data loading code
    memory_dataset = SegmentBPDataset(
        data_dir=args.seg_dir,
        split="train",
        stem="bass_other",
        eval_mode=True,
        num_seq_segments=args.num_seq_segments,
        fixed_second=args.fixed_second,
        sp_method=args.sp_method,
        p_ts=args.p_ts,
        p_ps=args.p_ps,
        p_tm=args.p_tm,
        p_tstr=args.p_tstr,
        semitone_range=args.semitone_range,
        tm_min_band_part=args.tm_min_band_part,
        tm_max_band_part=args.tm_max_band_part,
        tm_fade=args.tm_fade,
        amp_name=args.amp_name,
        use_lmdb=args.use_lmdb,
        lmdb_path=args.lmdb_dir,
    )
    
    test_dataset = SegmentBPDataset(
        data_dir=args.seg_dir,
        split="test",
        stem="bass_other",
        eval_mode=True,
        num_seq_segments=args.num_seq_segments,
        fixed_second=args.fixed_second,
        sp_method=args.sp_method,
        p_ts=args.p_ts,
        p_ps=args.p_ps,
        p_tm=args.p_tm,
        p_tstr=args.p_tstr,
        semitone_range=args.semitone_range,
        tm_min_band_part=args.tm_min_band_part,
        tm_max_band_part=args.tm_max_band_part,
        tm_fade=args.tm_fade,
        amp_name=args.amp_name,
        use_lmdb=args.use_lmdb,
        lmdb_path=args.lmdb_dir,
    )
    

    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=64, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    
    # MelSpectrogram
    to_spec = nnAudio.features.MelSpectrogram(
        sr=args.sample_rate,
        n_fft=args.n_fft,
        win_length=args.window_size,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        center=True,
        power=2,
        verbose=False,
    ).to(device)
    
    # Normalization: PrecomputedNorm
    pre_norm = PrecomputedNorm(np.array(args.norm_stats)).to(device)
    post_norm = NormalizeBatch().to(device)


    # Validation
    feature_bank, feature_paths = [], []

    with torch.no_grad():
        # Training memory loader
        for x_i, _, path in tqdm(memory_loader, desc='Training Dataset Feature extracting'):
            x_i = x_i.to(device, non_blocking=True)
            
            # Mel-spec transform and normalize
            x_i = (to_spec(x_i) + torch.finfo().eps).log()
            x_i = pre_norm(x_i).unsqueeze(1)
            
            # Form a batch and post-normalize it.
            bs = x_i.shape[0]
            paired_inputs = post_norm(x_i)
            
            # Forward pass
            feature = inference_encoder(paired_inputs)
            feature = F.normalize(feature, dim=1)
            
            feature_bank.append(feature)
            feature_paths.extend(path)

        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()  # [D, N]
        print(f"Training Feature bank shape: {feature_bank.shape}")
        
        
        # Testing memory loader
        feature_bank_test, label_bank_test, feature_paths_test = [], [], []
        for x_i, label, path in tqdm(test_loader, desc='Testing Dataset Feature extracting'):
            x_i = x_i.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)      
            
            # Mel-spec transform and normalize
            x_i = (to_spec(x_i) + torch.finfo().eps).log()
            x_i = pre_norm(x_i).unsqueeze(1)
            
            # Form a batch and post-normalize it.
            bs = x_i.shape[0]
            paired_inputs = post_norm(x_i)
            
            # Forward pass
            feature = inference_encoder(paired_inputs)
            feature = F.normalize(feature, dim=1)
            
            feature_bank_test.append(feature)
            label_bank_test.append(label)
            feature_paths_test.extend(path)
            
        feature_bank_test = torch.cat(feature_bank_test, dim=0).t().contiguous()  # [D, N]
        label_bank_test = torch.cat(label_bank_test, dim=0).t().contiguous()  # [D, N]
        print(f"Testing Feature bank shape: {feature_bank_test.shape}")
        print(f"Testing Label bank shape: {label_bank_test.shape}")


        
        # KNN Evaluation
        with open(f'info/test_matches_convnext_epoch{EPOCH}.txt', 'w') as f:
            for x_i, label, test_path in tqdm(test_loader, desc='KNN Evaluation'):
                x_i = x_i.to(device, non_blocking=True)
                
                # Mel-spec transform and normalize
                x_i = (to_spec(x_i) + torch.finfo().eps).log()
                x_i = pre_norm(x_i).unsqueeze(1)
                
                # Form a batch and post-normalize it.
                bs = x_i.shape[0]
                paired_inputs = post_norm(x_i)
                
                # Forward pass
                feature = inference_encoder(paired_inputs)
                feature = F.normalize(feature, dim=1)

                """ 
                1. Storing Training Memory Dataset Knn-similar sounds 
                """
                top_k_indices, _ = knn_predict(
                    feature, feature_bank, knn_k=args.knn_k, knn_t=args.knn_t
                )
                
                """
                2. Storing Testing Memory Dataset Knn-similar sounds: 
                see whether the nearest top 3 are same label, and same sound
                """
                top_k_indices_test, _ = knn_predict(
                    feature, feature_bank_test, knn_k=args.knn_k, knn_t=args.knn_t
                )
                
                # Write to file
                for i in range(len(test_path)):
                    test_sample_path = test_path[i]
                    
                    f.write("-----KNN-Prediction-----\n")
                    
                    # For Training
                    nearest_paths = [feature_paths[idx] for idx in top_k_indices[i].tolist()]
                    f.write(f"python npy2mp3.py --output_mp3 target.mp3    {os.path.join(args.seg_dir, test_sample_path, 'other_seg_0.npy')}\n")
                    
                    for rank, neighbor_path in enumerate(nearest_paths, 1):
                        f.write(f"python npy2mp3.py --output_mp3 top{rank}.mp3    {os.path.join(args.seg_dir, neighbor_path, 'other_seg_0.npy')}\n")
                    f.write("\n")
                    
                    
                    # For Testing 
                    nearest_paths_test = [feature_paths_test[idx] for idx in top_k_indices_test[i].tolist()]
                    nearest_labels_test = [label_bank_test[idx] for idx in top_k_indices_test[i].tolist()]

                    for rank, neighbor_path in enumerate(nearest_paths_test, 1):
                        f.write(f"python npy2mp3.py --output_mp3 test_top{rank}.mp3    {os.path.join(args.seg_dir, neighbor_path, 'other_seg_0.npy')}\n")
                    f.write("\n")
                    
                    
                    # Labels
                    f.write("-----Labels-----\n")
                    f.write(f"Testing sample label: {label[i]}\n")
                    for rank, neighbor_label in enumerate(nearest_labels_test, 1):
                        f.write(f"test_target{rank}: {neighbor_label}\n")
                    f.write("\n")
                    

if __name__ == "__main__":
    main()
