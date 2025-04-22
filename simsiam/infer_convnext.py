import argparse
import math
import os
import random
import warnings
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
import nnAudio.features

from utils import yaml_config_hook
from dataset import SimpleBPDataset, SegmentBPDataset
from model import SimSiam
from augmentation import PrecomputedNorm


torch.set_float32_matmul_precision('high')
warnings.filterwarnings(
    "ignore",
    message="`training_step` returned `None`. If this was on purpose, ignore this warning...",
    category=UserWarning,
)


def knn_predict(feature, feature_bank, knn_k=3, knn_t=1.0):
    sim_matrix = torch.mm(feature, feature_bank)
    top_k_similarities, top_k_indices = sim_matrix.topk(k=knn_k, dim=-1)
    top_k_similarities = (top_k_similarities / knn_t).exp()
    return top_k_indices, top_k_similarities

def main():
    parser = argparse.ArgumentParser(description="SimSiam Inference - Single GPU")
    config = yaml_config_hook("config/ssbp_convnext.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # build model
    print("=> Creating model with backbone encoder: '{}'".format(args.encoder_name))
    model = SimSiam(
        args=args,
        dim=args.dim,
        pred_dim=args.pred_dim,
    ).to(device)

    checkpoint = torch.load('/mnt/gestalt/home/buffett/simsiam_model_dict/convnext_model_dict_0422/checkpoint_0199.pth.tar')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)

    cudnn.benchmark = True
    model.eval()

    memory_dataset = SegmentBPDataset(
        data_dir=args.seg_dir,
        split="train",
        stem="other",
        eval_mode=True,
    )

    test_dataset = SegmentBPDataset(
        data_dir=args.seg_dir,
        split="test",
        stem="other",
        eval_mode=True,
    )

    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=64, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False,
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
    
    validate(memory_loader, test_loader, model, device, args, to_spec, pre_norm)

def validate(memory_loader, test_loader, model, device, args, to_spec, pre_norm):
    feature_bank, feature_paths = [], []

    with torch.no_grad():
        for x_i, path in tqdm(memory_loader, desc='Feature extracting'):
            x_i = x_i.to(device, non_blocking=True)
            
            # Mel-spec transform and normalize
            x_i = (to_spec(x_i) + torch.finfo().eps).log()
            x_i = pre_norm(x_i).unsqueeze(1)
            
            feature = model.encoder(x_i)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_paths.extend(path)

        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()  # [D, N]
        print(f"Feature bank shape: {feature_bank.shape}")

        test_bar = tqdm(test_loader, desc='KNN Evaluation')
        with open('info/test_matches_convnext.txt', 'w') as f:
            for x_i, test_path in test_bar:
                x_i = x_i.to(device, non_blocking=True)
                
                # Mel-spec transform and normalize
                x_i = (to_spec(x_i) + torch.finfo().eps).log()
                x_i = pre_norm(x_i).unsqueeze(1)
                
                feature = model.encoder(x_i)
                feature = F.normalize(feature, dim=1)

                top_k_indices, _ = knn_predict(
                    feature, feature_bank, knn_k=args.knn_k, knn_t=args.knn_t
                )

                for i in range(len(test_path)):
                    test_sample_path = test_path[i]
                    nearest_paths = [feature_paths[idx] for idx in top_k_indices[i].tolist()]
                    f.write(f"python npy2mp3.py --output_mp3 target.mp3    {os.path.join(args.seg_dir, test_sample_path, 'other_seg_0.npy')}\n")
                    for rank, neighbor_path in enumerate(nearest_paths, 1):
                        f.write(f"python npy2mp3.py --output_mp3 top{rank}.mp3    {os.path.join(args.seg_dir, neighbor_path, 'other_seg_0.npy')}\n")
                    f.write("\n")

                print(f"Test sample: {test_sample_path}")
                print(f"Top-3 Nearest Paths: {nearest_paths}")

if __name__ == "__main__":
    main()
