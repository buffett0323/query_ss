import argparse
import torch
import torch.nn as nn
from torch_models import Wavegram_Logmel_Cnn14, Wavegram_Logmel128_Cnn14
from utils import *


# TODO: augmentation
class SimSiam(nn.Module):
    def __init__(
        self, 
        args,
        dim=2048, 
        pred_dim=512,
    ):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()
        self.args = args
        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        # self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        self.encoder = Wavegram_Logmel128_Cnn14(
            sample_rate=self.args.sample_rate, 
            window_size=self.args.window_size, 
            hop_size=self.args.hop_length, 
            mel_bins=128, #self.args.n_mels, 
            fmin=self.args.fmin,
            fmax=self.args.fmax,
            classes_num=dim, #n_features,
        )
        

        # build a 3-layer projector
        prev_dim = self.encoder.fc1.weight.shape[1]
        self.encoder.fc1 = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True), # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True), # second layer
            self.encoder.fc1, # self.fc1 = nn.Linear(2048, classes_num, bias=True)
            nn.BatchNorm1d(dim, affine=False),
        ) # output layer
        self.encoder.fc1[6].bias.requires_grad = False # hack: not use bias as it is followed by BN


        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True), # hidden layer
            nn.Linear(pred_dim, dim),    
        ) # output layer



    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="SimCLR Encoder")

    config = yaml_config_hook("bp_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Load models
    model = SimSiam(args).to(device)
    x = torch.randn([16, 32000]).to(device)
    res = model(x, x)
    print(res[0].shape, res[2].shape)