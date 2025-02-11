import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim.lr_scheduler import LambdaLR
from torch_models import Wavegram_Logmel128_Cnn14
from utils import *


# knn monitor as in InstDisc http://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank -> [B, N]
    sim_matrix = torch.mm(feature, feature_bank) # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1) # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # Counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device) # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0) # weighted score -> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    
    return pred_labels


class SimSiamPL(pl.LightningModule):
    def __init__(
        self, 
        args,
        dim=2048, 
        pred_dim=512,
    ):
        """
        PyTorch Lightning version of SimSiam.
        """
        super().__init__()
        self.save_hyperparameters()  # Saves args automatically for checkpointing
        self.args = args
        
        # Create the encoder
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

        self.criterion = nn.CosineSimilarity(dim=1)
        
        for param in self.parameters():
            param.data = param.data.contiguous()


    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


    def training_step(self, batch, batch_idx):
        x_i, x_j, _ = batch
        x_i, x_j = x_i.contiguous(), x_j.contiguous()

        p1, p2, z1, z2 = self(x_i, x_j)
        loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
        
        # Compute per-channel std of L2-normalized output
        z1_normalized = F.normalize(z1, dim=1)
        z2_normalized = F.normalize(z2, dim=1)
        z1_std = z1_normalized.std(dim=0).mean()
        z2_std = z2_normalized.std(dim=0).mean()
        avg_std = (z1_std + z2_std) / 2
        
        # Log learning rate (fetch from optimizer)
        lr = self.optimizers().param_groups[0]["lr"]  # Get current learning rate
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        self.log("learning_rate", lr, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        self.log("avg_std_train", avg_std, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)

        return loss
    
    
    # def validation_step(self, batch, batch_idx):
    #     x_i, x_j, _ = batch
    #     x_i, x_j = x_i.contiguous(), x_j.contiguous()

    #     p1, p2, z1, z2 = self(x_i, x_j)
    #     loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5

    #     self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
    #     return loss


    # def test_step(self, batch, batch_idx):
    #     x_i, x_j, _ = batch
    #     x_i, x_j = x_i.contiguous(), x_j.contiguous()

    #     p1, p2, z1, z2 = self(x_i, x_j)
    #     loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5

    #     self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
    #     return loss


    # def on_validation_epoch_end(self):
    #     # kNN validation monitoring inside PyTorch Lightning
    #     model = self.encoder
    #     model.eval()
        
    #     # Get train (memory) and test (validation) data from datamodule
    #     if self.trainer.world_size > 1:
    #         memory_sampler = torch.utils.data.distributed.DistributedSampler(self.trainer.datamodule.memory_ds)
    #     else:
    #         memory_sampler = None
            
    #     memory_loader = torch.utils.data.DataLoader(
    #         self.trainer.datamodule.memory_ds,
    #         batch_size=self.args.batch_size,
    #         shuffle=False,
    #         sampler=memory_sampler,
    #         num_workers=self.args.workers,
    #         drop_last=self.args.drop_last,
    #         pin_memory=self.args.pin_memory,
    #     )
    #     test_loader = self.trainer.datamodule.test_dataloader()
        
        
    #     # Step 1: Extract feature bank
    #     feature_bank, feature_labels = [], []
    #     with torch.no_grad():
    #         for data, _, target in tqdm(memory_loader, desc='Extracting Features'):
    #             data = data.to(self.device, non_blocking=True)
    #             target = target.to(self.device, non_blocking=True)

    #             feature = model(data)
    #             feature = F.normalize(feature, dim=1)

    #             feature_bank.append(feature)
    #             feature_labels.append(target)

    #         feature_bank = torch.cat(feature_bank, dim=0)
    #         feature_labels = torch.cat(feature_labels, dim=0)
            
            
    #         # Handle Distributed Training (DDP)
    #         if self.trainer.world_size > 1:
    #             feature_bank_list = [torch.zeros_like(feature_bank) for _ in range(self.trainer.world_size)]
    #             torch.distributed.all_gather(feature_bank_list, feature_bank)
    #             feature_bank = torch.cat(feature_bank_list, dim=0)

    #             feature_labels_list = [torch.zeros_like(feature_labels) for _ in range(self.trainer.world_size)]
    #             torch.distributed.all_gather(feature_labels_list, feature_labels)
    #             feature_labels = torch.cat(feature_labels_list, dim=0)

    #         feature_bank = feature_bank.T.contiguous()  # [D, N]
        
        
    #     # Step 2: kNN Classification on Validation Data
    #     total_top1, total_num = 0.0, 0
    #     for data, target in tqdm(test_loader, desc="Testing kNN"):
    #         data = data.to(self.device, non_blocking=True)
    #         target = target.to(self.device, non_blocking=True)

    #         feature = model(data)
    #         feature = F.normalize(feature, dim=1)

    #         # kNN prediction
    #         pred_labels = self.knn_predict(
    #             feature, feature_bank, feature_labels,
    #             classes=self.args.knn_classes,
    #             knn_k=self.args.knn_k, 
    #             knn_t=self.args.knn_t,
    #         )

    #         total_num += data.size(0)
    #         total_top1 += (pred_labels[:, 0] == target).float().sum().item()

    #     knn_acc = total_top1 / total_num * 100
    #     self.log("knn_acc", knn_acc, on_epoch=True, prog_bar=True, logger=True,  batch_size=self.args.batch_size, sync_dist=True)


    # def knn_predict(self, feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    #     """
    #     kNN classification to evaluate representation quality.
    #     """
    #     # Compute cosine similarity between features and feature bank
    #     sim_matrix = torch.mm(feature, feature_bank)
    #     sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)

    #     # Get corresponding labels
    #     sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    #     sim_weight = (sim_weight / knn_t).exp()

    #     # Compute weighted class counts
    #     one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    #     one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    #     pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    #     pred_labels = pred_scores.argsort(dim=-1, descending=True)
    #     return pred_labels
    
    
    def inference(self, x):
        return self.encoder(x)


    def configure_optimizers(self):
        """Configure optimizer and scheduler with cosine LR decay & fixed predictor LR"""
        
        # Define optimizer parameters: Fix predictor LR if needed
        if self.args.fix_pred_lr:
            optim_params = [
                {'params': self.encoder.parameters(), 'fix_lr': False},
                {'params': self.predictor.parameters(), 'fix_lr': True},  # Fixed LR for predictor
            ]
        else:
            optim_params = self.parameters()
        
        # Define SGD optimizer
        optimizer = optim.SGD(
            optim_params, 
            lr=self.args.lr, 
            momentum=self.args.momentum, 
            weight_decay=self.args.weight_decay,
        )

        # Define cosine annealing learning rate scheduler
        def cosine_annealing(epoch):
            """Lambda function for cosine decay learning rate"""
            if epoch < self.args.warmup_epochs:
                return (epoch + 1) / self.args.warmup_epochs  # Linear warm-up
            else:
                return 0.5 * (1.0 + math.cos(math.pi * (epoch - self.args.warmup_epochs) /
                                             (self.args.epochs - self.args.warmup_epochs)))  # Cosine Decay

        scheduler = LambdaLR(optimizer, lr_lambda=cosine_annealing)

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss"}]




# TODO: augmentation
class SimSiam(nn.Module):
    """
    SimSiam model with Wavegram_Logmel128_Cnn14 as the encoder.
    """
    def __init__(self, args, dim=2048, pred_dim=512):
        super(SimSiam, self).__init__()
        self.args = args

        # **Initialize Wavegram_Logmel128_Cnn14 as the encoder**
        self.encoder = Wavegram_Logmel128_Cnn14(
            sample_rate=self.args.sample_rate,
            window_size=self.args.window_size,
            hop_size=self.args.hop_length,
            mel_bins=128,
            fmin=self.args.fmin,
            fmax=self.args.fmax,
            classes_num=dim  # Output embedding dimension
        )

        # **Remove the classification head to use raw feature embeddings**
        prev_dim = self.encoder.fc1.weight.shape[1] # Extracting feature dimension
        self.encoder.fc1 = nn.Identity()

        # **Build a separate 3-layer projector**
        self.projector = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # First layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # Second layer
            nn.Linear(prev_dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False)  # Output layer (no affine)
        )

        # **Build a 2-layer predictor**
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # Hidden layer
            nn.Linear(pred_dim, dim)  # Output layer
        )

    def forward(self, x1, x2):
        """
        Forward pass for SimSiam model.
        """
        # Compute features for both views
        z1 = self.projector(self.encoder(x1))  # NxC
        z2 = self.projector(self.encoder(x2))  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="SimCLR Encoder")

    config = yaml_config_hook("ssbp_pl_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    
    # Load models
    model = SimSiam(args)#.to(device)

    
    x = torch.randn([16, 32000])#.to(device)
    res = model(x, x)
    print(res[0].shape, res[2].shape)