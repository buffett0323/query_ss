import argparse
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
# from simclr.modules.resnet_hacks import modify_resnet_model
# from simclr.modules.identity import Identity

# SimCLR
from loss import NT_Xent

# from simclr.modules import NT_Xent, get_resnet
# from simclr.modules.transformations import TransformsSimCLR
# from simclr.modules.sync_batchnorm import convert_model

class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.global_pool = global_mean_pool  # Pooling layer to get a single graph embedding

    def forward(self, x, edge_index, batch):
        """
        Args:
            x (Tensor): Node features [num_nodes, input_dim].
            edge_index (Tensor): Edge indices [2, num_edges].
            batch (Tensor): Batch indices for global pooling [num_nodes].
        Returns:
            Tensor: Graph embeddings [batch_size, output_dim].
        """
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = self.global_pool(x, batch)  # Aggregate node embeddings into a graph embedding
        return x



class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        # self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        
        return h_i, h_j, z_i, z_j
    


class ContrastiveLearning(LightningModule):
    def __init__(self, args):
        super().__init__()

        self.hparams = args

        # initialize ResNet
        self.encoder = GCNEncoder(
            input_dim=self.hparams.input_dim,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=self.hparams.output_dim
        )  #get_resnet(self.hparams.resnet, pretrained=False)
        self.n_features = self.hparams.output_dim  # get dimensions of fc layer
        self.model = SimCLR(
            self.encoder, 
            self.hparams.projection_dim, 
            self.n_features
        )
        self.criterion = NT_Xent(
            self.hparams.batch_size, self.hparams.temperature, world_size=1
        )

    def forward(self, x_i, x_j):
        h_i, h_j, z_i, z_j = self.model(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        (x_i, x_j), _ = batch
        loss = self.forward(x_i, x_j)
        return loss

    def configure_criterion(self):
        criterion = NT_Xent(self.hparams.batch_size, self.hparams.temperature)
        return criterion

    def configure_optimizers(self):
        scheduler = None
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        elif self.hparams.optimizer == "LARS":
            # optimized using LARS with linear learning rate scaling
            # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
            learning_rate = 0.3 * args.batch_size / 256
            optimizer = LARS(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=args.weight_decay,
                exclude_from_weight_decay=["batch_normalization", "bias"],
            )

            # "decay the learning rate with the cosine decay schedule without restarts"
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.epochs, eta_min=0, last_epoch=-1
            )
        else:
            raise NotImplementedError

        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}