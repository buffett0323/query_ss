import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
    

class L1SNRLoss(_Loss):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = torch.tensor(eps)

    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]

        y_pred = y_pred.reshape(batch_size, -1)
        y_true = y_true.reshape(batch_size, -1)

        l1_error = torch.mean(torch.abs(y_pred - y_true), dim=-1)
        l1_true = torch.mean(torch.abs(y_true), dim=-1)

        snr = 20.0 * torch.log10((l1_true + self.eps) / (l1_error + self.eps))
        return -torch.mean(snr)





# Added from Banquet
class WeightedL1Loss(_Loss):
    def __init__(self, weights=None):
        super().__init__()

    def forward(self, y_pred, y_true):
        ndim = y_pred.ndim
        dims = list(range(1, ndim))
        loss = F.l1_loss(y_pred, y_true, reduction='none')
        loss = torch.mean(loss, dim=dims)
        weights = torch.mean(torch.abs(y_true), dim=dims)

        loss = torch.sum(loss * weights) / torch.sum(weights)

        return loss

class L1MatchLoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]

        y_pred = y_pred.reshape(batch_size, -1)
        y_true = y_true.reshape(batch_size, -1)

        l1_true = torch.mean(torch.abs(y_true), dim=-1)
        l1_pred = torch.mean(torch.abs(y_pred), dim=-1)
        loss = torch.mean(torch.abs(l1_pred - l1_true))

        return loss

class DecibelMatchLoss(_Loss):
    def __init__(self, eps=1e-3):
        super().__init__()

        self.eps = eps

    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]

        y_pred = y_pred.reshape(batch_size, -1)
        y_true = y_true.reshape(batch_size, -1)

        db_true = 10.0 * torch.log10(self.eps + torch.mean(torch.square(torch.abs(y_true)), dim=-1))
        db_pred = 10.0 * torch.log10(self.eps + torch.mean(torch.square(torch.abs(y_pred)), dim=-1))
        loss = torch.mean(torch.abs(db_pred - db_true))

        return loss
    
class L1SNRLossIgnoreSilence(_Loss):
    def __init__(self, eps=1e-3, dbthresh=-20, dbthresh_step=20):
        super().__init__()
        self.eps = torch.tensor(eps)
        self.dbthresh = dbthresh
        self.dbthresh_step = dbthresh_step

    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]

        y_pred = y_pred.reshape(batch_size, -1)
        y_true = y_true.reshape(batch_size, -1)

        l1_error = torch.mean(torch.abs(y_pred - y_true), dim=-1)
        l1_true = torch.mean(torch.abs(y_true), dim=-1)

        snr = 20.0 * torch.log10((l1_true + self.eps) / (l1_error + self.eps))
        
        db = 10.0 * torch.log10(torch.mean(torch.square(y_true), dim=-1) + 1e-6)
        
        if torch.sum(db > self.dbthresh) == 0:
            if torch.sum(db > self.dbthresh - self.dbthresh_step) == 0:
                return -torch.mean(snr)
            else:
                return -torch.mean(snr[db > self.dbthresh  - self.dbthresh_step])

        return -torch.mean(snr[db > self.dbthresh])

class L1SNRDecibelMatchLoss(_Loss):
    def __init__(self, db_weight=0.1, l1snr_eps=1e-3, dbeps=1e-3):
        super().__init__()
        self.l1snr = L1SNRLoss(l1snr_eps)
        self.decibel_match = DecibelMatchLoss(dbeps)
        self.db_weight = db_weight

    def forward(self, batch):
        loss_l1snr = self.l1snr(batch.estimates["target"].audio, batch.sources["target"].audio)
        loss_dem = self.decibel_match(batch.estimates["target"].audio, batch.sources["target"].audio)
        return loss_l1snr + loss_dem


class L1SNR_Recons_Loss(_Loss):
    """ Self-defined Loss Function """
    def __init__(
        self, 
        l1snr_eps=1e-3, 
        dbeps=1e-3,
        mask_type="MSE", 
    ):
        super().__init__()
        self.l1snr = L1SNRLoss(l1snr_eps)
        self.decibel_match = DecibelMatchLoss(dbeps)
        self.mask_type = mask_type
        
        if mask_type == "MSE":
            self.mask_loss = nn.MSELoss()
        elif mask_type == "BCE":
            self.mask_loss = nn.BCELoss()
        elif mask_type == "L1":
            self.mask_loss = nn.L1Loss()
 
        
    def forward(self, batch):
        
        # 1. Calculate Loss for Mask prediction
        if self.mask_type == "BCE":
            batch.masks.pred = torch.sigmoid(batch.masks.pred)    
        if self.mask_type != "None":
            loss_masks = self.mask_loss(batch.masks.pred, batch.masks.ground_truth)
        else: loss_masks = 0.0
        
        # 2. Calculate the L1SNR Loss of Separated query track
        loss_l1snr = self.l1snr(batch.estimates["target"].audio, batch.sources["target"].audio)
        
        # 3. Calculate the L1SNR Loss of Reconstruction Loss
        loss_dem = self.decibel_match(batch.estimates["target"].audio, batch.sources["target"].audio)
        
        return loss_masks + loss_l1snr + loss_dem





if __name__ == "__main__":
    # Example usage
    B, K, F, T, N = 4, 2, 512, 256, 65280  # Example dimensions
    predicted_masks = torch.randn(B, K, F, T)  # Predicted masks [B, K, F, T]
    gen_masks = torch.randn(B, K, F, T) 
    gt_stft = torch.randn(B, K, N)
    S_k = torch.randn(B, K, N)

    # Instantiate loss function
    loss_fn = L1SNRDecibelMatchLoss()

    # Calculate loss
    loss = loss_fn(predicted_masks, gen_masks, gt_stft, S_k)
    print(f"Loss: {loss.item()}")
