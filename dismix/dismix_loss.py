import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class ELBOLoss(nn.Module):
    def __init__(
        self,
        reduction_method='mean',
    ):
        super(ELBOLoss, self).__init__()
        self.reduction_method = reduction_method

    def forward(self, x_m, x_m_recon, tau_means, tau_logvars, nu_logits, y_pitch):
        """
        Computes the ELBO loss.

        Args:
            x_m (torch.Tensor): Original mixture data of shape (batch_size, data_dim).
            x_m_recon (torch.Tensor): Reconstructed mixture data of shape (batch_size, data_dim).
            tau_means (list of torch.Tensor): List of timbre latent means for each source.
            tau_logvars (list of torch.Tensor): List of timbre latent log variances for each source.
            nu_logits (list of torch.Tensor): List of pitch latent logits for each source.
            y_pitch (list of torch.Tensor): List of ground truth pitch labels for each source.

        Returns:
            torch.Tensor: Scalar loss value.
        """


        # 1. Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_m_recon, x_m, reduction=self.reduction_method)

        # 2. Pitch supervision loss
        pitch_loss = F.mse_loss(nu_logits, y_pitch, reduction=self.reduction_method)

        # 3. KL divergence loss for timbre latents
        kl_loss = 0.0
        for mu, logvar in zip(tau_means, tau_logvars):
            # KL divergence between N(mu, sigma^2) and N(0, 1)
            kl_loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Average loss over the batch
        kl_loss = kl_loss / x_m.size(0)

        # Total ELBO loss (negative ELBO)
        loss = recon_loss + pitch_loss + kl_loss

        return loss
    
    

class BarlowTwinsLoss(nn.Module):
    def __init__(
        self, 
        epsilon=1e-9,
        consider_off_diagonal=False,
    ):
        super(BarlowTwinsLoss, self).__init__()
        self.epsilon = epsilon  # Small constant to prevent NaNs
        self.consider_off_diagonal = consider_off_diagonal

    def forward(self, e_q, tau):
        """
        Compute the Barlow Twins loss between query embeddings and timbre latents.

        Parameters:
        - e_q (Tensor): Query embeddings for timbre. Shape: (batch_size, D_tau)
        - tau (Tensor): Sampled timbre latents τ. Shape: (batch_size, D_tau)

        Returns:
        - loss (Tensor): Barlow Twins loss value.
        """
        N_s, _ = e_q.size()

        # Calculate mean and std, add epsilon to std to avoid division by zero
        e_q_mean = e_q.mean(dim=0, keepdim=True)
        e_q_std = e_q.std(dim=0, keepdim=True).clamp(min=self.epsilon)

        tau_mean = tau.mean(dim=0, keepdim=True)
        tau_std = tau.std(dim=0, keepdim=True).clamp(min=self.epsilon)
        
        # Normalize embeddings, replace NaNs in normalized tensors with zeros
        e_q_norm = (e_q - e_q_mean) / e_q_std
        tau_norm = (tau - tau_mean) / tau_std
        e_q_norm = torch.nan_to_num(e_q_norm, nan=0.0)
        tau_norm = torch.nan_to_num(tau_norm, nan=0.0)

        # Compute the cross-correlation matrix C
        C = torch.einsum('bi, bj -> ij', e_q_norm, tau_norm) / N_s #(N_s + self.epsilon)
        
        # Clamp C to avoid extreme values leading to NaNs
        C = C.clamp(-1 + self.epsilon, 1 - self.epsilon)

        # Compute the loss: sum over diagonal elements
        c_diff = (1 - torch.diag(C)) ** 2
        loss = c_diff.sum()
        
        # # Cross-correlation matrix C
        # e_q = (e_q - e_q.mean(0)) / (e_q.std(0) - self.epsilon)
        # tau = (tau - tau.mean(0)) / (tau.std(0) - self.epsilon)

        # cross_corr = torch.mm(e_q.T, tau) / N_s

        # # Compute Barlow Twins loss
        # loss = 0
        # if self.consider_off_diagonal:
        #     for d in range(self.timbre_dim):
        #         loss += (1 - cross_corr[d, d]) ** 2  # Diagonal elements should be close to 1
        #         loss += (cross_corr[:, d].sum() - cross_corr[d, d]) ** 2  # Off-diagonal elements should be close to 0
        
        # else:
        #     # Only consider diagonal elements
        #     loss = ((1 - torch.diag(cross_corr)) ** 2).sum()

        return loss

    



# Example usage
if __name__ == '__main__':
    batch_size = 8
    latent_dim = 64

    # Example pitch and timbre latent representations
    pitch_latent = torch.randn(batch_size, latent_dim)  # ν(i)
    timbre_latent = torch.randn(batch_size, latent_dim)  # τ(i)

    # Initialize Barlow Twins loss
    loss_fn = BarlowTwinsLoss(lambda_param=0.005)

    # Compute the Barlow Twins loss
    loss = loss_fn(pitch_latent, timbre_latent)
    print(f"Barlow Twins Loss: {loss.item()}")

