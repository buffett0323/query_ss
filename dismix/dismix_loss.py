import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class ELBOLoss(nn.Module):
    def __init__(
        self,
        reduction='mean',
    ):
        super(ELBOLoss, self).__init__()
        self.reduction = reduction


    def forward(self, x_m, x_m_recon, x_s, x_s_recon, tau_means, tau_logvars, pitch_latent=None, pitch_priors=None):
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
        recon_loss_m = F.mse_loss(x_m_recon, x_m, reduction=self.reduction)
        recon_loss_x = F.mse_loss(x_s_recon, x_s, reduction=self.reduction)
        # recon_loss = self.gaussian_likelihood(x_m_recon, self.log_scale, x_m)

        # Added source reconstruction loss
        # source_recon_loss = F.mse_loss(x_s, x_s_recon, reduction=self.reduction)

        # 2. Pitch supervision loss
        if pitch_latent != None and pitch_priors != None:
            pitch_loss = F.mse_loss(pitch_latent, pitch_priors, reduction=self.reduction)

        # 3. KL Divergence for timbre latent (using standard Gaussian prior)
        # kl_loss = -0.5 * torch.sum(1 + tau_logvars - tau_means.pow(2) - tau_logvars.exp(), dim=1)
        # kl_loss = torch.mean(-0.5 * torch.sum(1 + tau_logvars - tau_means ** 2 - tau_logvars.exp(), dim = 1), dim = 0)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + tau_logvars - tau_logvars.exp() - tau_means ** 2, dim=2))

        # kl_loss = self.kl_divergence(timbre_latent, tau_means, tau_logvars)
        
        # Total ELBO loss
        loss = recon_loss_m + recon_loss_x + kl_loss #+ source_recon_loss # + pitch_loss
        
        if pitch_latent != None and pitch_priors != None:
            loss += pitch_loss
        # print(recon_loss_m.item(), recon_loss_x.item(), kl_loss.item()) #, source_recon_loss.item())
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
        batch_size, channels, _, D_tau = e_q.shape
        e_q = e_q.view(batch_size * channels, D_tau)  # Combine batch and channel dimensions
        tau = tau.view(batch_size * channels, D_tau)  # Combine batch and channel dimensions

        # Normalize embeddings along the feature dimension (D_tau)
        e_q = F.normalize(e_q, dim=1)  # Shape: [batch_size * channels, D_tau]
        tau = F.normalize(tau, dim=1)  # Shape: [batch_size * channels, D_tau]

        # Compute the cross-correlation matrix C
        C = torch.mm(e_q.T, tau) / (batch_size * channels)  # Shape: [D_tau, D_tau]

        # Create identity matrix for the same dimensionality as C
        identity = torch.eye(C.shape[0], device=C.device)  # Shape: [D_tau, D_tau]

        # Compute the Barlow Twins loss
        # Penalize off-diagonal terms and deviations from 1 on the diagonal
        loss = torch.sum((C - identity) ** 2)

        return loss
        # N_s, _ = e_q.size()

        # # Calculate mean and std, add epsilon to std to avoid division by zero
        # e_q_mean = e_q.mean(dim=0, keepdim=True)
        # e_q_std = e_q.std(dim=0, keepdim=True).clamp(min=self.epsilon)

        # tau_mean = tau.mean(dim=0, keepdim=True)
        # tau_std = tau.std(dim=0, keepdim=True).clamp(min=self.epsilon)
        
        # # Normalize embeddings, replace NaNs in normalized tensors with zeros
        # e_q_norm = (e_q - e_q_mean) / e_q_std
        # tau_norm = (tau - tau_mean) / tau_std
        # e_q_norm = torch.nan_to_num(e_q_norm, nan=0.0)
        # tau_norm = torch.nan_to_num(tau_norm, nan=0.0)

        # # Compute the cross-correlation matrix C
        # C = torch.einsum('bi, bj -> ij', e_q_norm, tau_norm) / N_s #(N_s + self.epsilon)
        
        # # Clamp C to avoid extreme values leading to NaNs
        # C = C.clamp(-1 + self.epsilon, 1 - self.epsilon)

        # # Compute the loss: sum over diagonal elements
        # c_diff = (1 - torch.diag(C)) ** 2
        # loss = c_diff.sum()

        # return loss

    



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

