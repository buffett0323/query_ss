import torch
import torch.nn as nn
import torch.nn.functional as F



class ELBOLoss(nn.Module):
    def __init__(self):
        super(ELBOLoss, self).__init__()

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

        # Number of sources in the mixture
        N_s = len(tau_means)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_m_recon, x_m, reduction='mean')

        # Pitch supervision loss
        pitch_loss = 0.0
        for nu_logit, y in zip(nu_logits, y_pitch):
            # Assume nu_logit is raw logits; apply CrossEntropyLoss
            pitch_loss += F.cross_entropy(nu_logit, y, reduction='mean')

        # KL divergence loss for timbre latents
        kl_loss = 0.0
        for mu, logvar in zip(tau_means, tau_logvars):
            # KL divergence between N(mu, sigma^2) and N(0, 1)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss += kl

        # Average KL loss over the batch
        kl_loss = kl_loss / x_m.size(0)

        # Total ELBO loss (negative ELBO)
        loss = recon_loss + pitch_loss + kl_loss

        return loss
    
    

class BarlowTwinsLoss(nn.Module):
    def __init__(self, epsilon=1e-9):
        super(BarlowTwinsLoss, self).__init__()
        self.epsilon = epsilon  # Small constant to prevent NaNs

    def forward(self, e_q, tau):
        """
        Compute the Barlow Twins loss between query embeddings and timbre latents.

        Parameters:
        - e_q (Tensor): Query embeddings for timbre. Shape: (batch_size, D_tau)
        - tau (Tensor): Sampled timbre latents τ. Shape: (batch_size, D_tau)

        Returns:
        - loss (Tensor): Barlow Twins loss value.
        """
        N_s, D_tau = e_q.size()

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
        C = torch.einsum('bi, bj -> ij', e_q_norm, tau_norm) / (N_s + self.epsilon)
        
        # Clamp C to avoid extreme values leading to NaNs
        C = C.clamp(-1 + self.epsilon, 1 - self.epsilon)

        # Compute the loss: sum over diagonal elements
        c_diff = (1 - torch.diag(C)) ** 2
        loss = c_diff.sum()
        
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

