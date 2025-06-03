import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class ELBOLoss_VAE(nn.Module):
    def __init__(self):
        super(ELBOLoss_VAE, self).__init__()
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2))#, 3)) # shape: torch.Size([32, 128, 10])

    def kl_divergence(self, z, mu, std): # Monte carlo KL divergence
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl


    def forward(
        self,
        x_m, x_m_recon,
        x_s, x_s_recon,
        timbre_latent, tau_mu, tau_std,
        pitch_latent=None, pitch_priors=None,
    ):

        # 1. Reconstruction loss (MSE)
        # recon_loss_m = F.mse_loss(x_m_recon, x_m, reduction='sum')
        # recon_loss_x = F.mse_loss(x_s_recon, x_s, reduction='mean')
        recon_loss_x = self.gaussian_likelihood(x_s_recon, self.log_scale, x_s)

        # 2. Pitch supervision loss
        # pitch_loss = F.mse_loss(pitch_latent, pitch_priors, reduction='mean')

        # 3. KL Divergence for timbre latent (using standard Gaussian prior)
        # kl_loss = torch.mean(-0.5 * torch.sum(1 + tau_logvars - tau_means**2 - tau_logvars.exp(), dim=1))
        kl_loss = self.kl_divergence(timbre_latent, tau_mu, tau_std)

        # Total ELBO loss
        elbo = (kl_loss - recon_loss_x).mean()
        loss = elbo # + pitch_loss #recon_loss_x + kl_loss #recon_loss_m + recon_loss_x + kl_loss + pitch_loss

        return {
            'loss': loss,
            'recon_x': recon_loss_x.mean(),
            'kld': kl_loss.mean(), #-kl_loss.detach(),
            # 'pitch_loss': pitch_loss.detach(),
            # 'recon_m': recon_loss_m.detach(),
        }


class BarlowTwinsLoss_VAE(nn.Module):
    def __init__(
        self,
        batch_size=32,
        lambda_weight=0.005,
        embedding_dim=64,
        epsilon=1e-6,
    ):
        """
        Initialize Barlow Twins Loss.

        Args:
            lambda_weight (float): Weight on the off-diagonal terms.
            embedding_dim (int): Dimensionality of the embeddings.
        """
        super(BarlowTwinsLoss_VAE, self).__init__()
        self.lambda_weight = lambda_weight
        self.epsilon = epsilon
        self.identity_matrix = torch.eye(embedding_dim, requires_grad=False)  # D x D identity matrix
        self.batch_size = batch_size

    def forward(self, e_q, tau):
        # Normalize embeddings along the batch dimension
        e_q_norm = (e_q - e_q.mean(dim=0)) / (e_q.std(dim=0) + self.epsilon) # N x D
        tau_norm = (tau - tau.mean(dim=0)) / (tau.std(dim=0) + self.epsilon) # N x D

        # Cross-correlation matrix C
        c = torch.mm(e_q_norm.T, tau_norm) / self.batch_size # D x D

        # Loss computation
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum() #torch.diagonal(c_diff).sum()
        # off_diag = self._off_diagonal(c).pow_(2).sum() #self._off_diagonal(c_diff).sum() * self.lambda_weight
        loss = on_diag # + off_diag * self.lambda_weight


        return {
            'loss': loss,
            # 'on_diag': on_diag.detach(),
            # 'off_diag': off_diag.detach()* self.lambda_weight,
        }

    def _off_diagonal(self, matrix):
        n, m = matrix.shape
        assert n == m
        return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
            # matrix.flatten()[1:].view(n - 1, n + 1)[:, :-1].flatten()



class ELBOLoss(nn.Module):
    def __init__(self):
        super(ELBOLoss, self).__init__()
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2, 3)) # shape: torch.Size([32, 128, 10])

    def kl_divergence(self, z, mu, std): # Monte carlo KL divergence
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def forward(
        self,
        x_m, x_m_recon,
        x_s, x_s_recon,
        timbre_latent, tau_mu, tau_std,
    ):

        # 1. Reconstruction loss (MSE)
        # recon_loss_m = F.mse_loss(x_m_recon, x_m, reduction=self.reduction)
        # recon_loss_x = F.mse_loss(x_s_recon, x_s, reduction=self.reduction)
        recon_loss_x = self.gaussian_likelihood(x_s_recon, self.log_scale, x_s)


        # 2. KL Divergence for timbre latent (using standard Gaussian prior)
        kl_loss = self.kl_divergence(timbre_latent, tau_mu, tau_std)

        # Total ELBO loss
        elbo = (kl_loss - recon_loss_x).mean()
        loss = elbo #recon_loss_x + kl_loss #recon_loss_m + recon_loss_x + kl_loss + pitch_loss

        return {
            'loss': loss,
            'recon_x': recon_loss_x.mean(),
            'kld': kl_loss.mean(), #-kl_loss.detach(),
            # 'pitch_loss': pitch_loss.detach(),
            # 'recon_m': recon_loss_m.detach(),
        }



class BarlowTwinsLoss(nn.Module):
    def __init__(
        self,
        lambda_weight=0.005,
        embedding_dim=64,
        epsilon=1e-6,
    ):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.epsilon = epsilon
        self.identity_matrix = torch.eye(embedding_dim, requires_grad=False)  # D x D identity matrix


    def forward(self, e_q, tau):
        e_q, tau = e_q.squeeze(2), tau.squeeze(2)

        BS, channels, F = e_q.shape
        e_q = e_q.reshape(BS*channels, F)
        tau = tau.reshape(BS*channels, F)

        # Normalize embeddings along the batch dimension
        e_q_norm = (e_q - e_q.mean(dim=0)) / (e_q.std(dim=0) + self.epsilon) # N x D
        tau_norm = (tau - tau.mean(dim=0)) / (tau.std(dim=0) + self.epsilon) # N x D

        # Cross-correlation matrix C
        c = torch.mm(e_q_norm.T, tau_norm) / BS*channels # D x D

        # Loss computation
        loss = torch.diagonal(c).add_(-1).pow_(2).sum()

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
