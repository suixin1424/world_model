from mmdet.models import LOSSES
from torch import nn
import torch

@LOSSES.register_module()
class vae_loss(nn.Module):
    def __init__(self):
        ''' Initialize a VAE loss function.
        '''
        super().__init__()
        self.mse_loss = nn.MSELoss(size_average=False)

    def forward(self, x, x_recon, mu, log_var):
        ''' Compute the VAE loss.

        Parameters
        ----------
        x : torch.Tensor, shape=(N, K, R, C)
            The input images.
        x_recon : torch.Tensor, shape=(N, K, R, C)
            The reconstructed images.
        mu : torch.Tensor, shape=(N, 32)
            The mean vectors.
        log_var : torch.Tensor, shape=(N, 32)
            The log variance vectors.

        Returns
        -------
        torch.Tensor
            The VAE loss.
        '''
        mse_loss = self.mse_loss(x_recon, x)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return mse_loss + kl_div