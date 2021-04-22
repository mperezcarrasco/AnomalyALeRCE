import torch
import torch.nn as nn
import torch.nn.functional as F


class vae(nn.Module):
    """Pytorch implementation of a variational autoencoder based on MLP."""

    def __init__(self, args):
        super(vae, self).__init__()

        #Encoder Architecture
        self.enc1 = nn.Linear(args.in_dim, 512)
        self.encbn1 = nn.BatchNorm1d(512)
        self.enc2 = nn.Linear(512, 256)
        self.encbn2 = nn.BatchNorm1d(256)
        self.enc3 = nn.Linear(256, 128)
        self.encbn3 = nn.BatchNorm1d(128)


        self.mu = nn.Linear(128, args.z_dim)
        self.log_var = nn.Linear(128, args.z_dim)

        #Decoder Architecture
        self.dec1 = nn.Linear(args.z_dim, 128)
        self.decbn1 = nn.BatchNorm1d(128)
        self.dec2 = nn.Linear(128, 256)
        self.decbn2 = nn.BatchNorm1d(256)
        self.dec3 = nn.Linear(256,512)
        self.decbn3 = nn.BatchNorm1d(512)
        self.dec4 = nn.Linear(512, args.in_dim)

    def encode(self, x):
        h = F.leaky_relu(self.encbn1(self.enc1(x)))
        h = F.leaky_relu(self.encbn2(self.enc2(h)))
        h = F.leaky_relu(self.encbn3(self.enc3(h)))
        return self.mu(h), self.log_var(h)

    def decode(self, x):
        h = F.leaky_relu(self.decbn1(self.dec1(x)))
        h = F.leaky_relu(self.decbn2(self.dec2(h)))
        h = F.leaky_relu(self.decbn3(self.dec3(h)))
        return torch.sigmoid(self.dec4(h))
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return mu, log_var, z, x_hat

    def compute_loss(self, x):
        """
        Compute Evidence Lower Bound (ELBO) for the variational autoencoder.
        """
        #TODO: for a couple of forward passes.
        mu, log_var, _, x_hat = self.forward(x)
        rec = F.mse_loss(x_hat, x, reduction='mean')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) 
        return rec + kl_div, rec.item()