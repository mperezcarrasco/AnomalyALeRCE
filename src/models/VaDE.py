import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class vade(nn.Module):
    """Pytorch implementation of a Variational Deep Embedding (VaDE) based on MLP."""

    def __init__(self, args):
        super(vade, self).__init__()
        self.pi_prior = Parameter(torch.ones(args.n_cluster)/args.n_cluster, requires_grad=True)
        self.mu_prior = Parameter(torch.zeros(args.n_cluster, args.z_dim), requires_grad=True)
        self.log_var_prior = Parameter(torch.randn(args.n_cluster, args.z_dim), requires_grad=True)

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
        return torch.tanh(self.dec4(h))
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """Forward pass for the Variational Deep Embedding model."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return mu, log_var, z, x_hat

    def compute_loss(self, x):
        """
        Compute Evidence Lower Bound (ELBO) for the variational deep embedding 
        ref: https://arxiv.org/abs/1611.05148.
        """
        mu, log_var, z, x_hat = self.forward(x)
        rec = F.mse_loss(x_hat, x, reduction='mean')
        
        means = self.mu_prior
        covs = self.log_var_prior.exp()
        p_c = self.pi_prior
        
        gamma = self.compute_pcz(z, p_c)

        h = log_var.exp().unsqueeze(1) + (mu.unsqueeze(1) - means).pow(2)
        h = torch.sum(torch.log(covs) + h / covs, dim=2)
        log_p_z_given_c = 0.5 * torch.sum(gamma * h)
        log_p_c = torch.sum(gamma * torch.log(p_c + 1e-20))
        log_q_c_given_x = torch.sum(gamma * torch.log(gamma + 1e-20))
        log_q_z_given_x = 0.5 * torch.sum(1 + log_var)

        kl_div_z = (log_p_z_given_c - log_q_z_given_x)/x.size(0)
        kl_div_c = (log_q_c_given_x - log_p_c)/x.size(0)
        
        loss = rec + kl_div_z + kl_div_c
        return loss
    
    def compute_pcz(self, z, p_c):
        covs = self.VaDE.log_var_prior.exp()
        means = self.VaDE.mu_prior

        h = (z.unsqueeze(1) - means).pow(2) / covs
        h += torch.log(2*np.pi*covs)
        p_z_c = torch.exp(torch.log(p_c + 1e-20).unsqueeze(0) - 0.5 * torch.sum(h, dim=2)) + 1e-20
        p_z_given_c = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True)
        return p_z_given_c