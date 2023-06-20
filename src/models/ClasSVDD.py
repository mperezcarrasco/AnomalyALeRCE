import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import AverageMeter
import numpy as np

class classvdd(nn.Module):
    """Pytorch implementation of an autoencoder based on MLP."""

    def __init__(self, args):
        super(classvdd, self).__init__()

        self.args = args
        #Encoder Architecture
        self.enc1 = nn.Linear(args.in_dim, 512)
        self.encbn1 = nn.BatchNorm1d(512)
        self.enc2 = nn.Linear(512, 256)
        self.encbn2 = nn.BatchNorm1d(256)
        self.enc3 = nn.Linear(256, 128)
        self.encbn3 = nn.BatchNorm1d(128)
        self.enc4 = nn.Linear(128, args.z_dim, bias=False)

    def encode(self, x):
        h = F.leaky_relu(self.encbn1(self.enc1(x)))
        h = F.leaky_relu(self.encbn2(self.enc2(h)))
        h = F.leaky_relu(self.encbn3(self.enc3(h)))
        return self.enc4(h)
    
    def forward(self, x):
        """Forward pass over the network architecture"""
        z = self.encode(x)
        return z

    def set_c(self, dataloader, eps=0.01):
        """Initializing the center for the hypersphere"""
        latents, labels = self.get_latent_space(dataloader)
        c = []
        for i in range(len(np.unique(labels))):
            ixs = np.where(labels == i)
            c.append(torch.mean(latents[ixs], dim=0))
        c = torch.stack(c)
        for i in range(len(c)):
            c[i][(abs(c[i]) < eps) & (c[i] < 0)] = -eps
            c[i][(abs(c[i]) < eps) & (c[i] > 0)] = eps
        self.c = c.to(self.args.device)

    def get_latent_space(self, dataloader):
        """Get the latent space and labels from the dataloader for the initialization."""
        latents = []
        labels = []
        with torch.no_grad():
            for (_, x, y, _) in dataloader:
                x, y = x.to(self.args.device).float(), y.long()
                z = self.forward(x)
                latents.append(z.detach().cpu())
                labels.append(y)
        return torch.cat(latents), torch.cat(labels)

    def set_metrics(self):
        # DEFINING METRICS
        self.mse = AverageMeter()

    def compute_loss(self, x, y):
        """
        Compute MSE Loss.
        """
        z = self.forward(x)
        self.loss = torch.mean(torch.sum((z - self.c[y]) ** 2, dim=1))
        return self.loss

    def compute_metrics(self, x):
        """
        Compute metrics (only reconstruction for AE).
        """
        self.mse.update(self.loss.item(), x.size(0))
        metrics = {'Loss': self.mse.avg,
                   'Distance': self.mse.avg}
        return metrics
    
    def compute_anomaly_score(self, x):
        """
        Computing the anomaly score for each sample x.
        """
        z = self.forward(x)
        score = torch.min(torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2), dim=1)[0]
        return score