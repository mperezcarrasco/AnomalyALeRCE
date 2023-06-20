import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import AverageMeter


class deepsvdd(nn.Module):
    """Pytorch implementation of an autoencoder based on MLP."""

    def __init__(self, args):
        super(deepsvdd, self).__init__()

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
        zs = []
        with torch.no_grad():
            for _, x, _, _ in dataloader:
                x = x.float().to(self.args.device)
                z = self.forward(x)
                zs.append(z.detach())
        zs = torch.cat(zs)
        c = torch.mean(zs, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        self.c = c

    def set_metrics(self):
        # DEFINING METRICS
        self.mse = AverageMeter()

    def compute_loss(self, x):
        """
        Compute MSE Loss.
        """
        z = self.forward(x)
        self.loss = torch.mean(torch.sum((z - self.c) ** 2, dim=1))
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
        score = torch.sum((z - self.c) ** 2, dim=1)
        return score