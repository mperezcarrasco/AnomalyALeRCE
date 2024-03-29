import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.utils import AverageMeter


class ae(nn.Module):
    """Pytorch implementation of an autoencoder based on MLP."""

    def __init__(self, args):
        super(ae, self).__init__()

        self.args = args
        #Encoder Architecture
        self.enc1 = nn.Linear(args.in_dim, 512)
        self.encbn1 = nn.BatchNorm1d(512)
        self.enc2 = nn.Linear(512, 256)
        self.encbn2 = nn.BatchNorm1d(256)
        self.enc3 = nn.Linear(256, 128)
        self.encbn3 = nn.BatchNorm1d(128)
        self.enc4 = nn.Linear(128, args.z_dim, bias=False)

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
        return self.enc4(h)

    def decode(self, x):
        h = F.leaky_relu(self.decbn1(self.dec1(x)))
        h = F.leaky_relu(self.decbn2(self.dec2(h)))
        h = F.leaky_relu(self.decbn3(self.dec3(h)))
        return torch.tanh(self.dec4(h))
    
    def forward(self, x):
        """Forward pass over the network architecture"""
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat

    def set_metrics(self):
        # DEFINING METRICS
        self.rec = AverageMeter()

    def compute_loss(self, x):
        """
        Compute MSE Loss.
        """
        _, x_hat = self.forward(x)
        self.loss = F.mse_loss(x_hat, x, reduction='mean')
        return self.loss

    def compute_metrics(self, x):
        """
        Compute metrics (only reconstruction for AE).
        """
        self.rec.update(self.loss.item(), x.size(0))
        metrics = {'Loss': self.rec.avg,
                   'Reconstruction': self.rec.avg}
        return metrics
    
    def compute_anomaly_score(self, x):
        """
        Computing the anomaly score for each sample x.
        """
        _, x_hat = self.forward(x)
        score = F.mse_loss(x_hat, x, reduction='none')
        return torch.sum(score, dim=1)