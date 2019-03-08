import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.functional import mse_loss
from utils import log_det_jacobian, compute_jacobian, jacobian

import matplotlib.pyplot as plt

EPS = 1e-8

class EAE(nn.Module):
    def __init__(self):
        super(EAE, self).__init__()
        nh = 5
        self.encoder = nn.Sequential(
            nn.Linear(2, nh),
            nn.ELU(),
            nn.Linear(nh, 2),
            nn.ELU()
        )
        self.phi1 = nn.Sequential(
            nn.Linear(1, nh),
            nn.ELU(),
            nn.Linear(nh, 1),
            nn.Sigmoid()
        )
        self.phi2 = nn.Sequential(
            nn.Linear(1, nh),
            nn.ELU(),
            nn.Linear(nh, 1),
            nn.Sigmoid()
        )
        self.f1 = nn.Sequential(
            nn.Linear(1, nh),
            nn.ELU(),
            nn.Linear(nh, nh),
            nn.ELU(),
            nn.Linear(nh, 1)
        )
        self.f2 = nn.Sequential(
            nn.Linear(2, nh),
            nn.ELU(),
            nn.Linear(nh, nh),
            nn.ELU(),
            nn.Linear(nh, 1)
        )

    def forward(self, x):
        y = self.encoder(x)
        # y1 = y[:, 0].reshape(-1, 1)
        # y2 = y[:, 1].reshape(-1, 1)
        y1, y2 = torch.split(y, [1, 1], dim=1)
        z1 = self.phi1(y1)
        z2 = self.phi2(y2)
        x1_hat = self.f1(z1)
        x2_hat = self.f2(torch.cat([x1_hat, z2], dim=1))
        
        x_hat = torch.cat([x1_hat, x2_hat], dim=1)
        z = torch.cat([z1, z2], dim=1)

        return z, x_hat

def train(x, **params):
    epochs = params['epochs']
    lr = params['lr']
    lamb = params['lamb']
    
    loss_np = []
    eae = EAE()
    optimizer = torch.optim.Adam(eae.parameters(), lr=lr)
    for i in range(epochs):
        z, x_hat = eae.forward(x)
        log_det_j = log_det_jacobian(x, z)
        rec = mse_loss(x, x_hat)
        loss = -torch.mean(log_det_j) + lamb*rec
    
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        loss_np.append(loss.data.numpy())
    
    z_np = z.data.numpy()
    x_hat_np = x_hat.data.numpy()
        
    return loss_np, z_np, x_hat_np

