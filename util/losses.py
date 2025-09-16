import numpy as np
import torch
import torch.nn as nn

class ObjectNormalizedL2Loss(nn.Module):

    def __init__(self):
        super(ObjectNormalizedL2Loss, self).__init__()

    def forward(self, output, dmap, num_objects):
        return ((output - dmap) ** 2).sum() / num_objects

def recons(output, gt_density, device):
    mask = np.random.binomial(n=1, p=0.8, size=[384,384])
    masks = np.tile(mask,(output.shape[0],1))
    masks = masks.reshape(output.shape[0], 384, 384)
    masks = torch.from_numpy(masks).to(device)
    loss = (output - gt_density) ** 2
    loss = (loss * masks / (384*384)).sum() / output.shape[0]
    return loss

# def kl(mu, logvar):
#     return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()), 0)

def kl(mu, logvar):
    div = torch.distributions.kl_divergence(mu, logvar)
    return div.mean()

def l2(x, y):
    if torch.isnan(x).any() or torch.isinf(x).any():
        raise ValueError("Input x contains NaN or inf values!")
    if torch.isnan(y).any() or torch.isinf(y).any():
        raise ValueError("Input y contains NaN or inf values!")
    
    # return nn.L1Loss()(x, y) / x.shape[0]
    return nn.MSELoss(reduction='mean')(x, y)
