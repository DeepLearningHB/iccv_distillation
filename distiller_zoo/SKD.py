import torch
import torch.nn as nn
import torch.nn.functional as F

class SKDLoss(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T, n_cls, multiplier=2):
        super(SKDLoss, self).__init__()
        self.T = T
        self.n_cls = n_cls
        self.multiplier = multiplier
    def forward(self, y_s, y_t):
        y_s = F.layer_norm(y_s, torch.Size((self.n_cls,)), None, None, 1e-7) * self.multiplier
        y_t = F.layer_norm(y_t, torch.Size((self.n_cls,)), None, None, 1e-7) * self.multiplier
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss