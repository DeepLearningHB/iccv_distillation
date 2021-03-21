import torch
import torch.nn as nn


# Cross Entropy Loss (Hinton et al.)
def kd_loss(output, target_output, kd_T=3):
    output = output / kd_T
    target_output = target_output / kd_T
    target_softmax = torch.softmax(target_output, dim=1)
    output_log_softmax = torch.log_softmax (output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_softmax, dim=1))
    return loss_kd


# Feature L2 Loss
def feature_loss(fea, target_fea):
    loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()

# Activation Boundary Loss
class AlternativeL2Loss (nn.Module):
    def __init__(self):
        super (AlternativeL2Loss, self).__init__ ()
        pass

    def forward(self, source, target, margin=2):
        loss = ((source + margin) ** 2 * ((source > -margin) & (target <= 0)).float () +
                (source - margin) ** 2 * ((source <= margin) & (target > 0)).float ())
        return torch.abs (loss).mean ()