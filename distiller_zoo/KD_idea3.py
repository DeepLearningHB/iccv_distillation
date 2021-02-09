from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch

class DistillKLIdea3(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKLIdea3, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, target):
        _, output_index = torch.max(y_t, 1)
        for idx, output in enumerate(output_index):
            if output != target[idx]:
                y_t[idx][target[idx]] += torch.max(y_t[idx])

        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


if __name__ == "__main__":
    a = DistillKLIdea3(3)
    y_s = torch.tensor(
        [[0.1, 0.3, 0.6],
        [0.7, 0.1, 0.2],
        [0.3, 0.1, 0.6]]
    )
    y_t = torch.tensor(
        [[0.2, 0.3, 0.5],
         [0.8, 0.1, 0.1],
         [0.5, 0.2, 0.3]
        ]
    )
    target = torch.tensor([2, 2, 0])
    a(y_s, y_t, target)