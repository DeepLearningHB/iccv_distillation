'''
ICCV 2021: Marrying the Best of Both Knowledge: Ensemble-based Latent Matching with Softmax Representation Knowledge Distillation
Official implementation of Latent Matching Distiller (LMD)
'''

import torch.nn as nn
import torch


class SoftmaxRepresentationDistiller (nn.Module):
    def __init__(self, input_size, num_classes=100):
        super (SoftmaxRepresentationDistiller, self).__init__ ()
        self.simple = nn.Sequential (
            nn.Linear(input_size, 128),
            nn.BatchNorm1d (128),
            nn.ReLU6(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_):
        return self.simple(input_)


def SRD(model_s, dataset):
    assert torch.cuda.is_available ()
    if dataset in ['CIFAR-10', 'CIFAR-100']:
        input_tensor_shape = (3, 32, 32)
        n_cls = 100
    # For TinyImageNet
    else:
        input_tensor_shape = (3, 64, 64)
        n_cls = 200
    random_tensor = torch.rand((1, input_tensor_shape[0], input_tensor_shape[1], input_tensor_shape[2]),
                                dtype=torch.float32).cuda ()
    out_feat, out_s = model_s(random_tensor, is_feat=True, preact=True)
    sr_distiller = SoftmaxRepresentationDistiller(out_feat[-1].size(1), num_classes=n_cls).cuda()
    return sr_distiller