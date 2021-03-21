'''
ICCV 2021: Marrying the Best of Both Knowledge: Ensemble-based Latent Matching with Softmax Representation Knowledge Distillation
Official implementation of Latent Matching Distiller (LMD)
'''
import torch.nn as nn
import torch


class SepConv (nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True, max_channel=256):
        super (SepConv, self).__init__ ()
        self.cadinality = int (max_channel / channel_in) if channel_in <= max_channel else 0.5
        self.op = nn.Sequential (
            nn.Conv2d (channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                       groups=channel_in, bias=False),
            nn.Conv2d (channel_in, int(channel_in * self.cadinality), kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d (int(channel_in * self.cadinality), affine=affine),
            nn.ReLU (inplace=False),
            nn.Conv2d (int(channel_in * self.cadinality), int(channel_in * self.cadinality), kernel_size=kernel_size, stride=1,
                       padding=padding, groups=int(channel_in * self.cadinality), bias=False),
            nn.Conv2d (int(channel_in * self.cadinality), channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d (channel_out, affine=affine),
            nn.ReLU (inplace=False),
        )

    def forward(self, x):
        return self.op (x)

class LatentMatchingDistiller (nn.Module):
    def __init__(self, in_channels, in_resolution, final_dim=32, num_classes=100):
        super (LatentMatchingDistiller, self).__init__ ()
        self.module_list = []
        temp = in_resolution
        max_feat_num = 256
        if temp <= 4:
            self.module_list.append (SepConv (
                channel_in=in_channels,
                channel_out=final_dim,
                stride=1, padding=1))

        while temp > 4:
            self.module_list.append (SepConv (
                channel_in=in_channels,
                channel_out=final_dim if temp == 8 else min(in_channels * 2, max_feat_num)
            ))
            #             print(temp)
            in_channels = final_dim if temp == 8 else min(in_channels * 2, max_feat_num)
            temp /= 2

        self.module_list.append (nn.AdaptiveAvgPool2d (output_size=(1, 1)))
        self.module_list = nn.Sequential (*self.module_list)
        self.fc = nn.Linear (final_dim, num_classes)
        self.bn = nn.BatchNorm1d (final_dim, affine=True, track_running_stats=True)

    def forward(self, in_feat):
        out = self.module_list (in_feat)
        out = out.view (out.size (0), -1)
        feature = out
        out_fc = self.fc (feature)
        return out_fc, feature



## added in exp1_8
class LatentMatchingDistiller_MLP (nn.Module):
    def __init__(self, in_channels, num_classes=100):
        super (LatentMatchingDistiller_MLP, self).__init__ ()
        self.fc2 = nn.Linear (in_channels, num_classes)

    def forward(self, x):
        out_fc = self.fc2 (x)
        return out_fc, x

def LMD(model_s, final_dim, dataset):
    assert torch.cuda.is_available()
    if dataset in ['CIFAR-10', 'CIFAR-100']:
        input_tensor_shape = (3, 32, 32)
        n_cls = 100
    # For TinyImageNet
    else:
        input_tensor_shape = (3, 64, 64)
        n_cls = 200
    random_tensor = torch.rand((1, input_tensor_shape[0], input_tensor_shape[1],input_tensor_shape[2]),
                               dtype=torch.float32).cuda()
    out_feat, out_s = model_s(random_tensor, is_feat=True, preact=True)

    lm_distiller = nn.ModuleList([LatentMatchingDistiller(in_channels=f.size(1), in_resolution=f.size(2), num_classes=n_cls,
                                                           final_dim=final_dim).cuda() for f in out_feat[:-1]])
    lm_distiller.append(LatentMatchingDistiller_MLP(out_feat[-1].size(1), num_classes=n_cls).cuda())
    return lm_distiller
