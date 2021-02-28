# BN T -> S

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import model_dict
from dataset.cifar100 import *
from torchsummary import summary
from models.resnet import ResNet
import torch.optim as optim
import time
import argparse
import matplotlib.pyplot as plt
from munch import Munch

train_batch_size = 128
test_batch_size = 100
n_cls = 100
num_workers = 1
total_epoch = 240
learning_rate = 0.05
lr_decay_epoch = [150, 180, 210]
lr_decay_rate = 0.1
weight_decay = 5e-4
momentum = 0.9
print_freq = 100

parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--model_t', type=str, default='resnet8', choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
parser.add_argument('--path_t', type=str, default=None)
parser.add_argument('--model_s', type=str, default='resnet8', choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
parser.add_argument('--cuda_visible_devices', type=int, default=0)
parser.add_argument('--trial', type=int, default=0)
parser.add_argument('--balance', type=float, default=.2)
parser.add_argument('--temp', type=float, default=3.0)
opt = parser.parse_args()

selection_dict = {
    'wrn401': ['wrn_40_1', './save/models/wrn_40_1_cifar100_lr_0.05_decay_0.0005_trial_0/wrn_40_1_best.pth',
            'wrn_16_1', 1, 1],
    'resnet324': ['resnet32x4', './save/models/resnet32x4_cifar100_lr_0.05_decay_0.0005_trial_0/resnet32x4_best.pth',
            'resnet8x4', 2, 1],
    'vgg19': ['vgg19', ' ./save/models/vgg19_cifar100_lr_0.05_decay_0.0005_trial_0/vgg19_best.pth',
            'vgg11', 2, 1],
    'resnet110_20': ['resnet110', ' ./save/models/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth',
            'resnet20', 2, 1],
    'resnet110_32:':['resnet110', ' ./save/models/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth',
            'resnet32', 3, 1]
}

is_pycharm = False
if is_pycharm:
    options = list(selection_dict.keys())
    arguments = selection_dict[options[1]]
    opt.model_t = arguments[0]
    opt.path_t = arguments[1]
    opt.model_s = arguments[2]
    opt.cuda_visible_devices = arguments[3]
    opt.trial = arguments[4]

print(opt)
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_visible_devices)

balance = opt.balance
model_t = model_dict[opt.model_t](num_classes=100)
model_s = model_dict[opt.model_s](num_classes=100)

path_t = opt.path_t
trial = 0
r = 1
a = 0
b = 1
p = 1
d = 1
kd_T = 4
fm_beta = 1e-3
fit_alpha = 0.2

model_t.load_state_dict(torch.load(path_t)['model'])
model_t.eval()

# rand_value = torch.rand((2, 3, 32, 32))
# feature, logit = model_s(rand_value, is_feat=True, preact=True)
train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=train_batch_size,
                                                                        num_workers=num_workers,
                                                                        is_instance=True)
if torch.cuda.is_available():
    model_s.cuda()
    model_t.cuda()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(epoch, learning_rate, lr_decay_epochs,  optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    if steps > 0:
        new_lr = learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True, max_channel=512):
        super(SepConv, self).__init__()
        self.cadinality = int(max_channel / channel_in)
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in*self.cadinality, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in*self.cadinality, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in*self.cadinality, channel_in*self.cadinality, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in*self.cadinality, bias=False),
            nn.Conv2d(channel_in*self.cadinality, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class TempDistiller(nn.Module):
    def __init__(self, in_channels, in_resolution, final_dim=64, num_classes=100):
        super(TempDistiller, self).__init__()
        self.module_list = []
        temp = in_resolution
        if temp <= 4:
            self.module_list.append(SepConv(
                channel_in=in_channels,
                channel_out = final_dim,
                stride=1))
        
        while temp > 4:
            self.module_list.append(SepConv(
                channel_in= in_channels,
                channel_out= final_dim if temp == 8 else in_channels*2 
            ))
#             print(temp)
            in_channels *= 2
            temp /= 2
        
        self.module_list.append(nn.AvgPool2d((4, 4), 1))
        self.module_list = nn.Sequential(*self.module_list)
        self.fc = nn.Linear(final_dim, num_classes)
        
    def forward(self, in_feat):
        out = self.module_list(in_feat)
        out = out.view(out.size(0), -1)
        feature = out
        out_fc = self.fc(feature)
        return out_fc, feature


## added in exp1_8
class FinalClassifier(nn.Module):
    def __init__(self, in_channels, final_dim=64, num_classes=100):
        super(FinalClassifier, self).__init__()
        self.fc2 = nn.Linear(final_dim, num_classes)
#         self.bn = nn.BatchNorm1d(final_dim, affine=True, track_running_stats=True)
    
    def forward(self, x):
        out_fc = self.fc2(x)
        return out_fc, x


class SimpleMLP(nn.Module):
    def __init__(self, input_size, num_classes=100):
        super(SimpleMLP, self).__init__()

        self.simple = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU6(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input):
        return self.simple(input)

# Knowledge distillation Idea
class AlternativeL2Loss(nn.Module):
    def __init__(self):
        super(AlternativeL2Loss, self).__init__()
        pass

    def forward(self, source, target, margin=2):
        loss = ((source + margin) ** 2 * ((source > -margin) & (target <= 0)).float() +
                (source - margin) ** 2 * ((source <= margin) & (target > 0)).float())
        return torch.abs(loss).mean()

# def KDLoss(outputs, targets):
#     log_softmax_outputs = F.log_softmax(outputs/opt.temp, dim=1)
#     softmax_targets = F.softmax(targets/opt.temp, dim=1)
#     return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

def kd_loss(output, target_output):
    output = output / kd_T
    target_output = target_output / kd_T
    target_softmax = torch.softmax(target_output, dim=1)
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_softmax, dim=1))
    return loss_kd

def feature_loss(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()

class BNLoss():

    def __init__(self, module, target_mean, target_var):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.T_mean = target_mean
        self.T_var = target_var

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        
        self.G_kd_loss = self.Gaussian_kd(mean, var, self.T_mean, self.T_var)

    def Gaussian_kd(self, mean, var, T_mean, T_var):
        num = (mean-T_mean)**2 + var
        denom = 2*T_var
        std = torch.sqrt(var)
        T_std = torch.sqrt(T_var)
        return num/denom - torch.log(std/T_std) - 0.5


    def close(self):
        self.hook.remove()

bn_dict = {
    'resnet110':['layer3.17.bn2'],
    'resnet32':['layer3.4.bn2'],
    'resnet20':['layer3.2.bn2'],
    'resnet32x4':['layer3.4.bn2'],
    'resnet8x4':['layer3.0.bn2'],
    'vgg19':['block4.10'],
    'vgg11':['block4.4'],
    'wrn_40_1': ['bn1'],
    'wrn_16_1': ['bn1']
}

bn_param = {
    'resnest32': 1e-4,
    'resnet20' : 1e-4,
    'resnet8x4': 1e-4,
    'vgg11': 1e-4,
    'wrn_16_1': 1e-4
}

# BN
t_bn = bn_dict[opt.model_t]
s_bn = bn_dict[opt.model_s]
bn_alpha = bn_param[opt.model_s]

t_mean = []
t_var = []
for name, module in model_t.named_modules():
    if name in t_bn:
        t_mean.append(module.running_mean)
        t_var.append(module.running_var)

bn_loss_list = []
bn_idx = 0
for name, module in model_s.named_modules():
    if name in s_bn:
        bn_loss_list.append(BNLoss(module, t_mean[bn_idx].detach(), t_var[bn_idx].detach()))
        bn_idx += 1

rand_ = torch.rand((1, 3, 32, 32), dtype=torch.float32).cuda()
out_feat, out_x = model_t(rand_, is_feat=True, preact=False)

# print("Batch size = {}".format(2))
for i, o in enumerate(out_feat):
    print("{} feature: {}".format(i+1, o.shape))
print("final feature: {}".format(out_x.shape))
final_dim = o.shape[1]

conv_distiller = nn.ModuleList([TempDistiller(in_channels=f.size(1), in_resolution=f.size(2), final_dim=final_dim).cuda() for f in out_feat[:-1]])
conv_distiller.append(FinalClassifier(out_feat[-1].size(1), final_dim=final_dim, num_classes=n_cls))

MLP = SimpleMLP(out_feat[-1].size(1), 100).cuda()
# MLP2 = SimpleMLP(out_feat[-1].size(1), 100).cuda()
trainable_models = nn.ModuleList([])
trainable_models.MLP = MLP
trainable_models.model_s = model_s
trainable_models.conv_distiller = conv_distiller
optimizer = optim.SGD(trainable_models.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
criterion_CE = nn.CrossEntropyLoss()
criterion_CE_1 = nn.CrossEntropyLoss()
criterion_FM = AlternativeL2Loss()
criterion_AB = AlternativeL2Loss()
criterion_LS = nn.BCEWithLogitsLoss()
criterion_MSE = nn.MSELoss()

best_accuracy = -1
best_mlp_accuracy = -1
best_ensemble_acc = -1

# merge all trainable models into trainable_models
trainable_models.cuda()

for epoch in range(1, total_epoch+1):
    adjust_learning_rate(epoch, learning_rate, lr_decay_epoch, optimizer)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_ensemble, top5_ensemble = AverageMeter(), AverageMeter()
    loss_ce_ = AverageMeter()
    loss_kd_ = AverageMeter()
    loss_simple_ = AverageMeter()

#     MLP.train() # for final_feature
#   #  MLP2.train()
#     model_s.train()
#     conv_distiller.train()
    trainable_models.train()
    end = time.time()

    avg_loss_ce = 0
    avg_loss_simple = 0
    count = 0
    loss_list = []

    for idx, data in enumerate(train_loader):
        loss_batch_list = []
        input, target, index = data
        input = input.float()
        # target = target.float()
        optimizer.zero_grad()

        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=False)
        feat_s, logit_s = trainable_models.model_s(input, is_feat=True, preact=False)

        cc = 0
        sub_loss = 0
        sub_loss_2 = 0
        sub_loss_3 = 0

        loss_1 = torch.FloatTensor([0.]).cuda()
        loss_2 = torch.FloatTensor([0.]).cuda()
        loss_3 = torch.FloatTensor([0.]).cuda()
        loss_4 = torch.FloatTensor([0.]).cuda()
        loss_bn = torch.FloatTensor([0.]).cuda()
        ensemble = torch.zeros(size=(input.size(0), n_cls)).cuda() # ensemble init (avg)

        # FitNet + SD part
        for idx_, (conv, fs, ft) in enumerate(zip(trainable_models.conv_distiller[::-1], feat_s[::-1], feat_t[::-1])):
            conv_logit_t, conv_feature_t = conv(ft.detach())
            conv_logit_s, conv_feature_s = conv(fs)
            ensemble += conv_logit_s # ensemble
            
            if idx_ == 0:
                last_logit = conv_logit_s
            if idx_ == 1:
                last_before_logit = conv_logit_s
            
            # Feature SP Distillation
            loss_1 += fit_alpha * (kd_loss(conv_logit_s, logit_t.detach()) * (kd_T ** 2))
            
            # FitNet MSE (feature teacher - feature student)
            loss_3 += fm_beta * criterion_AB(fs, ft.detach())
            
            # CE by label (target - student_feature_logit_i)
            loss_4 += (1-fit_alpha) * criterion_CE(conv_logit_s, target)

            cc += 1
    
        # feedback loss
#         loss_2 += fit_alpha * (kd_loss(logit_s, last_logit.detach()) * (kd_T**2))
#         loss_2 += fit_alpha * (kd_loss(logit_s, last_before_logit.detach()) * (kd_T**2))
        
        # BN loss
        for mod in bn_loss_list:
            loss_bn += mod.G_kd_loss.sum() * bn_alpha

        ensemble /= cc # ensemble avg
        loss_batch_list.append(loss_1.item() / cc)
        loss_batch_list.append(loss_2.item() / 2)
        loss_batch_list.append(loss_3.item() / cc)
        loss_batch_list.append(loss_4.item() / cc)
        loss_batch_list.append(loss_bn.item())

        simple_out_t = trainable_models.MLP(feat_t[-1].detach())
        simple_out_s = trainable_models.MLP(feat_s[-1])
        

        loss_simple = ((balance * F.kl_div(F.log_softmax(simple_out_t / kd_T,dim=1), F.softmax(logit_t.detach() / kd_T, dim=1),size_average=False) +
                       (1 - balance) * F.kl_div(F.log_softmax(simple_out_s / kd_T, dim=1),
                                                F.softmax(simple_out_t.detach() / kd_T, dim=1), size_average=False))) * (
                                  kd_T ** 2) / logit_s.size(0)

       # loss_simple = torch.FloatTensor([0.]).cuda()

        loss_batch_list.append(loss_simple.item())
        # loss_simple = (balance * criterion_AB(simple_out_t, logit_t.detach()) +
        #
        #              (1 - balance) * F.kl_div(F.log_softmax(simple_out_s / kd_T, dim=1), F.softmax(simple_out_t.detach() / kd_T, dim=1), size_average=False)) * (kd_T ** 2) / logit_s.size(0) \
        #             + (1 - balance) * F.kl_div(F.log_softmax(logit_s / kd_T, dim=1), F.softmax(simple_out_s / kd_T, dim=1), size_average=False) * (kd_T ** 2) / logit_s.size(0)
        # loss_simple = balance * criterion_CE2(simple_out_t / kd_T, logit_t.detach() / kd_T) + (1 - balance) * criterion_CE2(simple_out_s / kd_T, simple_out_t.detach())
        # loss_simple_ce = criterion_CE_1(simple_out_t, target) + criterion_CE_1(simple_out_s, target) # idea 5
       # loss_simple_ce = criterion_CE_1(simple_out_s, target) # idea 6
        # loss_simple.backward(retain_graph=True)
        loss_kd = (kd_T ** 2) * F.kl_div(F.log_softmax(logit_s / kd_T, dim=1), F.softmax(logit_t.detach() / kd_T, dim=1), size_average=False) / logit_s.size(0)

        loss_ce = criterion_CE(logit_s, target)
        loss_batch_list.append(loss_ce.item())
        loss_fm = criterion_AB(logit_s, logit_t.detach())
        loss_batch_list.append(loss_fm.item())

        acc_1, acc_5 = accuracy(logit_s, target, topk=(1, 5))
        acc_1_ensemble, acc_5_ensemble = accuracy(ensemble, target, topk=(1, 5))
        top1.update(acc_1[0], input.size(0))
        top5.update(acc_5[0], input.size(0))
        top1_ensemble.update(acc_1_ensemble[0], input.size(0))
        top5_ensemble.update(acc_5_ensemble[0], input.size(0))

        count += 1

        loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_bn + loss_simple + loss_ce * 0.1 + loss_fm + loss_kd * 0.9
        losses.update(loss.item())
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        loss_list.append(loss_batch_list)

    print('Epoch: {}, [Train]* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Acc_Ensemble@1 {top1_ens.avg:.3f} Acc_Ensemble@5 {top5_ens.avg:.3f}'
          .format(epoch, top1=top1, top5=top5, top1_ens=top1_ensemble, top5_ens=top5_ensemble))

    loss_list = np.array(loss_list)
    avg_loss_list = np.sum(loss_list, axis=0) / count

    print("ConvDistiller: %.4f, InterFeature: %.4f, Conv_FM: %.4f, Feature_CE: %.4f, BN: %.4f, Loss(simple): %.4f CE: %.4f FM: %.4f" %
          (avg_loss_list[0], avg_loss_list[1], avg_loss_list[2], avg_loss_list[3],avg_loss_list[4], avg_loss_list[5], avg_loss_list[6], avg_loss_list[7]))


    trainable_models.eval()
    with torch.no_grad():
        val_batch_time = AverageMeter()
        val_top1 = AverageMeter()
        val_top5 = AverageMeter()
        val_losses = AverageMeter()
        val_top1_ensemble = AverageMeter()
        val_top5_ensemble = AverageMeter()
        mlp_top1 = AverageMeter()
        mlp_top5 = AverageMeter()
        aux_classifier = [AverageMeter() for _ in range(len(trainable_models.conv_distiller))]
        for idx, (input, target) in enumerate(val_loader):
            input = input.float().cuda()
            target = target.cuda()

            val_feat_s, output = trainable_models.model_s(input, is_feat=True, preact=False)
            loss = criterion_CE(output, target)
            simple_val_out_s = trainable_models.MLP(val_feat_s[-1])
            ensemble = torch.zeros(size=[input.size(0), n_cls]).cuda()
            # ensemble
            cc = 0
            for k, (conv, fs) in enumerate(zip(trainable_models.conv_distiller, val_feat_s)):
                conv_logit_s, conv_feature_s = conv(fs)
                ensemble += conv_logit_s # ensemble
                aux_acc1, _ = accuracy(conv_logit_s, target, topk=(1,5))
                aux_classifier[k].update(aux_acc1[0], input.size(0))
                cc += 1
            ensemble /= cc # ensemble avg


          #  print(cos(simple_val_out_s, output))
#             similarity_t_list.append(cos(simple_val_out_s, output).mean().item())
            val_acc_1, val_acc_5 = accuracy(output, target, topk=(1, 5))
            val_acc_1_ensemble, val_acc_5_ensemble = accuracy(ensemble, target, topk=(1, 5))
            mlp_val_acc_1, mlp_val_acc_5 = accuracy(simple_val_out_s, target, topk = (1, 5))
            val_top1.update(val_acc_1[0], input.size(0))
            val_top5.update(val_acc_5[0], input.size(0))
            val_top1_ensemble.update(val_acc_1_ensemble[0], input.size(0))
            val_top5_ensemble.update(val_acc_5_ensemble[0], input.size(0))
            mlp_top1.update(mlp_val_acc_1[0], input.size(0))
            mlp_top5.update(mlp_val_acc_5[0], input.size(0))

    if best_accuracy < val_top1.avg:
        best_accuracy = val_top1.avg
    if best_mlp_accuracy < mlp_top1.avg:
        best_mlp_accuracy = mlp_top1.avg
    best_ensemble_acc = max(best_ensemble_acc, val_top1_ensemble.avg)
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Acc_Ensemble@1 {top1_ens.avg:.3f} Acc_Ensemble@5 {top5_ens.avg:.3f} mlpAcc@1 {mlp_top1.avg:.3f} mlpAcc@5 {mlp_top5.avg:.3f} Best_accuracy %.4f Best_mlp_accuracy %.4f Best_Ensemble_accuarcy %.4f'
          .format(top1=val_top1, top5=val_top5, top1_ens=val_top1_ensemble, top5_ens=val_top5_ensemble, mlp_top1=mlp_top1, mlp_top5=mlp_top5) % (best_accuracy, best_mlp_accuracy, best_ensemble_acc))
    print(" * ", end='')
    for k, score in enumerate(aux_classifier):
        print("Aux {} Acc1@ {:.3f}".format(k+1, score.avg), end='\t')
    print("Final Acc1@ {:.3f}".format(val_top1.avg), end='\t')
    print("Ensemble Acc1@ {:.3f}".format(val_top1_ensemble.avg))

