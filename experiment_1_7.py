# exp1_3 --> SepConv Bug fix (avgpool) + larger bottleneck

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
kd_T = 3.0
fm_beta = 1e-1
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
    def __init__(self, in_channels, in_resolution, num_classes=100):
        super(TempDistiller, self).__init__()
        self.module_list = []
        temp = in_resolution
        while temp > 4:
            self.module_list.append(SepConv(
                channel_in= in_channels,
                channel_out= in_channels * 2
            ))
            print(temp)
            in_channels *= 2
            temp /= 2
        self.module_list.append(nn.AvgPool2d((4, 4), 1))
        self.module_list = nn.Sequential(*self.module_list)
        self.feat_num = num_classes
#         self.fc = nn.Linear(in_channels, num_classes)
        self.fc1 = nn.Linear(in_channels, 256)
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, in_feat):
        out = self.module_list(in_feat)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        feature = out
        out_fc = self.fc2(F.relu(out))
        return out_fc, feature


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

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()


rand_ = torch.rand((1, 3, 32, 32), dtype=torch.float32).cuda()
out_feat, out_x = model_t(rand_, is_feat=True, preact=True)

conv_distiller = nn.ModuleList([TempDistiller(f.size(1), f.size(2)).cuda() for f in out_feat[:-1]])
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
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
if is_pycharm:
    loss_ce_list = []
    loss_kd_list = []
    loss_simple_list = []


print("Batch size = {}".format(2))
for i, o in enumerate(out_feat[:-1]):
    print(o.shape)
    print(conv_distiller[i](o)[0].shape)
print(out_x.shape)

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

    similarity_list = []
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
            feat_t, logit_t = model_t(input, is_feat=True, preact=True)
        feat_s, logit_s = trainable_models.model_s(input, is_feat=True, preact=True)

        cc = 0
        sub_loss = 0
        sub_loss_2 = 0
        sub_loss_3 = 0

        loss_1 = torch.FloatTensor([0.]).cuda()
        loss_2 = torch.FloatTensor([0.]).cuda()
        loss_3 = torch.FloatTensor([0.]).cuda()
        loss_4 = torch.FloatTensor([0.]).cuda()
        ensemble = torch.zeros(size=(input.size(0), n_cls)).cuda() # ensemble init (avg)
        prev_activation = None

        # FitNet + SD part
        for idx_, (conv, fs, ft) in enumerate(zip(trainable_models.conv_distiller[::-1], feat_s[::-1][1:], feat_t[::-1][1:])):
            conv_logit_t, conv_feature_t = conv(ft.detach())
            conv_logit_s, conv_feature_s = conv(fs)
            ensemble += conv_logit_s # ensemble
            
            # Feature SP Distillation
            loss_1 += 0.5 * (kd_loss(conv_logit_t, logit_t.detach()) * (kd_T ** 2))
            loss_1 += 2 * (kd_loss(conv_logit_s, conv_logit_t.detach()) * (kd_T ** 2))
            
            # final feature map KD (final teacher - student_feature_logit_i)
#             loss_1 += balance * (kd_loss(conv_logit_s, logit_t.detach()) * (kd_T ** 2))
    
            # inter-layer distance
            if idx_ == 0:
                loss_2 += criterion_AB(conv_feature_s, conv_feature_t.detach()) * fm_beta
            else:
                loss_2 += torch.dist(conv_feature_s, prev_activation) * 0.05
            
            # FitNet MSE (feature teacher - feature student)
            loss_3 += fm_beta * criterion_AB(fs, ft.detach())
            
            # CE by label (target - student_feature_logit_i)
            loss_4 += (1-fit_alpha) * criterion_CE(conv_logit_s, target)
            
            prev_activation = conv_feature_s.detach().clone()
            cc += 1

        ensemble /= cc # ensemble avg
        loss_batch_list.append(loss_1.item() / cc)
        loss_batch_list.append(loss_2.item() / cc)
        loss_batch_list.append(loss_3.item() / cc)

        simple_out_t = trainable_models.MLP(feat_t[-1].detach())
        simple_out_s = trainable_models.MLP(feat_s[-1])

        similarity_list.append \
            ([cos(simple_out_t, logit_t).mean().item(),
              cos(simple_out_s, simple_out_t).mean().item(),
              cos(simple_out_s, logit_s).mean().item(),
              cos(logit_s, logit_t).mean().item()])
        
         #
        # if epoch == 10:
        #     print(simple_out_t)
        #     print(simple_out_s)
        #     exit()

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

        loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_simple + loss_ce * 0.1 + loss_fm + loss_kd * 0.9
        losses.update(loss.item())
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        loss_list.append(loss_batch_list)

#         if idx % print_freq == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                   'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
#                   'Acc_Ensemble@1 {top1_ens.val:.3f} ({top1_ens.avg:.3f})\t'
#                   'Acc_Ensemble@5 {top5_ens.val:.3f} ({top5_ens.avg:.3f})'.format(
#                 epoch, idx, len(train_loader), batch_time=batch_time,
#                 data_time=data_time, loss=losses, top1=top1, top5=top5, top1_ens=top1_ensemble, top5_ens=top5_ensemble))
#             sys.stdout.flush()

    print('Epoch: {}, [Train]* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Acc_Ensemble@1 {top1_ens.avg:.3f} Acc_Ensemble@5 {top5_ens.avg:.3f}'
          .format(epoch, top1=top1, top5=top5, top1_ens=top1_ensemble, top5_ens=top5_ensemble))
    print('cos(simple_out_t, logit_s) | cos(simple_out_t, simple_out_s) | cos(simple_out_s, logit_s) | cos(logit_s, logit_t)')
    print(np.array(similarity_list).mean(axis=0))
    loss_list = np.array(loss_list)
    avg_loss_list = np.sum(loss_list, axis=0) / count

    print("Conv Distiller: %.4f, InterFeature: %.4f, Conv_FM: %.4f,  Loss(simple): %.4f CE: %.4f FM: %.4f" %
          (avg_loss_list[0], avg_loss_list[1], avg_loss_list[2],avg_loss_list[3],avg_loss_list[4], avg_loss_list[5]))
    # if False:
    #
    # #if is_pycharm and (epoch % 5 == 0 and epoch > 30) or epoch == 3:
    #     print(loss_ce_list)
    #     plt.plot(list(range(len(loss_ce_list))), loss_ce_list, label='ce')
    #     plt.plot(list(range(len(loss_kd_list))), loss_kd_list, label='kd')
    #     plt.plot(list(range(len(loss_simple_list))), loss_simple_list, label='ours')
    #     plt.legend()
    #     plt.show()

    trainable_models.eval()
    similarity_t_list = []
    with torch.no_grad():
        val_batch_time = AverageMeter()
        val_top1 = AverageMeter()
        val_top5 = AverageMeter()
        val_losses = AverageMeter()
        val_top1_ensemble = AverageMeter()
        val_top5_ensemble = AverageMeter()
        mlp_top1 = AverageMeter()
        mlp_top5 = AverageMeter()
        for idx, (input, target) in enumerate(val_loader):
            input = input.float().cuda()
            target = target.cuda()

            val_feat_s, output = trainable_models.model_s(input, is_feat=True, preact=True)
            loss = criterion_CE(output, target)
            simple_val_out_s = trainable_models.MLP(val_feat_s[-1])
            ensemble = torch.zeros(size=[input.size(0), n_cls]).cuda()
            # ensemble
            cc = 0
            for conv, fs in zip(trainable_models.conv_distiller, val_feat_s):
                conv_logit_s, conv_feature_s = conv(fs)
                ensemble += conv_logit_s # ensemble        
                cc += 1
            ensemble /= cc # ensemble avg


          #  print(cos(simple_val_out_s, output))
            similarity_t_list.append(cos(simple_val_out_s, output).mean().item())
            val_acc_1, val_acc_5 = accuracy(output, target, topk=(1, 5))
            val_acc_1_ensemble, val_acc_5_ensemble = accuracy(ensemble, target, topk=(1, 5))
            mlp_val_acc_1, mlp_val_acc_5 = accuracy(simple_val_out_s, target, topk = (1, 5))
            val_top1.update(val_acc_1[0], input.size(0))
            val_top5.update(val_acc_5[0], input.size(0))
            val_top1_ensemble.update(val_acc_1_ensemble[0], input.size(0))
            val_top5_ensemble.update(val_acc_5_ensemble[0], input.size(0))
            mlp_top1.update(mlp_val_acc_1[0], input.size(0))
            mlp_top5.update(mlp_val_acc_5[0], input.size(0))

#             if idx % print_freq == 0:
#                 print('Test: [{0}/{1}]\t'
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                       'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
#                       'Acc_Ensemble@1 {top1_ens.val:.3f} ({top1_ens.avg:.3f})\t'
#                       'Acc_Ensemble@5 {top5_ens.val:.3f} ({top5_ens.avg:.3f})\t'
#                       'AccMLP@1 {mlp_top1.val:.3f} ({mlp_top1.avg:.3f})\t'
#                       'AccMLP@5 {mlp_top5.val:.3f} ({mlp_top5.avg:.3f})'.format(
#                        idx, len(val_loader), batch_time=batch_time, loss=losses,
#                        top1=val_top1, top5=val_top5, top1_ens=val_top1_ensemble, top5_ens=val_top5_ensemble, mlp_top1=mlp_top1, mlp_top5=mlp_top5))
    if best_accuracy < val_top1.avg:
        best_accuracy = val_top1.avg
    if best_mlp_accuracy < mlp_top1.avg:
        best_mlp_accuracy = mlp_top1.avg
    best_ensemble_acc = max(best_ensemble_acc, val_top1_ensemble.avg)
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Acc_Ensemble@1 {top1_ens.avg:.3f} Acc_Ensemble@5 {top5_ens.avg:.3f} mlpAcc@1 {mlp_top1.avg:.3f} mlpAcc@5 {mlp_top5.avg:.3f} Best_accuracy %.4f Best_mlp_accuracy %.4f'
          .format(top1=val_top1, top5=val_top5, top1_ens=val_top1_ensemble, top5_ens=val_top5_ensemble, mlp_top1=mlp_top1, mlp_top5=mlp_top5) % (best_accuracy, best_mlp_accuracy))
    print("cos(simple_out_s, logit_s)", sum(similarity_t_list) / len(similarity_t_list))

