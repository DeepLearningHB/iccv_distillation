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
from itertools import product

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
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# class SimpleMLP(nn.Module):
#     def __init__(self, input_size, num_classes=100):
#         super(SimpleMLP, self).__init__()
#
#         self.simple = nn.Sequential(
#             nn.Linear(input_size, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU6(),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU6(),
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU6(),
#             nn.Linear(128, 100)
#         )
#
#     def forward(self, input):
#         return self.simple(input)


#        return self.simple(input)
class TempDistiller(nn.Module):
    def __init__(self, in_channels, in_resolution):
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
        self.avg_pool = nn.AvgPool2d(4, 4)
        self.module_list = nn.Sequential(*self.module_list)
        self.feat_num = 100

        self.fc1 = nn.Linear(in_channels, 256)
        self.fc2 = nn.Linear(256, 100)


    def forward(self, in_feat):
        out = self.module_list(in_feat)
        out = self.avg_pool(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        feature = out
        out_fc = self.fc2(F.relu(out))
        return out_fc, feature

class SimpleMLP(nn.Module):
    def __init__(self, input_size, num_classes=100):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, input):
        feature = self.fc1(input)
        out = self.bn1(feature)
        out = self.relu(out)
        return feature, self.fc2(out)

# Knowledge distillation Idea
class AlternativeL2Loss(nn.Module):
    def __init__(self):
        super(AlternativeL2Loss, self).__init__()
        pass

    def forward(self, source, target, margin=2):
        loss = ((source + margin) ** 2 * ((source > -margin) & (target <= 0)).float() +
                (source - margin) ** 2 * ((source <= margin) & (target > 0)).float())
        return torch.abs(loss).mean()

class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""
    def __init__(self, w_d=25, w_a=50):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)

        # RKD distance loss
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        # RKD Angle loss
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        loss = self.w_d * loss_d + self.w_a * loss_a

        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res


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
dataset = 'cifar100'

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
opt = parser.parse_args()
#
# a_list = [0.25, 0.75]
# b_list = [0.25, 0.75]
# c_list = [0.25, 0.75]

a_list = [0.75]
b_list = [0.25]
c_list = [0.25]

combi_list = [a_list, b_list, c_list]
combi_list = list(product(*combi_list))




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


class JSD (nn.Module):
    def __init__(self):
        super (JSD, self).__init__ ()

    def forward(self, net_1_logits, net_2_logits):
        net_1_probs = F.softmax (net_1_logits, dim=1)
        net_2_probs = F.softmax (net_2_logits, dim=1)

        m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0
        loss += F.kl_div (F.log_softmax (net_1_logits, dim=1), m, size_average=False)
        loss += F.kl_div (F.log_softmax (net_2_logits, dim=1), m, size_average=False)

        return (0.5 * loss)

print(opt)
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_visible_devices)

#file = open(opt.model_t+'_'+opt.model_s+"_v4.txt", 'w')

for a, b, c in combi_list:
    balance = opt.balance
    model_t = model_dict[opt.model_t](num_classes=100)
    model_s = model_dict[opt.model_s](num_classes=100)

    path_t = opt.path_t
    kd_T = 4
    model_t.load_state_dict(torch.load(path_t)['model'])
    model_t.eval()

    rand_value = torch.rand((2, 3, 32, 32))

    feature, logit = model_s(rand_value, is_feat=True, preact=True)

    train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=train_batch_size,
                                                                            num_workers=num_workers,
                                                                            is_instance=True)
    if torch.cuda.is_available():
        model_s.cuda()
        model_t.cuda()



    rand_ = torch.rand((2, 3, 32, 32), dtype=torch.float32).cuda()
    out_feat, out_x = model_t(rand_, is_feat=True, preact=True)

    conv_distiller = nn.ModuleList([TempDistiller(f.size(1), f.size(2)).cuda() for f in out_feat[:-1]])
    MLP = SimpleMLP(out_feat[-1].size(1), 100).cuda()
    # MLP2 = SimpleMLP(out_feat[-1].size(1), 100).cuda()
    trainable_models = nn.ModuleList([])
    trainable_models.append(MLP)
    # trainable_models.append(MLP2)
    trainable_models.append(model_s)
    trainable_models.extend(conv_distiller)
    optimizer = optim.SGD(trainable_models.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    criterion_CE = nn.CrossEntropyLoss()
    criterion_CE_1 = nn.CrossEntropyLoss()
    criterion_FM = AlternativeL2Loss()
    criterion_AB = AlternativeL2Loss()
    criterion_JS = JSD()
    best_accuracy = -1
    best_mlp_accuracy = -1
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    if is_pycharm:
        loss_ce_list = []
        loss_kd_list = []
        loss_simple_list = []

    #temp

    for epoch in range(1, total_epoch+1):
        adjust_learning_rate(epoch, learning_rate, lr_decay_epoch, optimizer)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        loss_ce_ = AverageMeter()
        loss_kd_ = AverageMeter()
        loss_simple_ = AverageMeter()


        MLP.train()
      #  MLP2.train()
        model_s.train()
        conv_distiller.train()
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
            feat_s, logit_s = model_s(input, is_feat=True, preact=True)

            cc = 0
            sub_loss = 0
            sub_loss_2 = 0
            sub_loss_3 = 0

            loss_1 = torch.FloatTensor([0.]).cuda()
            loss_2 = torch.FloatTensor([0.]).cuda()
            loss_3 = torch.FloatTensor([0.]).cuda()
            loss_4 = torch.FloatTensor([0.]).cuda()

            # prev_activation = None

            # for idx_, (conv, fs, ft) in enumerate(zip(conv_distiller[1:], feat_s[1:], feat_t[1:])):
            #     conv_logit_t, conv_feature_t = conv(ft.detach())
            #     conv_logit_s, conv_feature_s = conv(fs)
            #     if idx_ != 0 and idx_ != len(conv_distiller) - 1:
            #         loss_3 += torch.dist(prev_activation, conv_feature_s.detach()) * 0.05
            #     elif idx_ == len(conv_distiller) - 1:
            #         loss_3 += criterion_AB(conv_feature_s, conv_feature_t.detach())
            #     loss_1 += F.kl_div(F.log_softmax(conv_logit_t / kd_T, dim=1), F.softmax(logit_t.detach() / kd_T, dim=1), size_average=False) * (kd_T ** 2) / conv_logit_t.size(0)
            #     loss_2 += F.kl_div(F.log_softmax(conv_logit_s / kd_T, dim=1), F.softmax(conv_logit_t.detach() / kd_T, dim=1), size_average=False) * (kd_T ** 2) / conv_logit_s.size(0)
            #     loss_4 += criterion_CE(conv_logit_s, target)
            #     prev_activation = conv_feature_s
            #     cc += 1




            simple_out_t_feat, simple_out_t = MLP(feat_t[-1].detach())
            simple_out_s_feat, simple_out_s = MLP(feat_s[-1])

            similarity_list.append \
                ([cos(simple_out_t, logit_t).mean().item(),
                  cos(simple_out_s, simple_out_t).mean().item(),
                  cos(simple_out_s, logit_s).mean().item(),
                  cos(logit_s, logit_t).mean().item()])

            loss_simple = (balance * F.kl_div (F.log_softmax (simple_out_t / kd_T),
                                               F.softmax (logit_t.detach () / kd_T, dim=1), size_average=False) +
                           (1 - balance) * F.kl_div (F.log_softmax (simple_out_s / kd_T, dim=1),
                                                     F.softmax (simple_out_t.detach () / kd_T, dim=1),
                                                     size_average=False)) * (kd_T ** 2) / logit_s.size (0)
            prev_activation = None
            prev_activation_t = None

            for idx_, (conv, fs, ft) in enumerate(zip(conv_distiller[1:], feat_s[1:], feat_t[1:])):
                conv_logit_t, conv_feature_t = conv(ft.detach())
                conv_logit_s, conv_feature_s = conv(fs)
                if idx_ != 0 and idx_ != len(conv_distiller) - 1:
                    loss_3 += torch.dist(prev_activation, conv_logit_s.detach()) * 0.05
                elif idx_ == len(conv_distiller) - 1:
                    loss_3 += criterion_AB(conv_logit_s, conv_logit_t.detach())
                loss_1 += F.kl_div(F.log_softmax(conv_logit_t / kd_T, dim=1), F.softmax(logit_t.detach() / kd_T, dim=1), size_average=False) * (kd_T ** 2) / conv_logit_t.size(0)
                loss_2 += F.kl_div(F.log_softmax(conv_logit_s / kd_T, dim=1), F.softmax(conv_logit_t.detach() / kd_T, dim=1), size_average=False) * (kd_T ** 2) / conv_logit_s.size(0)
                loss_4 += criterion_CE(conv_logit_s, target)
                prev_activation = conv_logit_s
                cc += 1

#             for idx_, (conv, fs, ft) in enumerate(zip(conv_distiller, feat_s, feat_t)):
#                 conv_logit_t, conv_feature_t = conv(ft.detach())
#                 conv_logit_s, conv_feature_s = conv(fs)
#                 if idx_ != 0 and idx_ != len (conv_distiller) - 1:
#                     loss_2 += criterion_AB(prev_activation, conv_logit_s.detach())
#                     loss_3 += criterion_AB(prev_activation_t, conv_logit_t.detach())
#
#                     #loss_3 += torch.dist(prev_activation_t, conv_logit_t.detach()) * 0.05
#                 elif idx_ == len(conv_distiller) - 1:
#                     loss_2 += criterion_AB(conv_logit_t,simple_out_s.detach())
#
# #                    loss_2 += torch.dist(conv_logit_s, simple_out_s.detach()) * 0.05
#                     loss_3 += criterion_AB(conv_logit_s, simple_out_t.detach())
#
# #                    loss_3 += torch.dist(conv_logit_t, simple_out_t.detach()) * 0.05
#                 loss_1 += criterion_JS(conv_logit_s / kd_T, conv_logit_t / kd_T) * (kd_T ** 2) / logit_s.size(0)
#                 prev_activation = conv_logit_s
#                 prev_activation_t = conv_logit_t
#                 cc += 1

            loss_batch_list.append(loss_1.item() / cc)
            loss_batch_list.append(loss_2.item() / cc)
            loss_batch_list.append(loss_3.item() / cc)

            #
            # if epoch == 10:
            #     print(simple_out_t)
            #     print(simple_out_s)
            #     exit()



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
    #        loss_kd = (kd_T ** 2) * F.kl_div(F.log_softmax(logit_s / kd_T, dim=1), F.softmax(logit_t.detach() / kd_T, dim=1), size_average=False) / logit_s.size(0)

            loss_ce = criterion_CE(logit_s, target)
            loss_batch_list.append(loss_ce.item())
            # loss_fm =
            loss_fm = (kd_T ** 2) * F.kl_div(F.log_softmax(logit_s / kd_T, dim=1), F.softmax(logit_t.detach() / kd_T, dim=1), size_average=False) / logit_s.size(0)
            # loss_fm = criterion_AB(feat_s[-1], feat_t[-1].detach())
            loss_batch_list.append(loss_fm.item())

            acc_1, acc_5 = accuracy(logit_s, target, topk=(1, 5))
            top1.update(acc_1[0], input.size(0))
            top5.update(acc_5[0], input.size(0))

            count += 1

            #loss = 5 * loss_1 + 0.2 * loss_2 + 0.8 * loss_3 + loss_simple + loss_ce + loss_fm
            loss = 0.2 * loss_1 + 0.8 * loss_2 + loss_3 + 0.1 * loss_4 + loss_simple + 0.9 * loss_fm + 0.1 * loss_ce
            losses.update(loss.item())
            loss.backward()
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            loss_list.append(loss_batch_list)

            if idx % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, idx, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
                sys.stdout.flush()


        print('[Train]* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        print('cos(simple_out_t, logit_s) | cos(simple_out_t, simple_out_s) | cos(simple_out_s, logit_s) | cos(logit_s, logit_t)')
        print(np.array(similarity_list).mean(axis=0))
        loss_list = np.array(loss_list)
        avg_loss_list = np.sum(loss_list, axis=0) / count

        print("Conv(L_T, C_T): %.4f Conv(C_S, C_T)): %.4f Conv_FM: %.4f,  Loss(simple): %.4f CE: %.4f FM: %.4f" %
              (avg_loss_list[0], avg_loss_list[1], avg_loss_list[2], avg_loss_list[3],avg_loss_list[4],avg_loss_list[5]))
        # if False:
        #
        # #if is_pycharm and (epoch % 5 == 0 and epoch > 30) or epoch == 3:
        #     print(loss_ce_list)
        #     plt.plot(list(range(len(loss_ce_list))), loss_ce_list, label='ce')
        #     plt.plot(list(range(len(loss_kd_list))), loss_kd_list, label='kd')
        #     plt.plot(list(range(len(loss_simple_list))), loss_simple_list, label='ours')
        #     plt.legend()
        #     plt.show()


        model_s.eval()
        MLP.eval()
        conv_distiller.eval()
        similarity_t_list = []
        with torch.no_grad():
            val_batch_time = AverageMeter()
            val_top1 = AverageMeter()
            val_top5 = AverageMeter()
            val_losses = AverageMeter()
            mlp_top1 = AverageMeter()
            mlp_top5 = AverageMeter()
            for idx, (input, target) in enumerate(val_loader):
                input = input.float().cuda()
                target = target.cuda()

                val_feat_s, output = model_s(input, is_feat=True, preact=True)
                loss = criterion_CE(F.softmax(output), target)
                simple_val_feat_s, simple_val_out_s = MLP(val_feat_s[-1])

              #  print(cos(simple_val_out_s, output))
                similarity_t_list.append(cos(simple_val_out_s, output).mean().item())
                val_acc_1, val_acc_5 = accuracy(output, target, topk=(1, 5))
                mlp_val_acc_1, mlp_val_acc_5 = accuracy(simple_val_out_s, target, topk = (1, 5))
                val_top1.update(val_acc_1[0], input.size(0))
                val_top5.update(val_acc_5[0], input.size(0))
                mlp_top1.update(mlp_val_acc_1[0], input.size(0))
                mlp_top5.update(mlp_val_acc_5[0], input.size(0))

                if idx % print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                          'AccMLP@1 {mlp_top1.val:.3f} ({mlp_top1.avg:.3f})\t'
                          'AccMLP@5 {mlp_top5.val:.3f} ({mlp_top5.avg:.3f})'.format(
                           idx, len(val_loader), batch_time=batch_time, loss=losses,
                           top1=val_top1, top5=val_top5, mlp_top1=mlp_top1, mlp_top5=mlp_top5))

        if best_accuracy < val_top1.avg:
            best_accuracy = val_top1.avg
        if best_mlp_accuracy < mlp_top1.avg:
            best_mlp_accuracy = mlp_top1.avg
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} mlpAcc@1 {mlp_top1.avg:.3f} mlpAcc@5 {mlp_top5.avg:.3f} Best_accuracy %.4f Best_mlp_accuracy %.4f'
              .format(top1=val_top1, top5=val_top5, mlp_top1=mlp_top1, mlp_top5=mlp_top5) % (best_accuracy, best_mlp_accuracy))
        print("cos(simple_out_s, logit_s)", sum(similarity_t_list) / len(similarity_t_list))

    break
    # file.write("a: %f b: %f c: %f [Best Accuracy:%.4f]\n" % (a, b, c, best_accuracy))
    # file.flush()





# import os
# import sys
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models import model_dict
# from dataset.cifar100 import *
# from torchsummary import summary
# from models.resnet import ResNet
# from distiller_zoo import HintLoss
# import torch.optim as optim
# import time
# import argparse
# from models.util import ConvReg
# import matplotlib.pyplot as plt
# import torch.backends.cudnn as cudnn
#
# train_batch_size = 128
# test_batch_size = 100
# n_cls = 100
# num_workers = 1
# total_epoch = 240
# learning_rate = 0.05
# lr_decay_epoch = [150, 180, 210]
# lr_decay_rate = 0.1
# weight_decay = 5e-4
# momentum = 0.9
# print_freq = 100
# dataset = 'cifar100'
#
# parser = argparse.ArgumentParser('argument for training')
# parser.add_argument('--model_t', type=str, default='resnet8', choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
#                                  'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
#                                  'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
#                                  'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
# parser.add_argument('--path_t', type=str, default=None)
# parser.add_argument('--model_s', type=str, default='resnet8', choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
#                                  'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
#                                  'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
#                                  'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
# parser.add_argument('--cuda_visible_devices', type=int, default=0)
# parser.add_argument('--trial', type=int, default=0)
# parser.add_argument('--balance', type=float, default=.5)
# opt = parser.parse_args()
#
#
#
# selection_dict = {
#     'wrn401': ['wrn_40_1', './save/models/wrn_40_1_cifar100_lr_0.05_decay_0.0005_trial_0/wrn_40_1_best.pth',
#             'wrn_16_1', 1, 1],
#     'resnet324': ['resnet32x4', './save/models/resnet32x4_cifar100_lr_0.05_decay_0.0005_trial_0/resnet32x4_best.pth',
#             'resnet8x4', 2, 1],
#     'vgg19': ['vgg19', ' ./save/models/vgg19_cifar100_lr_0.05_decay_0.0005_trial_0/vgg19_best.pth',
#             'vgg11', 2, 1],
#     'resnet110_20': ['resnet110', ' ./save/models/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth',
#             'resnet20', 2, 1],
#     'resnet110_32:':['resnet110', ' ./save/models/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth',
#             'resnet32', 3, 1]
# }
#
# is_pycharm = False
# if is_pycharm:
#     options = list(selection_dict.keys())
#     arguments = selection_dict[options[1]]
#     opt.model_t = arguments[0]
#     opt.path_t = arguments[1]
#     opt.model_s = arguments[2]
#     opt.cuda_visible_devices = arguments[3]
#     opt.trial = arguments[4]
#
# print(opt)
# os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_visible_devices)
#
# balance = opt.balance
# model_t = model_dict[opt.model_t](num_classes=100)
# model_s = model_dict[opt.model_s](num_classes=100)
#
# path_t = opt.path_t
# trial = 0
# r = 1
# a = 0
# b = 1
# p = 1
# d = 1
# kd_T = 4
#
#
# model_t.load_state_dict(torch.load(path_t)['model'])
# model_t.eval()
#
# rand_value = torch.rand((2, 3, 32, 32))
#
# feature, logit = model_s(rand_value, is_feat=True, preact=True)
#
# train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=train_batch_size,
#                                                                         num_workers=num_workers,
#                                                                         is_instance=True)
# if torch.cuda.is_available():
#     model_s.cuda()
#     model_t.cuda()
#
#
# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
#
# def adjust_learning_rate(epoch, learning_rate, lr_decay_epochs,  optimizer):
#     """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
#     steps = np.sum(epoch > np.asarray(lr_decay_epochs))
#     if steps > 0:
#         new_lr = learning_rate * (lr_decay_rate ** steps)
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = new_lr
#
# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res
#
#
# # class SimpleMLP(nn.Module):
# #     def __init__(self, input_size, num_classes=100):
# #         super(SimpleMLP, self).__init__()
# #
# #         self.simple = nn.Sequential(
# #             nn.Linear(input_size, 512),
# #             nn.BatchNorm1d(512),
# #             nn.ReLU6(),
# #             nn.Linear(512, 256),
# #             nn.BatchNorm1d(256),
# #             nn.ReLU6(),
# #             nn.Linear(256, 128),
# #             nn.BatchNorm1d(128),
# #             nn.ReLU6(),
# #             nn.Linear(128, 100)
# #         )
# #
# #     def forward(self, input):
# #         return self.simple(input)
#
#
# #        return self.simple(input)
# class SimpleMLP(nn.Module):
#     def __init__(self, input_size, num_classes=100):
#         super(SimpleMLP, self).__init__()
#
#         self.simple = nn.Sequential(
#             nn.Linear(input_size, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU6(),
#             nn.Linear(128, num_classes)
#         )
#
#     def forward(self, input):
#         return self.simple(input)
#
# # Knowledge distillation Idea
# class AlternativeL2Loss(nn.Module):
#     def __init__(self):
#         super(AlternativeL2Loss, self).__init__()
#         pass
#
#     def forward(self, source, target, margin=2):
#         loss = ((source + margin) ** 2 * ((source > -margin) & (target <= 0)).float() +
#                 (source - margin) ** 2 * ((source <= margin) & (target > 0)).float())
#         return torch.abs(loss).mean()
#
# rand_ = torch.rand((2, 3, 32, 32), dtype=torch.float32).cuda()
# out_feat, out_x = model_t(rand_, is_feat=True, preact=True)
# with torch.no_grad():
#     feat_s, _ = model_s(rand_, is_feat=True, preact=True)
#
#
# MLP = SimpleMLP(out_feat[-1].size(1), 100).cuda()
# #MLP2 = SimpleMLP(out_feat[-1].size(1), 100).cuda()
# module_list = nn.ModuleList([])
# module_list.append(model_s)
# module_list.append(MLP)
#
# trainable_models = nn.ModuleList([])
# trainable_models.append(model_s)
# #trainable_models.append(MLP2)
# trainable_models.append(MLP)
#
# # for mixture fitnet
# criterion_hint = HintLoss()
# regress_s = ConvReg(feat_s[2].shape, out_feat[2].shape).cuda()
# module_list.append(regress_s)
# trainable_models.append(regress_s)
#
#
# optimizer = optim.SGD(trainable_models.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
# criterion_CE = nn.CrossEntropyLoss()
# criterion_CE_1 = nn.CrossEntropyLoss()
# criterion_FM = AlternativeL2Loss()
# criterion_AB = AlternativeL2Loss()
# criterion_LS = nn.BCEWithLogitsLoss()
#
#
#
# cudnn.benchmark = True
#
# best_accuracy = -1
# best_mlp_accuracy = -1
# cos = nn.CosineSimilarity(dim=1, eps=1e-6)
# if is_pycharm:
#     loss_ce_list = []
#     loss_kd_list = []
#     loss_simple_list = []
#
# for epoch in range(1, total_epoch+1):
#     adjust_learning_rate(epoch, learning_rate, lr_decay_epoch, optimizer)
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     loss_ce_ = AverageMeter()
#     loss_kd_ = AverageMeter()
#     loss_simple_ = AverageMeter()
#
#
#     MLP.train()
#  #   MLP2.train()
#     model_s.train()
#     regress_s.train()
#     end = time.time()
#
#     avg_loss_ce = 0
#     avg_loss_simple = 0
#     count = 0
#
#     similarity_list = []
#     for idx, data in enumerate(train_loader):
#         input, target, index = data
#         input = input.float()
#         # target = target.float()
#         optimizer.zero_grad()
#
#         data_time.update(time.time() - end)
#         if torch.cuda.is_available():
#             input = input.cuda()
#             target = target.cuda()
#
#         with torch.no_grad():
#             feat_t, logit_t = model_t(input, is_feat=True, preact=True)
#         feat_s, logit_s = model_s(input, is_feat=True, preact=True)
#
#
#         out_fs = regress_s(feat_s[2])
#         out_ft = feat_t[2]
#
#         loss_hint = criterion_hint(out_fs, out_ft)
#
#         simple_out_t = MLP(feat_t[-1].detach())
#         simple_out_s = MLP(feat_s[-1])
#
#         similarity_list.append \
#             ([cos(simple_out_t, logit_t).mean().item(),
#               cos(simple_out_s, simple_out_t).mean().item(),
#               cos(simple_out_s, logit_s).mean().item(),
#               cos(logit_s, logit_t).mean().item()])
#
#         #
#         # if epoch == 10:
#         #     print(simple_out_t)
#         #     print(simple_out_s)
#         #     exit()
#
#         loss_simple = (balance * F.kl_div(F.log_softmax(simple_out_t / kd_T,dim=1), F.softmax(logit_t.detach() / kd_T, dim=1),size_average=False) +
#                        (1 - balance) * F.kl_div(F.log_softmax(simple_out_s / kd_T, dim=1),
#                                                 F.softmax(simple_out_t.detach() / kd_T, dim=1), size_average=False)) * (
#                                   kd_T ** 2) / logit_s.size(0)
#
#         # loss_simple = (balance * criterion_AB(simple_out_t, logit_t.detach()) +
#         #
#         #              (1 - balance) * F.kl_div(F.log_softmax(simple_out_s / kd_T, dim=1), F.softmax(simple_out_t.detach() / kd_T, dim=1), size_average=False)) * (kd_T ** 2) / logit_s.size(0) \
#         #             + (1 - balance) * F.kl_div(F.log_softmax(logit_s / kd_T, dim=1), F.softmax(simple_out_s / kd_T, dim=1), size_average=False) * (kd_T ** 2) / logit_s.size(0)
#         # loss_simple = balance * criterion_CE2(simple_out_t / kd_T, logit_t.detach() / kd_T) + (1 - balance) * criterion_CE2(simple_out_s / kd_T, simple_out_t.detach())
#         # loss_simple_ce = criterion_CE_1(simple_out_t, target) + criterion_CE_1(simple_out_s, target) # idea 5
#         loss_simple_ce = criterion_CE_1(simple_out_t, target) # idea 6
#         # loss_simple.backward(retain_graph=True)
#         loss_origin = (kd_T ** 2) * F.kl_div(F.log_softmax(logit_s / kd_T, dim=1), F.softmax(logit_t.detach() / kd_T, dim=1), size_average=False) / logit_s.size(0)
#
#         loss_ce = criterion_CE(logit_s, target)
#         loss_fm = criterion_FM(logit_s, logit_t.detach())
#
#         acc_1, acc_5 = accuracy(logit_s, target, topk=(1, 5))
#         top1.update(acc_1[0], input.size(0))
#         top5.update(acc_5[0], input.size(0))
#         loss = r * loss_ce + a * loss_origin + b * loss_simple + p * loss_fm + 100 * loss_hint # 1 0 1
#
#         count += 1
#         #loss = r * loss_ce + a * loss_origin + b * loss_simple + p * loss_fm + d
#         losses.update(loss.item() )
#         loss_ce_.update(loss_ce.item())
#         loss_kd_.update(loss_origin.item())
#         loss_simple_.update(loss_simple.item())
#
#         loss.backward()
#         optimizer.step()
#         batch_time.update(time.time() - end)
#         end = time.time()
#         if idx % print_freq == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                   'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                 epoch, idx, len(train_loader), batch_time=batch_time,
#                 data_time=data_time, loss=losses, top1=top1, top5=top5))
#             sys.stdout.flush()
#     if is_pycharm:
#         loss_ce_list.append(loss_ce_.avg)
#         loss_kd_list.append(loss_kd_.avg)
#         loss_simple_list.append(loss_simple_.avg)
#     print('[Train]* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
#           .format(top1=top1, top5=top5))
#     print('cos(simple_out_t, logit_s) | cos(simple_out_t, simple_out_s) | cos(simple_out_s, logit_s) | cos(logit_s, logit_t)')
#     print(np.array(similarity_list).mean(axis=0))
#     if False:
#
#     #if is_pycharm and (epoch % 5 == 0 and epoch > 30) or epoch == 3:
#         print(loss_ce_list)
#         plt.plot(list(range(len(loss_ce_list))), loss_ce_list, label='ce')
#         plt.plot(list(range(len(loss_kd_list))), loss_kd_list, label='kd')
#         plt.plot(list(range(len(loss_simple_list))), loss_simple_list, label='ours')
#         plt.legend()
#         plt.show()
#
#
#     model_s.eval()
#     MLP.eval()
#     # MLP2.eval()
#     similarity_t_list = []
#     with torch.no_grad():
#         val_batch_time = AverageMeter()
#         val_top1 = AverageMeter()
#         val_top5 = AverageMeter()
#         val_losses = AverageMeter()
#         mlp_top1 = AverageMeter()
#         mlp_top5 = AverageMeter()
#         for idx, (input, target) in enumerate(val_loader):
#             input = input.float().cuda()
#             target = target.cuda()
#
#             val_feat_s, output = model_s(input, is_feat=True, preact=True)
#             loss = criterion_CE(F.softmax(output), target)
#             simple_val_out_s = MLP(val_feat_s[-1])
#
#           #  print(cos(simple_val_out_s, output))
#             similarity_t_list.append(cos(simple_val_out_s, output).mean().item())
#             val_acc_1, val_acc_5 = accuracy(output, target, topk=(1, 5))
#             mlp_val_acc_1, mlp_val_acc_5 = accuracy(simple_val_out_s, target, topk = (1, 5))
#             val_top1.update(val_acc_1[0], input.size(0))
#             val_top5.update(val_acc_5[0], input.size(0))
#             mlp_top1.update(mlp_val_acc_1[0], input.size(0))
#             mlp_top5.update(mlp_val_acc_5[0], input.size(0))
#
#             if idx % print_freq == 0:
#                 print('Test: [{0}/{1}]\t'
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                       'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
#                       'AccMLP@1 {mlp_top1.val:.3f} ({mlp_top1.avg:.3f})\t'
#                       'AccMLP@5 {mlp_top5.val:.3f} ({mlp_top5.avg:.3f})'.format(
#                        idx, len(val_loader), batch_time=batch_time, loss=losses,
#                        top1=val_top1, top5=val_top5, mlp_top1=mlp_top1, mlp_top5=mlp_top5))
#     if best_accuracy < val_top1.avg:
#         best_accuracy = val_top1.avg
#     if best_mlp_accuracy < mlp_top1.avg:
#         best_mlp_accuracy = mlp_top1.avg
#     print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} mlpAcc@1 {mlp_top1.avg:.3f} mlpAcc@5 {mlp_top5.avg:.3f} Best_accuracy %.4f Best_mlp_accuracy %.4f'
#           .format(top1=val_top1, top5=val_top5, mlp_top1=mlp_top1, mlp_top5=mlp_top5) % (best_accuracy, best_mlp_accuracy))
#     print("cos(simple_out_s, logit_s)", sum(similarity_t_list) / len(similarity_t_list))
#
#
#
#
