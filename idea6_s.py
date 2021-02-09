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
from torch.distributions import Categorical
import matplotlib.pyplot as plt

train_batch_size = 128
test_batch_size = 100
n_cls = 100
num_workers = 2
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
    arguments = selection_dict[options[0]]
    opt.model_t = arguments[0]
    opt.path_t = arguments[1]
    opt.model_s = arguments[2]
    opt.cuda_visible_devices = arguments[3]
    opt.trial = arguments[4]

print(opt)
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_visible_devices)


model_t = model_dict[opt.model_t](num_classes=100)
model_s = model_dict[opt.model_s](num_classes=100)

path_t = opt.path_t
trial = 0
r = 0.1
a = 0.9
b = 0
p = 1
d = 0
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

# Knowledge distillation Idea3

class AlternativeL2Loss(nn.Module):
    def __init__(self):
        super(AlternativeL2Loss, self).__init__()
        pass

    def forward(self, source, target, margin=2):
        loss = ((source + margin) ** 2 * ((source > -margin) & (target <= 0)).float() +
                (source - margin) ** 2 * ((source <= margin) & (target > 0)).float())
        return torch.abs(loss).mean()

rand_ = torch.rand((2, 3, 32, 32), dtype=torch.float32).cuda()
out_feat, out_x = model_t(rand_, is_feat=True, preact=True)
out_feat_s, out_x_s = model_s(rand_, is_feat=True, preact=True)
MLP_T = SimpleMLP(out_feat[-1].size(1), out_x.size(1)).cuda()
MLP_S = SimpleMLP(out_feat_s[-1].size(1), out_x_s.size(1)).cuda()


trainable_models = nn.ModuleList([])
trainable_models.append(MLP_T)
trainable_models.append(MLP_S)
trainable_models.append(model_s)
optimizer = optim.SGD(trainable_models.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
criterion_CE = nn.CrossEntropyLoss()
criterion_CE_1 = nn.CrossEntropyLoss()
criterion_FM = AlternativeL2Loss()

balance = 0.5
best_accuracy = -1

if is_pycharm:
    loss_ce_list = []
    loss_kd_list = []
    loss_simple_list = []
    entropy_teacher = []
    entropy_student = []
    entropy_teacher_t_2 = []
    entropy_teacher_t_4 = []


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


    model_s.train()
    end = time.time()

    avg_loss_ce = 0
    avg_loss_simple = 0
    count = 0


    for idx, data in enumerate(train_loader):
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

        # simple_out_t = MLP_T(feat_t[-1].detach())
        # simple_out_s = MLP_S(feat_s[-1])

        #
        # if epoch == 10:
        #     print(simple_out_t)
        #     print(simple_out_s)
        #     exit()
        # loss_simple = (balance * F.kl_div(F.log_softmax(simple_out_t / kd_T, dim=1), F.softmax(logit_t.detach() / kd_T, dim=1), size_average=False) +
        #              (1 - balance) * F.kl_div(F.log_softmax(simple_out_s / kd_T, dim=1), F.softmax(simple_out_t.detach() / kd_T, dim=1), size_average=False)) * (kd_T ** 2) / logit_s.size(0)
        # loss_simple = balance * criterion_CE2(simple_out_t / kd_T, logit_t.detach() / kd_T) + (1 - balance) * criterion_CE2(simple_out_s / kd_T, simple_out_t.detach())
        # loss_simple_ce = criterion_CE_1(simple_out_t, target) + criterion_CE_1(simple_out_s, target) # idea 5
        # loss_simple_ce = criterion_CE_1(simple_out_t, target) # idea 6
        # loss_simple.backward(retain_graph=True)
        loss_origin = (kd_T ** 2) * F.kl_div(F.log_softmax(logit_s / kd_T, dim=1), F.softmax(logit_t.detach() / kd_T, dim=1), size_average=False) / logit_s.size(0)

        # entropy_teacher.append(Categorical(probs=F.softmax(simple_out_t)).entropy().mean().cpu().detach().numpy().tolist())
        # entropy_teacher_t_2.append(
        #     Categorical(probs=F.softmax(simple_out_t / 2.)).entropy().mean().cpu().detach().numpy().tolist())
        # entropy_teacher_t_4.append(
        #     Categorical(probs=F.softmax(simple_out_t / 4.)).entropy().mean().cpu().detach().numpy().tolist())
        # entropy_student.append(Categorical(probs=F.softmax(simple_out_s)).entropy().mean().cpu().detach().numpy().tolist())
        # if epoch % 7 == 0 and idx == 0:
        #     line = plt.plot(list(range(len(entropy_teacher))), entropy_teacher,label='teacher_t_1',)
        #     plt.setp(line, linewidth=0.3)
        #     line2 = plt.plot(list(range(len(entropy_student))), entropy_student, label='student_t_1')
        #     plt.setp(line2, linewidth=0.3)
        #     line3 = plt.plot(list(range(len(entropy_teacher_t_2))), entropy_teacher_t_2, label='teacher_t_2')
        #     plt.setp(line3, linewidth=0.3)
        #     line4 = plt.plot(list(range(len(entropy_teacher_t_4))), entropy_teacher_t_4, label='teacher_t_4')
        #     plt.setp(line4, linewidth=0.3)
        #     plt.legend()
        #     plt.show()
        loss_ce = criterion_CE(logit_s, target)
        loss_fm = criterion_FM(logit_s, logit_t.detach())



        acc_1, acc_5 = accuracy(logit_s, target, topk=(1, 5))
        top1.update(acc_1[0], input.size(0))
        top5.update(acc_5[0], input.size(0))
        count += 1
        #loss = r * loss_ce + a * loss_origin + b * loss_simple + p * loss_fm + d * loss_simple_ce
        # loss = r * loss_ce + a * loss_origin + b * loss_simple + p * loss_fm + d * loss_simple_ce
        loss = r * loss_ce + a * loss_origin + p * loss_fm
        losses.update(loss)
        loss_ce_.update(loss_ce.item())
        loss_kd_.update(loss_origin.item())
       # loss_simple_.update(loss_simple.item())

        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
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
    if False:
        loss_ce_list.append(loss_ce_.avg)
        loss_kd_list.append(loss_kd_.avg)
        loss_simple_list.append(loss_simple_.avg)
    print('[Train]* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    if False:

    #if is_pycharm and (epoch % 5 == 0 and epoch > 30) or epoch == 3:
        print(loss_ce_list)
        plt.plot(list(range(len(loss_ce_list))), loss_ce_list, label='ce')
        plt.plot(list(range(len(loss_kd_list))), loss_kd_list, label='kd')
        plt.plot(list(range(len(loss_simple_list))), loss_simple_list, label='ours')
        plt.legend()
        plt.show()


    model_s.eval()
    with torch.no_grad():
        val_batch_time = AverageMeter()
        val_top1 = AverageMeter()
        val_top5 = AverageMeter()
        val_losses = AverageMeter()
        for idx, (input, target) in enumerate(val_loader):
            input = input.float().cuda()
            target = target.cuda()

            output = model_s(input)
            loss = criterion_CE(F.softmax(output), target)
            val_acc_1, val_acc_5 = accuracy(output, target, topk=(1, 5))
            val_top1.update(val_acc_1[0], input.size(0))
            val_top5.update(val_acc_5[0], input.size(0))


            if idx % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=val_top1, top5=val_top5))
    if best_accuracy < val_top1.avg:
        best_accuracy = val_top1.avg

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Best_accuracy %.4f'
          .format(top1=val_top1, top5=val_top5) % (best_accuracy))









