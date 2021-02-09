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
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_visible_devices)


model_t = model_dict[opt.model_t](num_classes=100)
model_s = model_dict[opt.model_s](num_classes=100)

path_t = opt.path_t
trial = 0
r = 0.1
a = 0.9
kd_T = 4


model_t.load_state_dict(torch.load(path_t)['model'])
model_t.eval()
teacher_classifier = model_t.fc
teacher_classifier.eval()

rand_value = torch.rand((2, 3, 32, 32))

feature, logit = model_s(rand_value, is_feat=True, preact=True)

train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=train_batch_size,
                                                                        num_workers=num_workers,
                                                                        is_instance=True)
if torch.cuda.is_available():
    model_s.cuda()
    model_t.cuda()
    teacher_classifier.cuda()


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
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, input):
        return self.simple(input)

# Knowledge distillation Idea3

MLP = SimpleMLP(256, 100).cuda()

trainable_models = nn.ModuleList([])
trainable_models.append(MLP)
trainable_models.append(model_s)
optimizer = optim.SGD(trainable_models.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
criterion_CE = nn.CrossEntropyLoss()
criterion_KL = nn.KLDivLoss()

balance = 0.5

for epoch in range(1, total_epoch+1):
    adjust_learning_rate(epoch, learning_rate, lr_decay_epoch, optimizer)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
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

        simple_out_t = MLP(feat_t[-1].detach())
        simple_out_s = MLP(feat_s[-1])

        #
        # if epoch == 10:
        #     print(simple_out_t)
        #     print(simple_out_s)
        #     exit()

        loss_simple = (balance * F.kl_div(F.log_softmax(simple_out_t / kd_T, dim=1), F.softmax(logit_t.detach() / kd_T, dim=1), size_average=False) +
                      (1 - balance) * F.kl_div(F.log_softmax(logit_s / kd_T, dim=1), F.softmax(simple_out_s / kd_T, dim=1), size_average=False)) * (kd_T ** 2) / logit_s.size(0)

        # loss_simple = (kd_T ** 2) * F.kl_div(F.log_softmax(logit_s / kd_T, dim=1), F.softmax(logit_t.detach() / kd_T, dim=1), size_average=False) / logit_s.size(0)

        loss_ce = criterion_CE(F.softmax(logit_s), target)

        avg_loss_ce += loss_ce.item()
        avg_loss_simple += loss_simple.item()

        acc_1, acc_5 = accuracy(logit_s, target, topk=(1, 5))
        top1.update(acc_1[0], input.size(0))
        top5.update(acc_5[0], input.size(0))
        count += 1
        loss = r * loss_ce + a * loss_simple
        losses.update(loss.item())
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

    print('[Train]* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

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
                       top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=val_top1, top5=val_top5))









