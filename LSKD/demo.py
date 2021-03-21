import os
import torch
import torch.nn as nn
import argparse
import environments as envs
from models import model_dict
import modules.LMD as LMD
import modules.SRD as SRD
from dataset.cifar100 import *
from utils.utils import AverageMeter, accuracy
def argument_parser():
    parser = argparse.ArgumentParser (
        "Argument for inferring Latant matching with Softmax representation Knowledge Distillation")
    parser.add_argument('--model_s', type=str, default='resnet20')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--cuda_visible_devices', type=int, default=0)
    return parser.parse_args()

def main():
    opt = argument_parser()
    if not torch.cuda.is_available():
        print ("Set cuda")
        exit()
    os.environ['CUDA_VISIBLE_DEVICES'] = str (opt.cuda_visible_devices)

    assert os.path.exists(opt.model_path), 'File Not Found Exception.'
    model_s, lmd, srd = None, None, None
    try:
        model_s = model_dict[opt.model_s](num_classes=envs.n_cls)
        lmd = LMD.LMD(model_s, envs.final_dim['resnet32'], envs.n_cls)
        srd = SRD.SRD(model_s, envs.n_cls)
        model_s.load_state_dict(torch.load(opt.model_path)['model_s'])
        lmd.load_state_dict(torch.load(opt.model_path)['lmd'])
        srd.load_state_dict(torch.load(opt.model_path)['srd'])
        print("Loading success")
    except:
        print("Occurred exception during model loading.")
        exit()

    trained_models = nn.ModuleList([])
    trained_models.model_s = model_s
    trained_models.LMD = lmd
    trained_models.SRD = srd
    trained_models.eval()

    _, test_loader, _ = get_cifar100_dataloaders (batch_size=train_batch_size, num_workers=num_workers, is_instance=True)

    with torch.no_grad():
        test_top1_1_k = AverageMeter()
        test_top1_all = AverageMeter()
        test_top1_auxiliary = [AverageMeter() for _ in range(len(trained_models.LMD))]

        for idx, (input_, target_) in enumerate(test_loader):
            input = input.float().cuda()
            target = target.cuda()
            out_feat, out_logit = trained_models.model_s (input_, is_feat=True, preact=True)
            ensemble_1_k = torch.zeros(size=[input_.size(0), envs.n_cls]).cuda()
            ensemble_all = out_logit

            for idx_, (lmd_, s_feat) in enumerate(zip(trained_models.LMD, out_feat)):
                lmd_logit_s, _ = lmd_(s_feat)
                if idx_ < opt.k:
                    ensemble_1_k += lmd_logit_s.clone()
                ensemble_all += lmd_logit_s.clone()
                lmd_aux_acc1, _ = accuracy(lmd_logit_s.clone(), target, topk=(1, 5))
                test_top1_auxiliary[idx_].update(lmd_aux_acc1[0], input.size(0))

            ensemble_all += trained_models.SRD(out_feat[-1])

            ensemble_1_k /= opt.k
            ensemble_all /= (len(trained_models) + 2) # all_LMDs + student_mlp + srd
            lmd_1_k_acc1, _ = accuracy(ensemble_1_k.clone(), target, topk=(1, 5))
            lmd_all_acc1, _ = accuracy(ensemble_all.clone(), target, topk=(1, 5))

            test_top1_1_k.update(lmd_1_k_acc1[0], input_.size(0))
            test_top1_all.update(lmd_all_acc1[0], input_.size(0))

        print("[Test] Auxiliary Classifier Accuracies: ", end=' ')
        for i in range(len(trained_models.LMD)):
            print("Aux@%d: %.4f" % (i, test_top1_auxiliary[i].avg), end=' ')
        print()
        print("* [Test] \n [Ensemble All Accuracy: %.4f]\n [** Ensemble 1_k Accuracy: %.4f]" % (test_top1_1_k.avg, test_top1_all.avg))


if __name__ == "__main__":
    main()
