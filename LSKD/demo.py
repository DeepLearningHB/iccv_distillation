import torch
import torch.nn as nn
import argparse
from utils import environments as envs
from models import model_dict
import modules.LMD as LMD
import modules.SRD as SRD
from dataset.cifar100 import *
from utils.utils import AverageMeter, accuracy


def argument_parser():
    parser = argparse.ArgumentParser (
        "Argument for inferring Latant matching with Softmax representation Knowledge Distillation")
    parser.add_argument('--model_s', type=str, default='vgg11')
    parser.add_argument('--model_path', type=str, default='./weights/vgg19_vgg11_LSKD_best.pth')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--cuda_visible_devices', type=int, default=0)
    return parser.parse_args()


def main():
    opt = argument_parser()
    if not torch.cuda.is_available():
        print ("Set cuda")
        exit()
    os.environ['CUDA_VISIBLE_DEVICES'] = str (opt.cuda_visible_devices)
    assert os.path.exists(opt.model_path), 'File Not Found Exception.'

    final_dim = envs.final_dim[opt.model_s]
    model_s = model_dict[opt.model_s](num_classes=envs.n_cls).cuda()
    lmd = LMD.LMD(model_s, final_dim, envs.dataset).cuda()
    srd = SRD.SRD(model_s, envs.dataset).cuda()
    trained_models = nn.ModuleList ([])

    trained_models.MLP = srd
    trained_models.model_s = model_s
    trained_models.conv_distiller = lmd
    trained_models.load_state_dict(torch.load(opt.model_path))
    trained_models.cuda ()
    trained_models.eval()

    print("Loading success")



    _, test_loader, _ = get_cifar100_dataloaders (batch_size=envs.batch_size, num_workers=envs.num_workers,
                                                  is_instance=True)

    with torch.no_grad():
        test_top1_1_k = AverageMeter()
        test_top1_all = AverageMeter()

        test_top1_student = AverageMeter()
        test_top1_auxiliary = [AverageMeter() for _ in range(len(trained_models.conv_distiller))]

        for idx, (input_, target_) in enumerate(test_loader):
            input_ = input_.float().cuda()
            target_ = target_.cuda()
            out_feat, out_logit = trained_models.model_s (input_, is_feat=True, preact=False)
            ensemble_1_k = torch.zeros(size=[input_.size(0), envs.n_cls]).cuda()
            ensemble_all = out_logit

            for idx_, (lmd_, s_feat) in enumerate(zip(trained_models.conv_distiller, out_feat)):
                lmd_logit_s, _ = lmd_(s_feat)
                if idx_ < opt.k:
                    ensemble_1_k += lmd_logit_s.clone()
                ensemble_all += lmd_logit_s.clone()
                lmd_aux_acc1, _ = accuracy(lmd_logit_s.clone(), target_, topk=(1, 5))
                test_top1_auxiliary[idx_].update(lmd_aux_acc1[0], input_.size(0))

            ensemble_all += trained_models.MLP(out_feat[-1])

            ensemble_1_k /= opt.k
            ensemble_all /= (len(trained_models.conv_distiller) + 2) # all_LMDs + student_mlp + srd
            lmd_1_k_acc1, _ = accuracy(ensemble_1_k.clone(), target_, topk=(1, 5))
            lmd_all_acc1, _ = accuracy(ensemble_all.clone(), target_, topk=(1, 5))
            acc1 = accuracy(out_logit.clone(), target_, topk=(1, 5))

            test_top1_student.update(acc1[0], input_.size(0))
            test_top1_1_k.update(lmd_1_k_acc1[0], input_.size(0))
            test_top1_all.update(lmd_all_acc1[0], input_.size(0))

        print ("[Test] Auxiliary Classifier Accuracies: ", end=' ')
        for i in range (len (trained_models.conv_distiller)):
            print ("Aux@%d: %.4f" % (i, test_top1_auxiliary[i].avg), end=' ')
        print ()
        print ("* [Test] \n [Ensemble All Accuracy: %.4f]\n [** Ensemble 1_k Accuracy: %.4f]" % (
        test_top1_all.avg, test_top1_1_k.avg))
        print (test_top1_student.avg)


if __name__ == "__main__":
    main()
