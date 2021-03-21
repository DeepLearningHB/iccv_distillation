import torch
import torch.nn as nn
import argparse
from utils import environments as envs
from models import model_dict
from dataset.cifar100 import *
from utils.utils import AverageMeter, accuracy

def argument_parser():
    parser = argparse.ArgumentParser (
        "Argument for model inference")
    parser.add_argument('--model', type=str, default='resnet20')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--cuda_visible_devices', type=int, default=0)
    return parser.parse_args()

def main():
    opt = argument_parser ()
    if not torch.cuda.is_available ():
        print ("Set cuda")
        exit ()
    os.environ['CUDA_VISIBLE_DEVICES'] = str (opt.cuda_visible_devices)

    assert os.path.exists (opt.model_path), 'File Not Found Exception.'
    model = None
    try:
        model = model_dict[opt.model_s] (num_classes=envs.n_cls)
        model.load_state_dict (torch.load (opt.model_path)['model_s'])
        print ("Loading success")
    except:
        print ("Occurred exception during model loading.")
        exit ()

    model.eval() 
    _, test_loader, _ = get_cifar100_dataloaders (batch_size=train_batch_size, num_workers=num_workers,
                                                  is_instance=True)

    with torch.no_grad():
        test_top1 = AverageMeter()
        for idx, (input_, target_) in enumerate(test_loader):
            input = input.float().cuda()
            target = target.cuda()
            out_feat, out_logit = model (input_, is_feat=True, preact=True)
            acc_1, _ = accuracy (out_logit.clone (), target, topk=(1, 5))
            test_top1.update(acc_1[0], input_.size(0))
            
        print("* [Test: %s] \n [Ensemble All Accuracy: %.4f]\n [** Ensemble 1_k Accuracy: %.4f]" % (opt.model, test_top1.avg, test_top1.avg))


if __name__ == "__main__":
    main()
