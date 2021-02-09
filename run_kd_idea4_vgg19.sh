nohup python idea4_copy.py --model_t vgg19 --path_t ./save/models/vgg19_cifar100_lr_0.05_decay_0.0005_trial_0/vgg19_best.pth --model_s vgg11 --balance 0.1 --cuda_visible_devices 3 --trial 2 > idea4_vgg19_vgg11_b_1.out &

nohup python idea4_copy.py --model_t vgg19 --path_t ./save/models/vgg19_cifar100_lr_0.05_decay_0.0005_trial_0/vgg19_best.pth --model_s vgg11 --balance 0.3 --cuda_visible_devices 3 --trial 2 > idea4_vgg19_vgg11_b_3.out &

nohup python idea4_copy.py --model_t vgg19 --path_t ./save/models/vgg19_cifar100_lr_0.05_decay_0.0005_trial_0/vgg19_best.pth --model_s vgg11 --balance 0.5 --cuda_visible_devices 3 --trial 2 > idea4_vgg19_vgg11_b_5.out &

nohup python idea4_copy.py --model_t vgg19 --path_t ./save/models/vgg19_cifar100_lr_0.05_decay_0.0005_trial_0/vgg19_best.pth --model_s vgg11 --balance 0.7 --cuda_visible_devices 3 --trial 2 > idea4_vgg19_vgg11_b_7.out &

nohup python idea4_copy.py --model_t vgg19 --path_t ./save/models/vgg19_cifar100_lr_0.05_decay_0.0005_trial_0/vgg19_best.pth --model_s vgg11 --balance 0.9 --cuda_visible_devices 3 --trial 2 > idea4_vgg19_vgg11_b_9.out &


