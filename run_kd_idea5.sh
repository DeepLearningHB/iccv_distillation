nohup python idea5.py --model_t wrn_40_1 --path_t ./save/models/wrn_40_1_cifar100_lr_0.05_decay_0.0005_trial_0/wrn_40_1_best.pth --model_s wrn_16_1 --cuda_visible_devices 1 --trial 2 > idea5_wrn401_wrn161_t2.out &

nohup python idea5.py --model_t resnet32x4 --path_t ./save/models/resnet32x4_cifar100_lr_0.05_decay_0.0005_trial_0/resnet32x4_best.pth --model_s resnet8x4 --cuda_visible_devices 2 --trial 2 > idea5_resnet324_resnet84_t2.out &

nohup python idea5.py --model_t vgg19 --path_t ./save/models/vgg19_cifar100_lr_0.05_decay_0.0005_trial_0/vgg19_best.pth --model_s vgg11 --cuda_visible_devices 3 --trial 2 > idea5_vgg19_vgg11_t2.out &

nohup python idea5.py --model_t resnet110 --path_t ./save/models/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth --model_s resnet20 --cuda_visible_devices 3 --trial 2 > idea5_resnet110_resnet20_t2.out &

nohup python idea5.py --model_t resnet110 --path_t ./save/models/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth --model_s resnet32 --cuda_visible_devices 3 --trial 2 > idea5_resnet110_resnet32_t2.out &
