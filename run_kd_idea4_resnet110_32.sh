nohup python idea4_copy.py --model_t resnet110 --path_t ./save/models/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth --model_s resnet32 --balance 0.1 --cuda_visible_devices 3 --trial 2 > idea4_resnet110_resnet32_b_1.out &

nohup python idea4_copy.py --model_t resnet110 --path_t ./save/models/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth --model_s resnet32 --balance 0.3 --cuda_visible_devices 3 --trial 2 > idea4_resnet110_resnet32_b_3.out &

nohup python idea4_copy.py --model_t resnet110 --path_t ./save/models/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth --model_s resnet32 --balance 0.5 --cuda_visible_devices 3 --trial 2 > idea4_resnet110_resnet32_b_5.out &

nohup python idea4_copy.py --model_t resnet110 --path_t ./save/models/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth --model_s resnet32 --balance 0.7 --cuda_visible_devices 3 --trial 2 > idea4_resnet110_resnet32_b_7.out &

nohup python idea4_copy.py --model_t resnet110 --path_t ./save/models/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth --model_s resnet32 --balance 0.9 --cuda_visible_devices 3 --trial 2 > idea4_resnet110_resnet32_b_9.out &


