# normal kd
 # WRN 40-1 -> WRN 16-1
nohup python train_normal_kd.py --path_t ./save/models/wrn_40_1_cifar100_lr_0.05_decay_0.0005_trial_0/wrn_40_1_best.pth --cuda_visible 0 --distill attention --model_s wrn_16_1 -a 1 -b 1000 --trial 2 > at_wrn401_wrn161_t2.out &
 # RESNET32X4 -> RESNET8X4

nohup python train_normal_kd.py --path_t ./save/models/resnet32x4_cifar100_lr_0.05_decay_0.0005_trial_0/resnet32x4_best.pth --distill attention --cuda_visible 1 --model_s resnet8x4 -a 1 -b 1000 --trial 2 > at_resnet324_resnet84_t2.out &


 # VGG19 -> VGG11

nohup python train_normal_kd.py --path_t ./save/models/vgg19_cifar100_lr_0.05_decay_0.0005_trial_0/vgg19_best.pth --distill attention --cuda_visible 2 --model_s vgg11 -a 1 -b 1000 --trial 2 > at_vgg19_vgg11_t2.out &

 # RESNET110 -> RESNET20
nohup python train_normal_kd.py --path_t ./save/models/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth --distill attention --cuda_visible 2 --model_s resnet20 -a 1 -b 1000 --trial 2 > at_resnet110_resnet20_t2.out &

 # RESNET110 -> RESNET32

nohup python train_normal_kd.py --path_t ./save/models/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth --distill attention --cuda_visible 3 --model_s resnet32 -a 1 -b 1000 --trial 2 > at_resnet110_resnet32_t2.out &
