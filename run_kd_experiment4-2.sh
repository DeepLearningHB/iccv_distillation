nohup python experiment_4_2.py --model_t resnet32x4 --path_t ./save/models/resnet32x4_best.pth --model_s resnet8x4 --cuda_visible_devices 0 --max_dim 310 > exp4_2_resnet32x4_8x4.out &

nohup python experiment_4_2.py --model_t resnet110 --path_t ./save/models/resnet110_best.pth --model_s resnet32 --cuda_visible_devices 1 --max_dim 400 > exp4_2_res110_32.out &
nohup python experiment_4_2.py --model_t resnet110 --path_t ./save/models/resnet110_best.pth --model_s resnet20 --cuda_visible_devices 4 --max_dim 230 > exp4_2_res110_20.out &
nohup python experiment_4_2.py --model_t resnet56 --path_t ./save/models/resnet56_best.pth --model_s resnet20 --cuda_visible_devices 5 --max_dim 230 > exp4_2_resnet56_20.out &

nohup python experiment_4_2.py --model_t wrn_40_2 --path_t ./save/models/wrn_40_2_best.pth --model_s wrn_16_2 --cuda_visible_devices 0 --max_dim 344 > exp4_2_wrn40_16.out &
nohup python experiment_4_2.py --model_t wrn_40_4 --path_t ./save/models/wrn_40_4_best.pth --model_s wrn_16_4 --cuda_visible_devices 4 --max_dim 800 > exp4_2_wrn40_16_4.out &
nohup python experiment_4_2.py --model_t wrn_16_10 --path_t ./save/models/wrn_16_10_best.pth --model_s wrn_10_10 --cuda_visible_devices 3 --max_dim 964 > exp4_2_1_wrn16_10_10.out &

nohup python experiment_4_2.py --model_t vgg19 --path_t ./save/models/vgg19_best.pth --model_s vgg11 --cuda_visible_devices 3 --max_dim 1024 > exp4_2_1_vgg19_11.out &
nohup python experiment_4_2.py --model_t vgg19 --path_t ./save/models/vgg19_best.pth --model_s vgg8 --cuda_visible_devices 4 --max_dim 480 > exp4_2_1_vgg19_8.out &
nohup python experiment_4_2.py --model_t vgg13 --path_t ./save/models/vgg13_best.pth --model_s vgg8 --cuda_visible_devices 5 --max_dim 480 > exp4_2_1_vgg13_8.out &


### repeat
nohup python experiment_4_2.py --model_t resnet110 --path_t ./save/models/resnet110_best.pth --model_s resnet20 --cuda_visible_devices 0 --max_dim 230 > exp4_2_1_res110_20.out &
nohup python experiment_4_2.py --model_t wrn_40_2 --path_t ./save/models/wrn_40_2_best.pth --model_s wrn_16_2 --cuda_visible_devices 1 --max_dim 344 > exp4_2_1_wrn40_16.out &
