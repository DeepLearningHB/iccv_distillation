nohup python experiment_1_10.py --model_t wrn_40_1 --path_t ./save/models/wrn_40_1_best.pth --model_s wrn_16_1 --cuda_visible_devices 0 --trial 0 > exp1_10_wrn.out &

nohup python experiment_1_10.py --model_t vgg19 --path_t ./save/models/vgg19_best.pth --model_s vgg11 --cuda_visible_devices 1 --trial 0 > exp1_10_vgg.out &

nohup python experiment_1_10.py --model_t resnet110 --path_t ./save/models/resnet110_best.pth --model_s resnet32 --cuda_visible_devices 2 --trial 0 > exp1_10_res32.out &

nohup python experiment_1_10.py --model_t resnet110 --path_t ./save/models/resnet110_best.pth --model_s resnet20 --cuda_visible_devices 3 --trial 0 > exp1_10_res20.out &
