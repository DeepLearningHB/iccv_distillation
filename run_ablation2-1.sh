nohup python ablation_b_2.py --model_t resnet32x4 --path_t ./save/models/resnet32x4_best.pth --model_s resnet8x4 --cuda_visible_devices 1 --max_dim 310 > abl_b2_resnet32x4_8x4.out &
nohup python ablation_b_2.py --model_t vgg19 --path_t ./save/models/vgg19_best.pth --model_s vgg8 --cuda_visible_devices 1 --max_dim 480 > abl_b2_vgg19_8.out &


nohup python ablation_c_2.py --model_t resnet32x4 --path_t ./save/models/resnet32x4_best.pth --model_s resnet8x4 --cuda_visible_devices 3 --max_dim 310 > abl_c2_resnet32x4_8x4.out &
nohup python ablation_c_2.py --model_t vgg19 --path_t ./save/models/vgg19_best.pth --model_s vgg8 --cuda_visible_devices 3 --max_dim 480 > abl_c2_vgg19_8.out &


nohup python ablation_e_2.py --model_t resnet32x4 --path_t ./save/models/resnet32x4_best.pth --model_s resnet8x4 --cuda_visible_devices 3 --max_dim 310 > abl_e2_resnet32x4_8x4.out &
nohup python ablation_e_2.py --model_t vgg19 --path_t ./save/models/vgg19_best.pth --model_s vgg8 --cuda_visible_devices 3 --max_dim 480 > abl_e2_vgg19_8.out &


nohup python ablation_f_2.py --model_t resnet32x4 --path_t ./save/models/resnet32x4_best.pth --model_s resnet8x4 --cuda_visible_devices 6 --max_dim 310 > abl_f2_resnet32x4_8x4.out &
nohup python ablation_f_2.py --model_t vgg19 --path_t ./save/models/vgg19_best.pth --model_s vgg8 --cuda_visible_devices 6 --max_dim 480 > abl_f2_vgg19_8.out &


nohup python ablation_g_2.py --model_t resnet32x4 --path_t ./save/models/resnet32x4_best.pth --model_s resnet8x4 --cuda_visible_devices 7 --max_dim 310 > abl_g2_resnet32x4_8x4.out &
nohup python ablation_g_2.py --model_t vgg19 --path_t ./save/models/vgg19_best.pth --model_s vgg8 --cuda_visible_devices 8 --max_dim 480 > abl_g2_vgg19_8.out &
