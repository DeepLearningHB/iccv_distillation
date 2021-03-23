# Marrying the Best of Both Knowledge: Ensemble-based Latent Matching with Softmax Representation Knowledge Distillation.

## Overview of our framework.
<img src='./images/overview.png' width=1000>


## Dataset
The datasets for our method can be downloaded in Torchvision, and shell file, which is created by us. 
CIFAR-10
CIFAR-100
Tiny ImageNet
Facial Key-points

```
# CIFAR-100 will be downloaded automatically when you run demo.py 
cd ..
# https://www.kaggle.com/c/facial-keypoints-detection/overview
```



## Download pre-trained model weights
The pretrained weights can be downloaded by running the file in dataset or [here](https://skku0-my.sharepoint.com/:f:/g/personal/byo7000_skku_edu/EoP8mWpbyDhNtIaZ9rBoPWcB5QRsinPBKwr0V18dHsUR8w?e=7oNCXY).

```
# Download the pre-trained baseline weights
cd weights
bash download_baseline.sh
cd ..

# Download the pre-trained LSKD weights
cd weights
bash download_lskd_weight.sh
cd ..
```

## Setup
```
pip install -r requirements.txt
```


## Evaluation (editing)
```
# Dataset and model weights need to be downloaded.

python demo.py  --model_s vgg11 \
                --model_path weights/vgg19_vgg11_LSKD_demo.py \
                --k 4 \
                --cuda_visible_devices=0

# referenced by CRD
```
