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
# Download the pre-trained weights
cd dataset
bash download_datasets.sh
cd ..
# https://www.kaggle.com/c/facial-keypoints-detection/overview
```



## Download pre-trained model weights
The pretrained weights can be downloaded by running the file in dataset or [here](https://skku0-my.sharepoint.com/:f:/g/personal/byo7000_skku_edu/EoP8mWpbyDhNtIaZ9rBoPWcB5QRsinPBKwr0V18dHsUR8w?e=7oNCXY).

```
# Download the pre-trained weights
cd weights
bash download_weights.sh
cd ..
```

## Setup
```
pip install -r requirements.txt
```


## Evaluation (editing)
```
# Dataset and model weights need to be downloaded.
# source and target dataset dir. i.e., StarGAN --> StyleGAN2
# pretrained weight. i.e., efficientnet/stargan.pth.tar
# t-gd pretrained weight. i.e., t-gd/efficientnet/star_to_style2.pth.tar
python eval.py --source_dataset dataset/StarGAN_128 \
                --target_dataset dataset/StyleGAN2_256 \
                --pretrained_dir weights/pre-train/efficientnet/stargan.pth.tar \
                --resume weights/t-gd/efficientnet/star_to_style2.pth.tar
```
