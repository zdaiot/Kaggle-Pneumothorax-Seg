import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
from PIL import Image

from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, CLAHE, RandomRotate90, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightnessContrast, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,Cutout, Rotate,
)


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

        plt.show()

def data_augmentation(original_image, original_mask):
    """进行样本和掩膜的随机增强
    
    Args:
        original_image: 原始图片
        original_mask: 原始掩膜
    Return:
        image_aug: 增强后的图片
        mask_aug: 增强后的掩膜
    """

    augmentations = Compose([
        # 翻转
        OneOf([
                HorizontalFlip(p=0.5), 
                Rotate(limit=30, p=0.3),   
            ], p=0.75),
        
        # 直方图均衡化
        CLAHE(p=1.0),

        RandomGamma(gamma_limit=(80, 120), p=0.5),
        RandomBrightnessContrast(p=0.5),
        
        OneOf([
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.3),
        
        OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2)
    ])
    
    augmented = augmentations(image=original_image, mask=original_mask)
    image_aug = augmented['image']
    mask_aug = augmented['mask']

    return image_aug, mask_aug


if __name__ == "__main__":
    image_path = '/home/apple/program/MXQ/Competition/Kaggle/Pneumothorax-Seg/Kaggle-Pneumothorax-Seg/datasets/SIIM_data/sample_images/1.2.276.0.7230010.3.1.4.8323329.1314.1517875167.222290.jpg'
    mask_path = '/home/apple/program/MXQ/Competition/Kaggle/Pneumothorax-Seg/Kaggle-Pneumothorax-Seg/datasets/SIIM_data/sample_mask/1.2.276.0.7230010.3.1.4.8323329.1314.1517875167.222290.jpg'

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)

    image_aug, mask_aug = data_augmentation(image, mask)

    visualize(image_aug, mask_aug, original_image=image, original_mask=mask)

