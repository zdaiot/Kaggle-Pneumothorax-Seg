import numpy as np
import os
import cv2
import random
import glob
import tqdm
import shutil
from matplotlib import pyplot as plt
from PIL import Image

from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, CLAHE, RandomRotate90, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightnessContrast, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,Cutout,Rotate
)

AUG = [HorizontalFlip(p=1.0), Rotate(limit=15, p=1.0)]


def sample_aug(image, mask, aug):
    """对单个图片和掩膜应用aug
    """
    augmented = aug(image=image, mask=mask)
    image_aug, mask_aug = augmented['image'], augmented['mask']

    return image_aug, mask_aug


def aug_save(image_name, mask_name, augs=AUG, original_path, save_path):
    image_path = os.path.join(original_path, 'train_images', image_name)
    mask_path = os.path.join(original_path, 'train_mask', mask_name)
    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path)
    image = np.asarray(image)
    mask = np.asarray(mask)

    # 将原始样本复制一份到目标目录
    image_cp_name = os.path.join(save_path, 'train_images', image_name)
    mask_cp_name = os.path.join(save_path, 'train_mask', mask_name)
    shutil.copyfile(image_path, image_cp_name)
    shutil.copyfile(mask_path, mask_cp_name)

    # 对样本进行增强并保存至目标目录
    for aug_index, aug in enumerate(augs):
        image_aug, mask_aug = sample_aug(image, mask, aug)
        image_aug = Image.fromarray(image_aug)
        mask_aug = Image.fromarray(mask_aug)

        image_aug_name = os.path.join(save_path, 'train_images', \
                                        image_name.replace('.jpg', '_' + str(aug_index) + '.jpg'))
        mask_aug_name = os.path.join(save_path, 'train_mask', \
                                        mask_aug.replace('.png', '_' + str(aug_index) + '.png'))
        
        image_aug.save(image_aug_name)
        mask_aug.save(mask_aug_name)

    pass

def dataset_aug(dataset_root, save_root, augs=AUG):
    images_path = os.path.join(dataset_root, 'train_images')
    masks_path = os.path.join(dataset_root, 'train_mask')

    images_name = os.listdir(images_path)
    tbar = tqdm.tqdm(images_name)

    for index, image_name in enumerate(tbar):
        mask_name = image_name.replace('.jpg', '.png')
        aug_save(image_name, mask_name, augs=AUG, dataset_root, save_root)
    
    pass

if __name__ == "__main__":
    # 创建所需文件夹，也可以直接使用os.makedirs()创建多级目录
    save_root = './SIIM_AUG'
    if not os.path.exists(save_root)
        os.mkdir(save_root)
    
    train_images_path = os.path.join(save_root, 'train_images')
    if not os.path.exists(train_images_path):
        os.mkdir(train_images_path)
    
    train_mask_path = os.path.join(save_root, 'train_mask')
    if not os.path.exists(train_mask_path):
        os.mkdir(train_mask_path)

    dataset_root = './SIIM_data'
    dataset_aug(dataset_root, save_root)
    
