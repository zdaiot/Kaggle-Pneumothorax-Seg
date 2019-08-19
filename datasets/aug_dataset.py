import numpy as np
import os
import sys
import cv2
import random
import glob
import tqdm
import shutil
from matplotlib import pyplot as plt
from PIL import Image
import threading
from multiprocessing import Pool
from functools import partial

from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, CLAHE, RandomRotate90, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightnessContrast, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,Cutout,Rotate, OpticalDistortion, RandomScale, PadIfNeeded,
)

'''
AUG = [
    HorizontalFlip(p=1.0), 
    Rotate(limit=15, p=1.0),
    ElasticTransform(p=1, alpha=500, sigma=500 * 0.05, alpha_affine=500 * 0.03)
]
'''


AUG = [
    # 角度旋转
    Compose([
        Rotate(limit=(5, 15), border_mode=0, value=0, always_apply=True),
        ]),

    Compose([
        Rotate(limit=(-15, -5), border_mode=0, value=0, always_apply=True),
        ]),

    # 多尺度缩放，偏移等
    Compose([
        # 随机偏移、尺度变换、角度翻转(尺度放大)
        ShiftScaleRotate(shift_limit=0.05, scale_limit=(0.2, 0.3), rotate_limit=5, border_mode=0, value=0, always_apply=True),
        PadIfNeeded(min_height=1024, min_width=1024, border_mode=0, value=0, always_apply=True),
        ]),
    
    Compose([
        # 随机偏移、尺度变换、角度翻转(尺度缩小)
        # RandomScale(scale_limit=0.3, p=1.0),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=(-0.3, -0.2), rotate_limit=5, border_mode=0, value=0, always_apply=True),
        PadIfNeeded(min_height=1024, min_width=1024, border_mode=0, value=0, always_apply=True),
        ])
    ]


def sample_aug(image, mask, aug):
    """对单个图片和掩膜应用单个aug
    """
    augmented = aug(image=image, mask=mask)
    image_aug, mask_aug = augmented['image'], augmented['mask']

    return image_aug, mask_aug


def aug_save(image_name, original_path, save_path, augs=AUG):
    """对单个图片和掩膜应用多个增强方法
    """
    mask_name = image_name.replace('jpg', 'png')
    image_path = os.path.join(original_path, 'train_images', image_name)
    mask_path = os.path.join(original_path, 'train_mask', mask_name)
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)
    image = np.asarray(image)
    mask = np.asarray(mask)

    # 将原始样本复制一份到目标目录
    image_cp_name = os.path.join(save_path, 'train_images', image_name)
    mask_cp_name = os.path.join(save_path, 'train_mask', mask_name)
    shutil.copyfile(image_path, image_cp_name)
    shutil.copyfile(mask_path, mask_cp_name)
    
    # 判断是否包含掩膜
    mask_thresh = mask > 0
    mask_pixel_num = np.sum(mask_thresh)
    
    # 只对包含掩膜的样本进行增强
    if mask_pixel_num:
        # 对样本进行增强并保存至目标目录
        for aug_index, aug in enumerate(augs):
            image_aug, mask_aug = sample_aug(image, mask, aug)
            image_aug = Image.fromarray(image_aug)
            mask_aug = Image.fromarray(mask_aug)

            # 为增强后的样本赋予新的名称
            image_aug_name = os.path.join(save_path, 'train_images', \
                                            image_name.replace('.jpg', '_' + str(aug_index) + '.jpg'))
            mask_aug_name = os.path.join(save_path, 'train_mask', \
                                            mask_name.replace('.png', '_' + str(aug_index) + '.png'))
        
            image_aug.save(image_aug_name)
            mask_aug.save(mask_aug_name)
    else:
        # 对于负样本，按照概率进行扩增
        prob = random.random()
        if prob < 0.45:
            for aug_index, aug in enumerate(augs):
                image_aug, mask_aug = sample_aug(image, mask, aug)
                image_aug = Image.fromarray(image_aug)
                mask_aug = Image.fromarray(mask_aug)

                # 为增强后的样本赋予新的名称
                image_aug_name = os.path.join(save_path, 'train_images', \
                                                image_name.replace('.jpg', '_' + str(aug_index) + '.jpg'))
                mask_aug_name = os.path.join(save_path, 'train_mask', \
                                                mask_name.replace('.png', '_' + str(aug_index) + '.png'))
        
                image_aug.save(image_aug_name)
                mask_aug.save(mask_aug_name)

    return image_name, mask_pixel_num


def dataset_aug(dataset_root, save_root, augs=AUG):
    """对原始路径下的样本进行增强，并将结果保存至目标目录下

    Args:
        dataset_root: 原始数据集的根目录
        save_root: 目标目录的根目录
        augs: 将采用的增强方法
    """
    images_path = os.path.join(dataset_root, 'train_images')
    images_name = os.listdir(images_path)

    partial_aug = partial(aug_save, original_path=dataset_root, save_path=save_root, augs=augs)
    pool = Pool(40)
    
    for index, (image_name, mask_pixel_num) in enumerate(pool.imap(partial_aug, images_name)):
        descript = '%s :pixels, done %d / %d\r' % (str(bool(mask_pixel_num)), index + 1, len(images_name))
        sys.stdout.write(descript)

    pool.close()
    pool.join()
    
    pass


class DataAugThread (threading.Thread):
    def __init__(self, threadID, dataset_root, save_root, threadLock):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.dataset_root = dataset_root
        self.save_root = save_root
        self.threadLock = threadLock
    
    def run(self):
        # 获取锁，用于线程同步
        self.threadLock.acquire()
        dataset_aug(self.dataset_root, self.save_root)
        # 释放锁，开启下一个线程
        self.threadLock.release()


if __name__ == "__main__":
    # 创建所需文件夹，也可以直接使用os.makedirs()创建多级目录
    save_root = './SIIM_AUG'
    if not os.path.exists(save_root):
        os.mkdir(save_root)
        print("make: {}".format(save_root))
    
    train_images_path = os.path.join(save_root, 'train_images')
    if not os.path.exists(train_images_path):
        os.mkdir(train_images_path)
        print("make: {}".format(train_images_path))
    
    train_mask_path = os.path.join(save_root, 'train_mask')
    if not os.path.exists(train_mask_path):
        os.mkdir(train_mask_path)
        print("make: {}".format(train_mask_path))

    dataset_root = './SIIM_data'
    dataset_aug(dataset_root, save_root, augs=AUG)

    
