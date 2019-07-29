import os, glob
import torch
import numpy as np
import pandas as pd
import collections
import torch

from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.mask_functions import rle2mask
from utils.data_augmentation import data_augmentation


# SIIM Dataset Class
class SIIMDataset(torch.utils.data.Dataset):
    """从csv标注文件中抽取有标记的样本用作训练集
    """
    def __init__(self, train_image, train_mask, image_size, augmentation_flag, compare_image_mask_path=False):
        """
        Args:
            param df_path: csv文件的路径
            img_dir: 训练样本图片的存放路径
            image_size: 模型的输入图片尺寸
        """
        super(SIIMDataset).__init__()
        self.class_num = 2

        self.image_size = image_size
        # 是否使用数据增强
        self.augmentation_flag = augmentation_flag
        # self.mean = (0.490, 0.490, 0.490)
        # self.std = (0.229, 0.229, 0.229)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)    

        # 所有样本和掩膜的名称
        self.image_names = train_image
        self.mask_names = train_mask
        self.compare_image_mask_path = compare_image_mask_path

    def __getitem__(self, idx):
        """得到样本与其对应的mask
        Return:
            img: 经过预处理的样本图片
            mask: 值为0/1，0表示属于背景，1表示属于目标类
        """
        # 依据idx读取样本图片
        img_path = self.image_names[idx]
        img = Image.open(img_path).convert("RGB")
        # 依据idx读取掩膜
        mask_path = self.mask_names[idx]
        mask = Image.open(mask_path)

        if self.compare_image_mask_path:
            assert img_path.split('/')[-1][:-4] == mask_path.split('/')[-1][:-4]

        if self.augmentation_flag:
            img, mask = self.augmentation(img, mask)

        # 对图片和mask同时进行转换
        img = self.image_transform(img)
        mask = self.mask_transform(mask)

        return img, mask

    def image_transform(self, image):
        """对样本进行预处理
        """
        resize = transforms.Resize(self.image_size)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(self.mean, self.std)

        transform_compose = transforms.Compose([resize, to_tensor, normalize])

        return transform_compose(image)

    def mask_transform(self, mask):
        """对mask进行预处理
        """
        mask = mask.resize((self.image_size, self.image_size))
        # 将255转换为1， 0转换为0
        mask = np.around(np.array(mask.convert('L'))/256.)
        # # mask = mask[:, :, np.newaxis] # Wrong, will convert range
        # mask = np.reshape(mask, (np.shape(mask)[0],np.shape(mask)[1],1)).astype("float32")
        # to_tensor = transforms.ToTensor()

        # transform_compose = transforms.Compose([to_tensor])
        # mask = transform_compose(mask)
        # mask = torch.squeeze(mask)
        mask = torch.from_numpy(mask)
        return mask.float()
    
    def augmentation(self, image, mask):
        """进行数据增强
        Args:
            image: 原始图像，Image图像
            mask: 原始掩膜，Image图像
        Return:
            image_aug: 增强后的图像，Image图像
            mask: 增强后的掩膜，Image图像
        """
        image = np.asarray(image)
        mask = np.asarray(mask)

        image_aug, mask_aug = data_augmentation(image, mask)

        image_aug = Image.fromarray(image_aug)
        mask_aug = Image.fromarray(mask_aug)

        return image_aug, mask_aug

    def __len__(self):
        return len(self.image_names)


def get_loader(train_image, train_mask, val_image, val_mask, image_size=224, batch_size=2, num_workers=2, augmentation_flag=False):
    """Builds and returns Dataloader."""
    # train loader
    dataset_train = SIIMDataset(train_image, train_mask, image_size, augmentation_flag)
    train_data_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    # val loader, 验证集要保证augmentation_flag为False
    dataset_val = SIIMDataset(val_image, val_mask, image_size, augmentation_flag=False)
    val_data_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)

    return train_data_loader, val_data_loader


if __name__ == "__main__":
    mask_path = "datasets/SIIM_data/train_mask"
    image_path = "datasets/SIIM_data/train_images"
    batch_size = 16

    dataset_train = SIIMDataset(glob.glob(image_path+'/*.jpg'), glob.glob(mask_path+'/*.png'), 512, False)
    print(len(dataset_train))

    dataloader = DataLoader(dataset_train, batch_size=batch_size, num_workers=32, shuffle=True, pin_memory=True)

    import cv2
    data = iter(dataloader)
    tbar = tqdm(data)
    error_mask_count = 0
    for index, (images, masks) in enumerate(tbar):
        for i in range(images.size(0)):
            image = images[i]
            mask_max = torch.max(masks[i])
            mask_min = torch.min(masks[i])

            descript = 'Mask_max %d, Mask_min %d'%(mask_max, mask_min)
            tbar.set_description(descript)
            if mask_max != 1 and mask_max != 0:
                error_mask_count += 1

            mask = masks[i].float()*255
            image = image + mask

            image = image.permute(1, 2, 0).numpy()
            cv2.imshow('win', image)
            cv2.waitKey(1000)

    if error_mask_count != 0:
        print("There exits wrong mask...")