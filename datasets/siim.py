import os, glob
import torch
import numpy as np
import pandas as pd
import collections
import torch
import random

from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.mask_functions import rle2mask
from utils.data_augmentation import data_augmentation
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.dataloader import default_collate
import pickle


# SIIM Dataset Class
class SIIMDataset(torch.utils.data.Dataset):
    """从csv标注文件中抽取有标记的样本用作训练集
    """
    def __init__(self, train_image, train_mask, aug_flag=False):
        """
        Args:
            param df_path: csv文件的路径
            img_dir: 训练样本图片的存放路径
        """
        super(SIIMDataset).__init__()
        self.class_num = 2

        # 所有样本和掩膜的名称
        self.image_names = train_image
        self.mask_names = train_mask
        self.aug_flag = aug_flag

    def __getitem__(self, idx):
        """得到样本与其对应的mask
        Return:
            img: 样本图片
            mask: 掩膜
        """
        # 依据idx读取样本图片
        img_path = self.image_names[idx]
        img = Image.open(img_path).convert("RGB")
        # 依据idx读取掩膜
        mask_path = self.mask_names[idx]
        mask = Image.open(mask_path)
        if self.aug_flag:
            img, mask = augmentation(img, mask)

        img = np.asarray(img)
        mask = np.asarray(mask)
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        return img, mask

    def __len__(self):
        return len(self.image_names)

# TODO 测试下面这几个转换函数
def image_transform(image, image_size):
    """对样本进行预处理

    Args:
        image: PIL图像
        image_size: resize的图像大小
    Return:
        转换为tensor的image
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225) 
    resize = transforms.Resize(image_size)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)

    transform_compose = transforms.Compose([resize, to_tensor, normalize])

    return transform_compose(image)


def mask_transform(mask, image_size):
    """对mask进行预处理

    Args:
        mask: PIL image
        image_size: resize后的图像大小
    
    Return:
        mask: tensor
    """
    mask = mask.resize((image_size, image_size))
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


def augmentation(image, mask):
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


def exist_mask(mask):
    """判断是否存在掩膜
    """
    pix_max = torch.max(mask)
    if pix_max == 1:
        flag = 1
    elif pix_max == 0:
        flag = 0
    
    return flag


def weight_mask(dataset, weights_sample=[1, 3]):
    """计算每一个样本的权重

    Args:
        dataset: 数据集
        weight_sample: 正负类样本对应的采样权重
    
    Return:
        weights: 每一个样本对应的权重 
    """
    print('Start calculating weights of sample...')
    weights = list()
    tbar = tqdm(dataset)
    for index, (image, mask) in enumerate(tbar):
        flag = exist_mask(mask)
        # 存在掩膜的样本的采样权重为3，不存在的为1
        if flag:
            weights.append(weights_sample[1])
        else:
            weights.append(weights_sample[0])
        descript = 'Image %d, flag: %d' % (index, flag)
        tbar.set_description(descript)

    print('Finish calculating weights of sample...')
    return weights


def multi_scale_collate_fn(batch):
    """自定义collate_fn函数，用于对图片和掩膜进行resize
    """
    scale = random.choice([512, 768, 1024])
    batch_scale = list()
    for image, mask in batch:
        pair = list()
        image, mask = image.numpy(), mask.numpy()
        image, mask = Image.fromarray(image), Image.fromarray(mask)
        image = image_transform(image, scale)
        mask = mask_transform(mask, scale)
        pair.append(image)
        pair.append(mask)
        batch_scale.append(pair)
    batch_scale = default_collate(batch_scale)

    return batch_scale


def get_loader(train_image, train_mask, val_image, val_mask, batch_size=2, num_workers=2, weights_sample=None):
    """Builds and returns Dataloader."""
    # train loader
    dataset_train = SIIMDataset(train_image, train_mask, aug_flag=True)
    # val loader, 验证集要保证augmentation_flag为False
    dataset_val = SIIMDataset(val_image, val_mask, aug_flag=False)    
    
    # 依据weigths_sample决定是否对训练集的样本进行采样
    if weights_sample:
        if os.path.exists('weights_sample.pkl'):
            print('Extract weights of sample from: weights_sample.pkl...')
            with open('weights_sample.pkl', 'rb') as f:
                weights = pickle.load(f)
        else:
            print('Calculating weights of sample...')
            weights = weight_mask(dataset_train, weights_sample)
            with open('weights_sample.pkl', 'wb') as f:
                pickle.dump(weights, f)            
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset_train), replacement=True)
        train_data_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, sampler=sampler, 
                                        collate_fn=multi_scale_collate_fn, pin_memory=True)
    else: 
        train_data_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True, 
                                        collate_fn=multi_scale_collate_fn, pin_memory=True)
    
    val_data_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=True, 
                                    collate_fn=multi_scale_collate_fn, pin_memory=True)

    return train_data_loader, val_data_loader


if __name__ == "__main__":
    mask_path = "datasets/SIIM_data/train_mask"
    image_path = "datasets/SIIM_data/train_images"
    batch_size = 16
    images_name = glob.glob(image_path+'/*.jpg')
    
    masks_name = list()
    for image_name in images_name:
        mask_name = image_name.replace('jpg', 'png')
        mask_name = mask_name.replace('train_images', 'train_mask')
        masks_name.append(mask_name)

    dataset_train = SIIMDataset(images_name, masks_name, aug_flag=True)
    print(len(dataset_train))

    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=multi_scale_collate_fn, pin_memory=True)

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

            mask = masks[i]*255
            image = image + mask

            image = image.permute(1, 2, 0).numpy()

            image_size = image.shape[0]
            cv2.imshow('shape:%d'%image_size, image)
            cv2.waitKey(0)

    if error_mask_count != 0:
        print("There exits wrong mask...")