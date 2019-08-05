import os
from glob import glob
import numpy as np
import time
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm_notebook, tqdm
from utils.evaluation import *
from models.network import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from models.linknet import LinkNet34
from models.deeplabv3.deeplabv3plus import DeepLabV3Plus
from models.classify_unet import UnetWithClassify
from backboned_unet import Unet
import segmentation_models_pytorch as smp
import pandas as pd
from utils.mask_functions import rle2mask, mask2rle, mask_to_rle
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
import cv2
from albumentations import CLAHE
import json


class Test(object):
    def __init__(self, model_type, image_size, mean, std, t=None):
        # Models
        self.unet = None
        self.image_size = image_size # 模型的输入大小

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.t = t
        self.mean = mean
        self.std = std

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=3, output_ch=1)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=3, output_ch=1, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=3, output_ch=1)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3, output_ch=1, t=self.t)
        elif self.model_type == 'unet_resnet34':
            self.unet = UnetWithClassify(encoder_name='resnet34', classes=1)
        elif self.model_type == 'linknet':
            self.unet = LinkNet34(num_classes=1)
        elif self.model_type == 'deeplabv3plus':
            self.unet = DeepLabV3Plus(num_classes=1)
        elif self.model_type == 'pspnet_resnet34':
            self.unet = smp.PSPNet('resnet34', encoder_weights='imagenet', classes=1, activation=None)
        print('build model done！')

        self.unet.to(self.device)

    def test_model(self, threshold, stage, n_splits, test_best_model=True, less_than_sum=2048*2, csv_path=None, test_image_path=None):
        """
        threshold: 阈值，高于这个阈值的置为1，否则置为0
        stage: 测试第几阶段的结果
        n_splits: 测试多少折的结果进行平均
        test_best_model: 是否要使用最优模型测试，若不是的话，则取最新的模型测试
        less_than_sum: 预测图片中有预测出的正样本总和小于这个值时，则忽略所有
        """
        self.build_model()

        # 对于每一折加载模型，对所有测试集测试，并取平均
        sample_df = pd.read_csv(csv_path)
        preds = np.zeros([len(sample_df), self.image_size, self.image_size])

        for fold in range(n_splits):
            if test_best_model:
                unet_path = os.path.join('checkpoints', self.model_type, self.model_type+'_{}_{}_best.pth'.format(stage, 0))
            else:
                unet_path = os.path.join('checkpoints', self.model_type, self.model_type+'_{}_{}.pth'.format(stage, fold))
            self.unet.load_state_dict(torch.load(unet_path)['state_dict'])
            self.unet.eval()
            
            with torch.no_grad():
                # sample_df = sample_df.drop_duplicates('ImageId ', keep='last').reset_index(drop=True)
                for index, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
                    file = row['ImageId']
                    img_path = os.path.join(test_image_path, file.strip() + '.jpg')
                    img = Image.open(img_path).convert('RGB')
                    
                    pred = self.tta(img)
                    preds[index, ...] += np.reshape(pred, (self.image_size, self.image_size))
            # 如果取消注释，则只测试一个fold的
            n_splits = 1
            break

        rle = []
        count_has_mask = 0
        preds_average = preds/n_splits
        for index, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
            file = row['ImageId']

            pred = cv2.resize(preds_average[index,...],(1024, 1024))
            pred = np.where(pred>threshold, 1, 0)

            if np.sum(pred) < less_than_sum:
                pred[:] = 0
            encoding = mask_to_rle(pred.T, 1024, 1024)
            if encoding == ' ':
                rle.append([file.strip(), '-1'])
            else:
                count_has_mask += 1
                rle.append([file.strip(), encoding[1:]])

        print('The number of masked pictures predicted:',count_has_mask)
        submission_df = pd.DataFrame(rle, columns=['ImageId','EncodedPixels'])
        submission_df.to_csv('submission.csv', index=False)
    
    def image_transform(self, image):
        """对样本进行预处理
        """
        resize = transforms.Resize(self.image_size)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(self.mean, self.std)

        transform_compose = transforms.Compose([resize, to_tensor, normalize])

        return transform_compose(image)
    
    def detection(self, image):
        """对输入样本进行检测
        
        Args:
            image: 待检测样本，Image
        Return:
            pred: 检测结果
        """
        image = self.image_transform(image)
        image = torch.unsqueeze(image, dim=0)
        image = image.float().to(self.device)
        class_pred, seg_pred = self.unet(image)
        class_pred = torch.sigmoid(class_pred)
        seg_pred = torch.sigmoid(seg_pred)
        class_pred_mask = class_pred > 0.5
        class_pred_mask_flat = class_pred_mask.view(-1).float()
        pred = seg_pred * class_pred_mask_flat
        # 预测出的结果
        pred = pred.view(self.image_size, self.image_size)
        pred = pred.detach().cpu().numpy()

        return pred

    def tta(self, image):
        """执行TTA预测
        Args:
            image: Image图片
        Return:
            pred: 最后预测的结果
        """
        preds = np.zeros([self.image_size, self.image_size])
        # 左右翻转
        image_hflip = image.transpose(Image.FLIP_LEFT_RIGHT)

        hflip_pred = self.detection(image_hflip)
        hflip_pred_img = Image.fromarray(hflip_pred)
        pred_img = hflip_pred_img.transpose(Image.FLIP_LEFT_RIGHT)
        preds += np.asarray(pred_img)

        # CLAHE
        aug = CLAHE(p=1.0)
        image_np = np.asarray(image)
        clahe_image = aug(image=image_np)['image']
        clahe_image = Image.fromarray(clahe_image)
        clahe_pred = self.detection(clahe_image)
        preds += clahe_pred

        # 原图
        original_pred = self.detection(image)
        preds += original_pred

        # 求平均
        pred = preds / 3

        return pred


if __name__ == "__main__":
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # mean = (0.490, 0.490, 0.490)
    # std = (0.229, 0.229, 0.229)
    csv_path = './submission.csv' 
    test_image_path = 'datasets/SIIM_data/test_images'
    model_name = 'unet_resnet34'
    # stage表示测试第几阶段的代码，对应不同的image_size，index表示为交叉验证的第几个
    stage, n_splits = 2, 5
    if stage == 1:
        image_size = 768
    elif stage == 2:
        image_size = 1024
    solver = Test(model_name, image_size, mean, std)
    solver.test_model(threshold=0.38, stage=stage, n_splits=n_splits, test_best_model=True, less_than_sum=1024, csv_path=csv_path, test_image_path=test_image_path)
