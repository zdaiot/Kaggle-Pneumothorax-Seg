import os
from glob import glob
import numpy as np
import time
import torch
import torchvision
from PIL import Image
from torch import optim
from tqdm import tqdm_notebook, tqdm
from utils.evaluation import *
from models.network import U_Net, R2U_Net, AttU_Net, R2AttU_Net
import pandas as pd
from utils.mask_functions import rle2mask, mask2rle, mask_to_rle
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
from backboned_unet import Unet
import cv2
from albumentations import CLAHE


class Test(object):
    def __init__(self, model_type, weight_path, image_size, mean, std, t=None, less_than_sum=2048*2):
        # Models
        self.unet = None
        self.optimizer = None
        self.image_size = image_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.t = t
        self.unet_path = weight_path
        self.mean = mean
        self.std = std
        self.less_than_sum = less_than_sum

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
            self.unet = Unet(backbone_name='resnet34', classes=1)
        print('build model done！')

        self.unet.to(self.device)

    def test_model(self, threshold=0.5, csv_path=None, test_image_path=None):
        """
        """
        self.build_model()
        self.unet.load_state_dict(torch.load(self.unet_path)['state_dict'])

        self.unet.train(False)
        rle = []
        sample_df = pd.read_csv(csv_path)
        count_has_mask = 0

        with torch.no_grad():
            # sample_df = sample_df.drop_duplicates('ImageId ', keep='last').reset_index(drop=True)
            for index, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
                file = row['ImageId']
                img_path = os.path.join(test_image_path, file.strip() + '.jpg')
                img = Image.open(img_path).convert('RGB')

                aug = CLAHE(p=1.0)
                img = np.asarray(img)
                img = aug(image=img)['image']
                img = Image.fromarray(img)

                img = self.image_transform(img)
                img = torch.unsqueeze(img, dim=0)

                img = img.float().to(self.device)
                pred = torch.sigmoid(self.unet(img))
                pred = pred.detach().cpu().numpy()
                pred = cv2.resize(pred[0,0,...],(1024, 1024))
                pred = np.where(pred>threshold, 1, 0)      
                if np.sum(pred) < self.less_than_sum:
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


if __name__ == "__main__":
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    csv_path = './submission.csv' 
    test_image_path = 'datasets/SIIM_data/test_images'
    model_name = 'unet_resnet34'
    # stage表示测试第几阶段的代码，对应不同的image_size，index表示为交叉验证的第几个
    stage, index = 1, 0
    if stage == 1:
        image_size = 1024
    elif stage == 2:
        image_size = 1024
    checkpoint_path = os.path.join('checkpoints', model_name, model_name+'_{}_{}.pth'.format(stage, index))
    solver = Test(model_name, checkpoint_path, image_size, mean, std, less_than_sum=2048*2)
    solver.test_model(threshold=0.30, csv_path=csv_path, test_image_path=test_image_path)