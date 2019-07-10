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


class Test(object):
    def __init__(self, model_type, t):
        # Models
        self.unet = None
        self.optimizer = None
        
        # Path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.t = t
        self.unet_path = '/home/apple/program/MXQ/Competition/Kaggle/Pneumothorax-Seg/Image_Segmentation-master/checkpoints/U_Net/U_Net_167.pth'
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=3, output_ch=2)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=3, output_ch=1, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=3, output_ch=1)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3, output_ch=1, t=self.t)
        print('build model done！')

        self.unet.to(self.device)

    def test_model(self):
        # ===================================== Test ====================================#
        del self.unet
        self.build_model()
        self.unet.load_state_dict(torch.load(self.unet_path).state_dict())

        self.unet.train(False)
        self.unet.eval()
        rle = []
        sample_df = pd.read_csv("/home/apple/data/zdaiot/Image_Segmentation/input/sample_submission.csv")
        # sample_df = sample_df.drop_duplicates('ImageId                                               ', keep='last').reset_index(drop=True)
        for index, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
            file = row['ImageId                                               ']
            img_path = os.path.join('/home/apple/data/zdaiot/Image_Segmentation/input/test_images', file.strip() + '.jpg')
            img = Image.open(img_path)
            img = img.resize((512, 512))
            img = np.array(img.convert('RGB'))

            # clahe = CLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # img = clahe(img)
            img = img / 255.0

            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
            img = transform(img).view(1, 3, 512, 512)
            img = img.float().to(self.device)
            pred = torch.nn.functional.softmax(self.unet(img))
            pred = pred.detach().cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred = np.reshape(pred, (512, 512))
            # plt.imshow(pred)
            # plt.show()
            encoding = mask_to_rle(pred, 512, 512)
            if encoding == ' ':
                rle.append([file.strip(), '-1'])
            else:
                rle.append([file.strip(), encoding[1:]])

        submission_df = pd.DataFrame(rle, columns=['ImageId','EncodedPixels'])
        submission_df.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    solver = Test('U_Net', 2) 
    solver.build_model()
    solver.test_model()
