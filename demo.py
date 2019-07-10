import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import cv2

import torch
from models.network import U_Net


def detect(model, image_path, input_size=224, threshold=0.6, cuda=True):
        model.eval()
        image = Image.open(image_path).convert('RGB')
        image_raw = np.array(image)

        resize = transforms.Resize(input_size)(image)
        to_tensor = transforms.ToTensor()(resize)
        normalize = transforms.Normalize((0.490, 0.490, 0.490), (0.229, 0.229, 0.229))(to_tensor)
       
        image = torch.unsqueeze(normalize, dim=0)

        if cuda:
            image = image.cuda()
        with torch.no_grad(): 
            output = model(image)
        output = torch.nn.functional.sigmoid(output)
        pred = output.data.cpu().numpy()
        pred = np.where(pred>threshold, 1, 0)[0]
        pred = np.reshape(pred, (input_size, input_size)).astype(np.int8)*255.

        cv2.imshow(image_path.split('/')[-1].split('.')[-2], pred)
        cv2.waitKey(0)
        pass



def demo(model_name, checkpoint_path, images_path, input_size=224, threshold=0.25, cuda=True):
    if model_name == 'U_Net':
        model = U_Net(img_ch=3, output_ch=1)
    else:
        raise ValueError('The model should be one of [Unet ]')
    
    if cuda:
        model.cuda()
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint.state_dict())

    images = os.listdir(images_path)
    for image in images:
        image_path = os.path.join(images_path, image)
        detect(model, image_path, input_size, threshold, cuda)


if __name__ == "__main__":
    images_floder = 'img/image'
    checkpoint_path = 'checkpoints/U_Net/U_Net_79.pth'

    demo('U_Net', checkpoint_path, images_floder)

    