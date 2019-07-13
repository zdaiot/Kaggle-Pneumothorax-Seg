import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from backboned_unet import Unet
import torch
from models.network import U_Net


def detect(model, image_path, input_size=224, threshold=0.6, cuda=True):
        model.eval()
        image = Image.open(image_path).convert('RGB')
        image_raw = image.resize((input_size, input_size))

        resize = transforms.Resize(input_size)(image)
        to_tensor = transforms.ToTensor()(resize)
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(to_tensor)
       
        image = torch.unsqueeze(normalize, dim=0)

        if cuda:
            image = image.cuda()
        with torch.no_grad(): 
            output = model(image)
        output = torch.nn.functional.sigmoid(output)
        pred = output.data.cpu().numpy()
        pred = np.where(pred>threshold, 1, 0)[0]
        pred = np.reshape(pred, (input_size, input_size)).astype(np.int8)*255.

        return image_raw, pred


def combine_display(image_raw, mask, pred, title_diplay):
    plt.suptitle(title_diplay)
    plt.subplot(1, 3, 1)
    plt.title('image_raw')
    plt.imshow(image_raw)

    plt.subplot(1, 3, 2)
    plt.title('mask')
    plt.imshow(mask)

    plt.subplot(1, 3, 3)
    plt.title('pred')
    plt.imshow(pred)

    plt.show()

def demo(model_name, checkpoint_path, images_path, masks_path, input_size=512, threshold=0.25, cuda=True):
    if model_name == 'U_Net':
        model = U_Net(img_ch=3, output_ch=1)
    elif model_name == 'unet_resnet34':
        model = Unet(backbone_name='resnet34', classes=1)
    else:
        raise ValueError('The model should be one of [Unet/unet_resnet34]')
    
    if cuda:
        model.cuda()
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    images = os.listdir(images_path)
    for image in images:
        image_path = os.path.join(images_path, image)
        image_raw, pred_mask = detect(model, image_path, input_size, threshold, cuda)

        mask_path = os.path.join(masks_path, image.replace('jpg', 'jpg'))
        mask = Image.open(mask_path).resize((input_size, input_size))

        combine_display(image_raw, mask, pred_mask, 'Result Show')


if __name__ == "__main__":
    base_dir = 'img'
    images_folder = os.path.join(base_dir, 'image')
    masks_folder = os.path.join(base_dir, 'mask')
    model_name = 'unet_resnet34'
    checkpoint_path = os.path.join('checkpoints', model_name, model_name+'_129.pth')

    demo(model_name, checkpoint_path, images_folder, masks_folder)

    