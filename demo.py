import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import torch
from backboned_unet import Unet
import segmentation_models_pytorch as smp
from models.network import U_Net
from models.linknet import LinkNet34
from albumentations import CLAHE
from models.deeplabv3.deeplabv3plus import DeepLabV3Plus
from models.Transpose_unet.unet.model import Unet as Unet_t
from models.octave_unet.unet.model import OctaveUnet


def detect(model, mean, std, image_path, input_size=224, threshold=0.6, cuda=True):
    image = Image.open(image_path).convert('RGB')
    image_raw = image.resize((input_size, input_size))

    # aug = CLAHE(p=1.0)
    # image = np.asarray(image)
    # image = aug(image=image)['image']
    # image = Image.fromarray(image)

    resize = transforms.Resize(input_size)(image)
    to_tensor = transforms.ToTensor()(resize)
    normalize = transforms.Normalize(mean, std)(to_tensor)
    
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

def demo(model_name, mean, std, checkpoint_path, images_path, masks_path, input_size=512, threshold=0.25, cuda=True):
    if model_name == 'U_Net':
        model = U_Net(img_ch=3, output_ch=1)
    elif model_name == 'unet_resnet34':
        # model = Unet(backbone_name='resnet34', classes=1)
        model = smp.Unet('resnet34', encoder_weights='imagenet', activation=None)
    elif model_name == 'linknet':
        model = LinkNet34(num_classes=1)
    elif model_name == 'deeplabv3plus':
        model = DeepLabV3Plus(model_backbone='res50_atrous', num_classes=self.output_ch)
        # model = DeepLabV3Plus(num_classes=1)
    elif model_name == 'pspnet_resnet34':
        model = smp.PSPNet('resnet34', encoder_weights='imagenet', classes=1, activation=None)
    elif model_name == 'unet_se_resnext50_32x4d':
        model = smp.Unet('se_resnext50_32x4d', encoder_weights='imagenet', activation=None)
    elif model_name == 'unet_resnet50':
        model = smp.Unet('resnet50', encoder_weights='imagenet', activation=None)
    elif model_name == 'unet_resnet34_t':
        model = Unet_t('resnet34', encoder_weights='imagenet', activation=None, use_ConvTranspose2d=True)
    elif model_name == 'unet_resnet34_oct':
        model = OctaveUnet('resnet34', encoder_weights='imagenet', activation=None)
    else:
        raise ValueError('The model should be one of [Unet/unet_resnet34/]')
    
    if cuda:
        model.cuda()
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.train(False)
    images = os.listdir(images_path)
    for image in images:
        image_path = os.path.join(images_path, image)
        image_raw, pred_mask = detect(model, mean, std, image_path, input_size, threshold, cuda)

        mask_path = os.path.join(masks_path, image.replace('jpg', 'png'))
        mask = Image.open(mask_path).resize((input_size, input_size))

        combine_display(image_raw, mask, pred_mask, 'Result Show')


if __name__ == "__main__":
    base_dir = 'datasets/SIIM_data'
    images_folder = os.path.join(base_dir, 'train_images')
    masks_folder = os.path.join(base_dir, 'train_mask')
    model_name = 'unet_se_resnext50_32x4d'

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # mean = (0.490, 0.490, 0.490)
    # std = (0.229, 0.229, 0.229)

    # stage表示测试第几阶段的代码，对应不同的image_size，fold表示为交叉验证的第几个
    stage, fold = 1, 0
    if stage == 1:
        image_size = 768
    elif stage == 2:
        image_size = 1024
    checkpoint_path = os.path.join('checkpoints', model_name, model_name+'_{}_{}_best.pth'.format(stage, fold))
    demo(model_name, mean, std, checkpoint_path, images_folder, masks_folder, image_size, threshold=0.65)