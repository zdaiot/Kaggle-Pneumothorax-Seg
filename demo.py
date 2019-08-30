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
import json

def load_image(mean, std, image_path, input_size, cuda=True):
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
    return image_raw, image

def detect(model, image, threshold, less_than_sum):
    with torch.no_grad(): 
        output = model(image)
    output = torch.sigmoid(output)
    pred = output.data.cpu().numpy()
    pred = np.where(pred>threshold, 1, 0)[0]
    if np.sum(pred) < less_than_sum:
        pred[:] = 0
    return pred

def detect_stage3(model, image_tensor, pred_stage2, threshold_stage3):
    '''
    model: 第三阶段的模型
    image: 第二阶段的预测结果
    input_size: 第三阶段的图像尺寸
    threshold_stage3: 第三阶段的阈值
    '''
    if np.sum(pred_stage2) > 0:
        return detect(model, image_tensor, threshold=threshold_stage3, less_than_sum=0)
    else:
        return pred_stage2

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

def combine_display_stage3(image_raw, mask, pred_stage2, pred_stage3, title_diplay):
    plt.suptitle(title_diplay)
    plt.subplot(1, 4, 1)
    plt.title('image_raw')
    plt.imshow(image_raw)

    plt.subplot(1, 4, 2)
    plt.title('mask')
    plt.imshow(mask)

    plt.subplot(1, 4, 3)
    plt.title('pred_stage2')
    plt.imshow(pred_stage2)

    plt.subplot(1, 4, 4)
    plt.title('pred_stage3')
    plt.imshow(pred_stage3)

    plt.show()

def load_model(model_name, checkpoint_path, cuda=True):
    if model_name == 'U_Net':
        model = U_Net(img_ch=3, output_ch=1)
    elif model_name == 'unet_resnet34':
        # model = Unet(backbone_name='resnet34', classes=1)
        model = smp.Unet('resnet34', encoder_weights='imagenet', activation=None)
    elif model_name == 'linknet':
        model = LinkNet34(num_classes=1)
    elif model_name == 'deeplabv3plus':
        model = DeepLabV3Plus(model_backbone='res50_atrous', num_classes=1)
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
    return model

def demo(model_name, mean, std, checkpoint_path, images_path, masks_path, input_size, threshold, less_than_sum, use_stage3, threshold_stage3, cuda=True):
    model = load_model(model_name, checkpoint_path)

    if use_stage3:
        tmp = checkpoint_path.split('_')
        tmp[3] = '3'
        checkpoint_path_stage3 = '_'.join(tmp)
        model_stage3 = load_model(model_name, checkpoint_path_stage3)

    images = os.listdir(images_path)
    for image in images:
        image_path = os.path.join(images_path, image)
        image_raw, image_tensor = load_image(mean, std, image_path, input_size=input_size)
        pred_stage2 = detect(model, image_tensor, threshold=threshold, less_than_sum=less_than_sum)
        pred_stage2_reshape = np.reshape(pred_stage2, (input_size, input_size)).astype(np.int8)*255.

        mask_path = os.path.join(masks_path, image.replace('jpg', 'png'))
        mask = Image.open(mask_path).resize((input_size, input_size))

        if use_stage3:
            pred_stage3 = detect_stage3(model_stage3, image_tensor, pred_stage2, threshold_stage3=threshold_stage3)
            pred_stage3_reshape = np.reshape(pred_stage3, (input_size, input_size)).astype(np.int8)*255.
            combine_display_stage3(image_raw, mask, pred_stage2_reshape, pred_stage3_reshape, 'Result Show for stage2 and stage3')
        else:
            combine_display(image_raw, mask, pred_stage2_reshape, 'Result Show for stage2')


if __name__ == "__main__":
    base_dir = 'datasets/SIIM_data'
    images_folder = os.path.join(base_dir, 'test_images')
    masks_folder = os.path.join(base_dir, 'test_mask')
    model_name = 'unet_resnet34'

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # mean = (0.490, 0.490, 0.490)
    # std = (0.229, 0.229, 0.229)

    # stage表示测试第几阶段的代码，对应不同的image_size，fold表示为交叉验证的第几个
    stage, fold = 2, 4
    if stage == 1:
        image_size = 768
    elif stage == 2:
        image_size = 1024
    use_stage3 = False

    with open('checkpoints/'+model_name+'/result_stage2.json', 'r', encoding='utf-8') as json_file:
        config_cla = json.load(json_file)
    
    with open('checkpoints/'+model_name+'/result_stage3.json', 'r', encoding='utf-8') as json_file:
        config_seg = json.load(json_file)
    
    thresholds_stage2 = config_cla[str(fold)][0]
    less_than_sum = config_cla[str(fold)][1]
    thresholds_stage3 = config_seg[str(fold)][0]

    checkpoint_path = os.path.join('checkpoints', model_name, model_name+'_{}_{}_best.pth'.format(stage, fold))
    demo(model_name, mean, std, checkpoint_path, images_folder, masks_folder, image_size, \
        threshold=thresholds_stage2, less_than_sum=less_than_sum, use_stage3=use_stage3, threshold_stage3=thresholds_stage3)