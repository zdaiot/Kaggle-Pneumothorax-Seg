import numpy as np
import cv2
import random
import tqdm
import os

def cal_mean_std(images_txt, images_path, CNum, channels):
    """计算给定数据集的通道均值和方差
    Args:
        images_txt：txt文件，存放数据集样本名称
        images_path：数据集路径
        CNum：参与计算的样本数目
        channels：图片通道数
    Return:
        means：各个通道的均值
        stdevs：各个通道的方差
    """
    means = [0] * channels
    stdevs = [0] * channels
    with open(images_txt, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)   # shuffle , 随机挑选图片
        for i in tqdm.tqdm(range(CNum)):
            img_path = os.path.join(images_path, lines[i])[:-1]
            img_path = img_path + '.jpg'
            img = cv2.imread(img_path)
            img = img.astype(np.float32) / 255.0
            # 计算各个通道的均值和方差
            for channel in range(channels):
                means[channel] += img[:, :, channel].mean()
                stdevs[channel] += img[:, :, channel].std() 
    
    # cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
    means.reverse() # BGR --> RGB
    stdevs.reverse()
    # 取平均
    means = np.asarray(means) / CNum
    stdevs = np.asanyarray(stdevs) / CNum

    return means, stdevs


if __name__ == "__main__":
    # 包含样本名称的txt文件的路径
    train_txt_path = '/home/apple/data/MXQ/competition/kaggle/Pneumothorax Segmentation/VOC/ImageSets/Segmentation/train.txt'
    # dataset path
    dataset_train_path = '/home/apple/data/MXQ/competition/kaggle/Pneumothorax Segmentation/VOC/JPEGImages'

    CNum = 10712
    means, stdevs = cal_mean_std(train_txt_path, dataset_train_path, CNum, 3)
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))