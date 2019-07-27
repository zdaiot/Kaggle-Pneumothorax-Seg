from PIL import Image
import numpy as np
import os


class DatasetsStatic(object):
    def __init__(self, data_root, image_folder, mask_folder, sort_flag):
        """
        Args: 
            data_root: 数据集的根目录
            image_folder: 样本文件夹名
            mask_folder: 掩膜文件夹名
            sort_flag: bool，控制样本路径是否随机排列
        """
        self.data_root = data_root
        self.image_folder = os.path.join(self.data_root, image_folder)
        self.mask_folder = os.path.join(self.data_root, mask_folder)
        self.sort_flag = sort_flag

    def mask_static_bool(self):
        """统计数据集中的每一个样本是否存在掩膜
        Return:
            images_path: 所有样本的路径
            masks_path: 样本对应的掩膜的路径
            masks_bool: 各样本是否有掩膜
        """
        image_names = os.listdir(self.image_folder)
        if self.sort_flag:
            image_names = sorted(image_names)
        
        images_path = list()
        masks_path = list()
        masks_bool = list()
        for index, image_name in enumerate(image_names):
            image_path = os.path.join(self.image_folder, image_name)
            mask_name = image_name.replace('jpg', 'png')
            mask_path = os.path.join(self.mask_folder, mask_name)

            mask_pixes_num = self.cal_mask_pixes(mask_path)
            mask_flag = bool(mask_pixes_num)

            images_path.append(image_path)
            masks_path.append(mask_path)
            masks_bool.append(mask_flag)
        
        return images_path, masks_path, masks_bool

    def mask_average_num(self):
        """统计数据集掩膜的平均大小
        """
    
    def mask_pixes_num_static(self):
        """统计数据集掩膜分布情况
        """


    def cal_mask_pixes(self, mask_path):
        """计算样本的标记的掩膜所包含的像素的总数
        Args:
            mask_path: 标记存放路径
        Return:
            mask_pixes: 掩膜的像素总数
        """
        mask_img = Image.open(mask_path)
        mask_np = np.asarray(mask_img)

        mask_pixes = np.sum(mask_np)

        return mask_pixes