from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np


class DatasetsStatic(object):
    def __init__(self, data_root, image_folder, mask_folder, sort_flag=True):
        """
        Args: 
            data_root: 数据集的根目录
            image_folder: 样本文件夹名
            mask_folder: 掩膜文件夹名
            sort_flag: bool，是否对样本路径进行排序
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

    def mask_pixes_average_num(self):
        """统计每个样本所包含的掩膜的像素的平均数目

        Return:
            average: 每个样本所包含的像素的平均数目
        """
        mask_names = os.listdir(self.mask_folder)
        # 有掩膜的样本的总数
        mask_num = 0
        # 掩膜像素总数
        mask_pixes_sum = 0
        for index, mask_name in enumerate(mask_names):
            mask_path = os.path.join(self.mask_folder, mask_name)
            mask_pixes_num = self.cal_mask_pixes(mask_path)
            
            mask_pixes_sum += mask_pixes_num
            if mask_pixes_num:
                mask_num += 1
        
        average = mask_pixes_sum / mask_num
        return average
    
    def mask_num_static(self):
        """统计数据集掩膜分布情况
        """
        mask_names = os.listdir(self.mask_folder)
        # 各样本掩膜的像素数目
        mask_pix_num = list()

        for index, mask_name in enumerate(mask_names):
            mask_path = os.path.join(self.mask_folder, mask_name)
            
            mask_pix_per_image = self.cal_mask_pixes(mask_path)
            if mask_pix_per_image:
                mask_pix_num.append(mask_pix_per_image)

        masks = len(mask_pix_num)

        mask_pix_num_np = np.asarray(mask_pix_num)
        # 掩膜像素数目的最小值
        pix_num_min = np.min(mask_pix_num_np)
        # 掩膜像素数目的最大值
        pix_num_max = np.max(mask_pix_num_np)
        # 具有最小的掩膜像素数目的样本个数
        mask_num_pix_min = np.sum(mask_pix_num_np == pix_num_min)
        # 具有最大的掩膜像素数目的样本个数
        mask_num_pix_max = np.sum(mask_pix_num_np == pix_num_max)

        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(9, 6))
        ax0.hist(mask_pix_num, 100, histtype='bar', facecolor='yellowgreen', alpha=0.75, log=True)
        ax0.set_title('Pixes Num of Mask')
        # ax0.text(pix_num_min, mask_num_pix_min, 'min: %d, number: %d'%(pix_num_min, mask_num_pix_min))
        # ax0.text(pix_num_max, mask_num_pix_max, 'max: %d, number: %d'%(pix_num_max, mask_num_pix_max))

        ax0.annotate('min: %d, number: %d'%(pix_num_min, mask_num_pix_min), 
                    xy=(pix_num_min, mask_num_pix_min), xytext=(pix_num_min, mask_num_pix_min+5),
                    arrowprops=dict(facecolor='blue', shrink=0.0005))
        ax0.annotate('max: %d, number: %d'%(pix_num_max, mask_num_pix_max), 
                    xy=(pix_num_max, mask_num_pix_max), xytext=(pix_num_max-20000, mask_num_pix_max+5),
                    arrowprops=dict(facecolor='green', shrink=0.0005))        
        ax0.text(2000, 2000, "masks number: %d"%(masks))

        ax1.hist(mask_pix_num, 100, histtype='bar', facecolor='pink', alpha=0.75, cumulative=True, rwidth=0.8)
        ax1.set_title('Accumlation Pixes Num of Mask')
        fig.subplots_adjust(hspace=0.4)
        plt.show()

    def cal_mask_pixes(self, mask_path):
        """计算样本的标记的掩膜所包含的像素的总数
        Args:
            mask_path: 标记存放路径
        Return:
            mask_pixes: 掩膜的像素总数
        """
        mask_img = Image.open(mask_path)
        mask_np = np.asarray(mask_img)
        mask_np = mask_np > 0

        mask_pixes = np.sum(mask_np)

        return mask_pixes


if __name__ == "__main__":
    dataset_root = 'datasets/SIIM_data'
    images_folder = 'train_images'
    masks_folder = 'train_mask'
    ds = DatasetsStatic(dataset_root, images_folder, masks_folder)

    ds.mask_num_static()
    average_num = ds.mask_pixes_average_num()
    print('average num: %d'%(average_num))

    pass