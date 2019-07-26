import matplotlib.image as mpimg
import numpy as np
import cv2,os,glob
from skimage import exposure

# 切割数据和类标，统计切割后的图片中有多少个含有mask，对数据进行直方图均衡化
def cut_img_mask(data_type):
    src_path = '../input/{}_images'.format(data_type)
    save_path = '../input/{}_images_cut/'.format(data_type)
    save_path_mask = '../input/{}_mask_cut/'.format(data_type)
    pic_list = glob.glob(src_path+'/*.jpg')

    if not os.path.exists(save_path):
        print('Making folder...')
        os.makedirs(save_path)
    if not os.path.exists(save_path_mask):
        print('Making folder...')
        os.makedirs(save_path_mask)
    count_no_mask, count_mask = 0, 0

    for picture_path in pic_list:
        img = cv2.imread(picture_path, cv2.IMREAD_GRAYSCALE)
        img_shape = np.shape(img)

        mask_path = picture_path.replace('images','mask')
        mask_path = mask_path.replace('jpg','png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # 读入灰度图

        size = 256
        height_number = img_shape[0]/size  # 4
        width_number = img_shape[1]/size  # 4
        # print('cut height_number:{}, width_number:{}'.format(height_number, width_number))
        assert img_shape == np.shape(mask)

        count = 0
        for x in range(int(height_number)):
            for y in range(int(width_number)):
                # 裁剪数据并保存
                cut_img = img[x*size:(x+1)*size, y*size:(y+1)*size]
                cut_img = exposure.equalize_adapthist(cut_img) # contrast correction 对比度受限自适应直方图均衡(CLAHE)
                cut_img = ((cut_img*255)).clip(0,255).astype(np.uint8) # 再次归一化到255
                # print("%s%s.%s.jpg" %(save_path, picture_path.split('/')[-1][:-4], count))
                cv2.imwrite("%s%s.%s.jpg" %(save_path, picture_path.split('/')[-1][:-4], count), cut_img)

                # 裁剪类标并保存
                cut_mask = mask[x*size:(x+1)*size, y*size:(y+1)*size]
                if np.sum(cut_mask)==0: count_no_mask += 1 
                else: count_mask += 1
                # print("%s%s.%s.png" %(save_path_mask, picture_path.split('/')[-1][:-4], count))
                cv2.imwrite("%s%s.%s.png" %(save_path_mask, picture_path.split('/')[-1][:-4], count), cut_mask)

                count += 1
    print('count_no_mask:{}, count_mask:{}'.format(count_no_mask, count_mask))

# 切割数据和类标，统计切割后的图片中有多少个含有mask，对数据进行直方图均衡化
def cut_img(data_type):
    src_path = '../input/{}_images'.format(data_type)
    save_path = '../input/{}_images_cut/'.format(data_type)
    pic_list = glob.glob(src_path+'/*.jpg')

    if not os.path.exists(save_path):
        print('Making folder...')
        os.makedirs(save_path)

    for picture_path in pic_list:
        img = cv2.imread(picture_path, cv2.IMREAD_GRAYSCALE)
        img_shape = np.shape(img)

        size = 256
        height_number = img_shape[0]/size  # 4
        width_number = img_shape[1]/size  # 4
        # print('cut height_number:{}, width_number:{}'.format(height_number, width_number))

        count = 0
        for x in range(int(height_number)):
            for y in range(int(width_number)):
                # 裁剪数据并保存
                cut_img = img[x*size:(x+1)*size, y*size:(y+1)*size]
                cut_img = exposure.equalize_adapthist(cut_img) # contrast correction 对比度受限自适应直方图均衡(CLAHE)
                cut_img = ((cut_img*255)).clip(0,255).astype(np.uint8) # 再次归一化到255
                # print("%s%s.%s.jpg" %(save_path, picture_path.split('/')[-1][:-4], count))
                cv2.imwrite("%s%s.%s.jpg" %(save_path, picture_path.split('/')[-1][:-4], count), cut_img)

                count += 1

if __name__ == "__main__":
    data_type = 'train'
    cut_img_mask(data_type) # count_no_mask:163501, count_mask:7299
    cut_img('test')
