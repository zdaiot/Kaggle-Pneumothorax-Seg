import pydicom
import os
import glob
import sys
sys.path.append('..')
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy import misc
import pandas as pd
from utils.mask_functions import rle2mask
import numpy as np
import matplotlib.image as mpimg


def show_dcm_info(dataset, file_path):
    print("Filename.........:", file_path)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Age.......:", dataset.PatientAge)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)
    print("View Position.......:", dataset.ViewPosition)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)


def draw_img_mask():
    img = mpimg.imread('input/train_images/1.2.276.0.7230010.3.1.4.8323329.4440.1517875182.865105.jpg')
    mask = mpimg.imread('./input/train_mask/1.2.276.0.7230010.3.1.4.8323329.307.1517875162.311533.jpg')
    plt.imshow(img, cmap=plt.cm.bone)
    plt.imshow(mask, alpha=0.3, cmap="Reds")
    plt.show()


def test():
    df = pd.read_csv('./input/train-rle.csv', header=None, index_col=0)
    Y_train = np.zeros((1024, 1024, 1))
    for x in df.loc['1.2.276.0.7230010.3.1.4.8323329.307.1517875162.311533',1]:
        # plt.imshow(rle2mask(x, 1024, 1024).T)
        # plt.show()
        Y_train =  Y_train + np.expand_dims(rle2mask(x, 1024, 1024).T, axis=2)
    print(np.shape(Y_train))
    Y_train = np.squeeze(Y_train, 2)
    Y_train = np.where(Y_train>0, 255, 0)
    plt.imshow(Y_train)
    plt.show()

def plot_pixel_array(dataset, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()


def dcm2jpg(dcm_path, save_jpg_path, csv_path=None, save_mask_path=None, show_info_pic=False):
    count_no_mask = 0

    if not os.path.exists(save_jpg_path):
        os.makedirs(save_jpg_path)
    
    # if has .csv
    if csv_path:
        # header=None : Instead of using the first row as a column index, the index is automatically generated
        # index_col=0 : Use the first column as an index column
        df = pd.read_csv(csv_path, header=None, index_col=0)
        if not os.path.exists(save_mask_path):
            os.makedirs(save_mask_path)

    for file_path in glob.glob(dcm_path):
        # convert dcm to jpg
        dataset = pydicom.dcmread(file_path)
        if show_info_pic:
            show_dcm_info(dataset, file_path)
            plot_pixel_array(dataset)
        print(file_path)

        if not csv_path:
            misc.imsave(os.path.join(save_jpg_path, file_path.split('/')[-1][:-4]+'.jpg'), dataset.pixel_array)
        
        # if has .csv, convert csv to mask matrix
        # notice, .loc[row_index, col_index], and if the mask matrix must be transposed
        if csv_path:
            try:         
                # there is no mask
                if '-1' in df.loc[file_path.split('/')[-1][:-4],1] or ' -1' in df.loc[file_path.split('/')[-1][:-4],1]:
                    print('no mask')
                    mask = np.zeros((1024, 1024))   
                else:
                    # if has one mask
                    if type(df.loc[file_path.split('/')[-1][:-4],1]) == str:
                        print('one mask')
                        mask = rle2mask(df.loc[file_path.split('/')[-1][:-4],1], 1024, 1024).T
                        mask = np.where(mask>0, 255, 0)
                    # more than one mask
                    else:
                        print('more than one mask')
                        mask = np.zeros((1024, 1024))
                        for x in df.loc[file_path.split('/')[-1][:-4],1]:
                            mask = mask + rle2mask(x, 1024, 1024).T
                        mask = np.where(mask>0, 255, 0)
                print('save')
                misc.imsave(os.path.join(save_jpg_path, file_path.split('/')[-1][:-4]+'.jpg'), dataset.pixel_array)
                misc.imsave(os.path.join(save_mask_path, file_path.split('/')[-1][:-4]+'.png'), mask)   
            except KeyError:
                print("Key" + file_path.split('/')[-1][:-4] + ",without mask")
                count_no_mask += 1
    print('count_no_mask:',count_no_mask)

def run(): 
    # dcm_path = '../../input/sample images/*.dcm'
    # save_jpg_path = '../../input/sample_images'
    # csv_path = '../../input/sample images/train-rle-sample.csv'
    # save_mask_path = '../../input/sample_mask'
    # dcm2jpg(dcm_path, save_jpg_path, csv_path, save_mask_path)

    # dcm_path = '../../input/dicom-images-train/*/*/*.dcm'
    # save_jpg_path = '../../input/train_images'
    # csv_path = '../../input/train-rle.csv'
    # save_mask_path = '../../input/train_mask'
    # dcm2jpg(dcm_path, save_jpg_path, csv_path, save_mask_path)

    '''there are four pic no be masked, handed del please
    1.2.276.0.7230010.3.1.4.8323329.6491.1517875198.577052
    1.2.276.0.7230010.3.1.4.8323329.7013.1517875202.343274
    1.2.276.0.7230010.3.1.4.8323329.6370.1517875197.841736
    1.2.276.0.7230010.3.1.4.8323329.6082.1517875196.407031
    1.2.276.0.7230010.3.1.4.8323329.7020.1517875202.386064
    '''
    dcm_path = '../input/dicom-images-test/*/*/*.dcm'
    save_jpg_path = '../input/test_images'
    csv_path = '../input/stage_2_train.csv'
    save_mask_path = '../input/test_mask'
    dcm2jpg(dcm_path, save_jpg_path, csv_path, save_mask_path)

    dcm_path = '../input/stage_2_images/*.dcm'
    save_jpg_path = '../input/test_images_stage2'
    dcm2jpg(dcm_path, save_jpg_path)

if __name__ == "__main__":
    run()
    # draw_img_mask()
