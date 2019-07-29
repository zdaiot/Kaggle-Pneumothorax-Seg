import numpy as np
from pprint import pprint
import os
from datetime import datetime

def write_txt(path, content):
    # TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now()) 
    f = open(os.path.join(path, 'log.txt'), 'a')
    if isinstance(content, dict) or isinstance(content, list):
        pprint(content, f)
    else:
        f.write(content + '\n')
    f.flush()
    f.close()

def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255: # 预测目标置为255的时候使用
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return " ".join(rle)


# Code Credits: https://www.kaggle.com/abhishek/inference-for-mask-rcnn
def mask_to_rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 1: # 预测目标置为1的时候使用
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel+=1
    return " " + " ".join(rle) # 若rle为[]，则" ".join(rle)结果为''


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position + lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)
