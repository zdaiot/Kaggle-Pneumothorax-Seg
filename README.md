code for [kaggle siim-acr-pneumothorax-segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)

## Requirements
* Install [backboned-unet](https://github.com/mkisantal/backboned-unet) first
```
git clone https://github.com/mkisantal/backboned-unet.git
cd backboned-unet
pip install .
```
* Install image augumentation library [albumentations](https://github.com/albu/albumentations)
```
conda install -c conda-forge imgaug
conda install albumentations -c albumentations
```

## TODO
- [x] unet_resnet34
- [x] data augmentation
- [x] two stage set: two stage batch size and two stages epoch
- [x] epoch freezes the encoder layer in the first stage
- [x] epoch gradients accumulate in the second stage
- [x] adapt to torchvison0.2.0
- [ ] cross validation
- [ ] leak

## Dataset
Creat dataset soft links in the following directories.
```
ln -s ../../../input/train_images/ train_images
ln -s ../../../input/train_mask/ train_mask
ln -s ../../../input/test_images/ test_images
ln -s ../../../input/sample_mask sample_mask
ln -s ../../../input/sample_images/ sample_images
ln -s ../../../input/train-rle.csv train-rle.csv
```

## Results
|backbone|batch_size|image_size|pretrained|data proprecess|lr|score|
|--|--|--|--|--|--|--|
|U-Net|32|224|w/o|w/o|random|0.7019|
|ResNet34|32|224|w/|w/o|random|0.7172|
|ResNet34|32|224|w/o|w/o|random|0.7295|
|ResNet34|20|512|w/|w/o|random|0.7508|
|ResNet34|20|512|w/|w/|random|0.7603|