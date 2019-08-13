from .unets import Unet
import torch.nn as nn


class UnetWithClassify(nn.Module):

    def __init__(self, encoder_name, classes):
        super(UnetWithClassify, self).__init__()
        self.classes = classes
        self.encoder_name = encoder_name
        self.encoder, self.decoder = self.get_encoder_decoder()

        # 分类分支
        self.classifier_ConvBN = nn.Sequential(
            nn.Conv2d(512, 256, 3, 2),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 64, 3, 2), 
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(64, classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        # 选取多尺度特征中最抽象的特征进行分类
        feature = x[0]
        x = self.decoder(x)
        class_feature = self.classifier_ConvBN(feature)
        class_feature = class_feature.mean([2, 3])
        class_feature = self.classifier(class_feature)

        return class_feature, x

    def get_encoder_decoder(self):
        unet = Unet(self.encoder_name, classes=self.classes, activation=None)
        encoder = unet.encoder
        decoder = unet.decoder

        return encoder, decoder