import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import Conv2dReLU, OctaveConv2ReLU, FirstOctaveConv2ReLU, LastOctaveConv2ReLU
from ..base.model import Model


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class OctaveInterpolate(nn.Module):
    """对八角卷积的输出进行上采样，需要对高频和低频部分分开进行上采样
    """
    def __init__(self, scale_factor=2, mode='nearest'):
        super(OctaveInterpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x_h, x_l = x
        x_h = F.interpolate(x_h, scale_factor=self.scale_factor, mode=self.mode)
        x_l = F.interpolate(x_l, scale_factor=self.scale_factor, mode=self.mode)
        return x_h, x_l


class CatSkipOctave(nn.Module):
    """将跳层特征图和八角卷积的输出在指定维度进行拼接，将跳层特征图转换为高频和低频两部分，再和八角卷积的输出进行拼接
    """
    def __init__(self, skip_channels, dim=1, alpha=0.5):
        """
        :param skip_channels: 跳级特征的通道数
        :param dim: 拼接维度
        :param alpha: 低频的通道数比例
        :return: 无
        """
        super(CatSkipOctave, self).__init__()
        self.dim = dim
        self.alpha = alpha
        # 将skip转换为高频，只需使用1x1卷积改变通道数
        self.h_conv2d = nn.Conv2d(skip_channels, skip_channels-int(skip_channels * self.alpha), kernel_size=1)
        # 将skip转换为低频，首先使用平均池化改变空间大小，再使用1x1卷积改变通道数
        self.l_avgpool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.l_conv2d = nn.Conv2d(skip_channels, int(skip_channels * self.alpha), kernel_size=1)

    def forward(self, x, skip):
        """
        :param x: 八角特征
        :param skip: 跳层特征
        :return: cat_h, cat_l
        """
        # 将skip转换为高频和低频
        skip_h = self.h_conv2d(skip)
        skip_l = self.l_avgpool(skip)
        skip_l = self.l_conv2d(skip_l)

        x_h, x_l = x
        cat_h = torch.cat([x_h, skip_h], dim=self.dim)
        cat_l = torch.cat([x_l, skip_l], dim=self.dim)

        return cat_h, cat_l


class FirstOctaveDecoderBlock(nn.Module):
    """第一层八角解码模块，第一层的输入x未进行频率划分，因而需要对其进行频率划分后，再进行八角卷积
    """
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super(FirstOctaveDecoderBlock, self).__init__()
        self.block = nn.Sequential(
            FirstOctaveConv2ReLU(in_channels, out_channels, kernel_size=(3, 3), padding=1, use_batchnorm=use_batchnorm),
            OctaveConv2ReLU(out_channels, out_channels, kernel_size=(3, 3), padding=1, use_batchnorm=use_batchnorm)
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class OctaveDecoderBlock(nn.Module):
    """中间层八角卷积解码模块，高频、低频输入，高频低频输出
    """
    def __init__(self, in_channels, out_channels, skip_channels, use_batchnorm=True):
        super(OctaveDecoderBlock, self).__init__()
        self.block = nn.Sequential(
            OctaveConv2ReLU(in_channels, out_channels, kernel_size=(3, 3), padding=1, use_batchnorm=use_batchnorm),
            OctaveConv2ReLU(out_channels, out_channels, kernel_size=(3, 3), padding=1, use_batchnorm=use_batchnorm),
        )
        self.interpolate = OctaveInterpolate(scale_factor=2, mode='nearest')
        self.cat = CatSkipOctave(skip_channels, dim=1, alpha=0.5)

    def forward(self, x):
        x, skip = x
        x = self.interpolate(x)
        if skip is not None:
            x = self.cat(x, skip)
        x = self.block(x)
        return x


class LastOctaveDecoderBlock(nn.Module):
    """最后一层八角卷积解码模块，高频、低频输入，合并输出
    """
    def __init__(self, in_channels, out_channels, skip_channels=None, use_batchnorm=True):
        super(LastOctaveDecoderBlock, self).__init__()
        self.block = nn.Sequential(
            OctaveConv2ReLU(in_channels, out_channels, kernel_size=(3, 3), padding=1, use_batchnorm=use_batchnorm),
            LastOctaveConv2ReLU(out_channels, out_channels, kernel_size=(3, 3), padding=1, use_batchnorm=use_batchnorm),
        )
        self.interpolate = OctaveInterpolate(scale_factor=2, mode='nearest')
        if skip_channels:
            self.cat = CatSkipOctave(skip_channels, dim=1, alpha=0.5)

    def forward(self, x):
        x, skip = x
        x = self.interpolate(x)
        if skip is not None:
            x = self.cat(x, skip)

        x = self.block(x)
        return x


class CenterBlock(FirstOctaveDecoderBlock):

    def forward(self, x):
        return self.block(x)


class UnetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=False,
    ):
        super().__init__()
        # 计算各个解码块的输入和输出通道数
        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        # 如果存在centerblock，则layer1为普通的八角卷积，否则是第一八角卷积
        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
            self.layer1 = OctaveDecoderBlock(in_channels[0], out_channels[0], encoder_channels[0], use_batchnorm=use_batchnorm)
        else:
            self.center = None
            self.layer1 = FirstOctaveDecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)

        self.layer2 = OctaveDecoderBlock(in_channels[1], out_channels[1], encoder_channels[2], use_batchnorm=use_batchnorm)
        self.layer3 = OctaveDecoderBlock(in_channels[2], out_channels[2], encoder_channels[3], use_batchnorm=use_batchnorm)
        self.layer4 = OctaveDecoderBlock(in_channels[3], out_channels[3], encoder_channels[4], use_batchnorm=use_batchnorm)
        self.layer5 = LastOctaveDecoderBlock(in_channels[4], out_channels[4], skip_channels=None, use_batchnorm=use_batchnorm)
        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        """
        依据编码器各输出特征图的通道和解码器各输出特征图的通道数计算解码器的输入通道数（跳级连接通道数＋上一级的输出特征图的通道数）
        :param encoder_channels:
        :param decoder_channels:
        :return: channels: 各解码器的输入通道数
        """
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)

        return x
