import torch.nn as nn
from .OctaveConv2 import *


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, **batchnorm_params):

        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=not (use_batchnorm)),
            nn.ReLU(inplace=True),
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class OctaveConv2ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True):
        super(OctaveConv2ReLU, self).__init__()

        if use_batchnorm:
            self.block = OctaveCBR(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   alpha=0.5, stride=stride, padding=padding, bias=False)
        else:
            self.block = OctaveCR(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                  alpha=0.5, bias=True)

    def forward(self, x):
        return self.block(x)


class FirstOctaveConv2ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True):
        super(FirstOctaveConv2ReLU, self).__init__()

        if use_batchnorm:
            self.block = FirstOctaveCBR(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, alpha=0.5, bias=False)
        else:
            self.block = FirstOctaveCR(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                       padding=padding, alpha=0.5, bias=True)

    def forward(self, x):
        return self.block(x)


class LastOctaveConv2ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True):
        super(LastOctaveConv2ReLU, self).__init__()
        if use_batchnorm:
            self.block = LastOCtaveCBR(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                       padding=padding, alpha=0.5, bias=False)
        else:
            self.block = LastOCtaveCR(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, alpha=0.5, bias=True)

    def forward(self, x):
        return self.block(x)
