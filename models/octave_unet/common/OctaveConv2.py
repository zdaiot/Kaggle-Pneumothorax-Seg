#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiangtai Li(lxtpku@pku.edu.cn)
# Pytorch Implementation of Octave Conv Operation
# This version use nn.Conv2d because alpha_in always equals alpha_out

import torch
import torch.nn as nn


class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)
        
        X_l2h = self.upsample(X_l2h)
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l


class FirstOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        if self.stride == 2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)
        X_h = x
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)

        return X_h, X_l


class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        X_h, X_l = x

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        
        X_l2h = self.l2h(X_l)
        X_h2h = self.h2h(X_h)
        X_l2h = self.upsample(X_l2h)
        
        X_h = X_h2h + X_l2h

        return X_h


class OctaveCBR(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(OctaveCBR, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_h = norm_layer(int(out_channels*(1-alpha)))
        self.bn_l = norm_layer(int(out_channels*alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.relu(self.bn_h(x_h))
        x_l = self.relu(self.bn_l(x_l))
        return x_h, x_l


class OctaveCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(OctaveCB, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation,
                               groups, bias)
        self.bn_h = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_l = norm_layer(int(out_channels * alpha))

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l)
        return x_h, x_l


class OctaveCR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveCR, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation,
                               groups, bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.relu(x_h)
        x_l = self.relu(x_l)
        return x_h, x_l


class FirstOctaveCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False,norm_layer=nn.BatchNorm2d):
        super(FirstOctaveCBR, self).__init__()
        self.conv = FirstOctaveConv(in_channels, out_channels, kernel_size, alpha,stride,padding,dilation,groups,bias)
        self.bn_h = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_l = norm_layer(int(out_channels * alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.relu(self.bn_h(x_h))
        x_l = self.relu(self.bn_l(x_l))
        return x_h, x_l


class LastOCtaveCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(LastOCtaveCBR, self).__init__()
        self.conv = LastOctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_h = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h = self.conv(x)
        x_h = self.relu(self.bn_h(x_h))
        return x_h


class FirstOctaveCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha=0.5,stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(FirstOctaveCB, self).__init__()
        self.conv = FirstOctaveConv(in_channels,out_channels,kernel_size, alpha,stride,padding,dilation,groups,bias)
        self.bn_h = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_l = norm_layer(int(out_channels * alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l)
        return x_h, x_l


class LastOCtaveCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(LastOCtaveCB, self).__init__()
        self.conv = LastOctaveConv( in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_h = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h = self.conv(x)
        x_h = self.bn_h(x_h)
        return x_h


class FirstOctaveCR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha=0.5,stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveCR, self).__init__()
        self.conv = FirstOctaveConv(in_channels, out_channels, kernel_size, alpha,stride,padding,dilation,groups,bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.relu(x_h)
        x_l = self.relu(x_l)
        return x_h, x_l


class LastOCtaveCR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOCtaveCR, self).__init__()
        self.conv = LastOctaveConv( in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h = self.conv(x)
        x_h = self.relu(x_h)
        return x_h
