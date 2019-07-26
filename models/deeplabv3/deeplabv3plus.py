# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from .sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from .backbone import build_backbone
from .ASPP import ASPP


class DeepLabV3Plus(nn.Module):
	def __init__(self, 
				model_backbone='xception', 
				model_output_stride=16, 
				model_aspp_outdim=256, 
				model_shortcut_dim=48, 
				model_shortcut_kernel=1, 
				num_classes=1,
				bn_mom=0.1
				):
		
		super(DeepLabV3Plus, self).__init__()
		self.backbone = None		
		self.backbone_layers = None
		input_channel = 2048		
		self.aspp = ASPP(dim_in=input_channel, 
				dim_out=model_aspp_outdim, 
				rate=16//model_output_stride,
				bn_mom = bn_mom)
		self.dropout1 = nn.Dropout(0.5)
		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=model_output_stride//4)

		indim = 256
		self.shortcut_conv = nn.Sequential(
				nn.Conv2d(indim, model_shortcut_dim, model_shortcut_kernel, 1, padding=model_shortcut_kernel//2,bias=True),
				SynchronizedBatchNorm2d(model_shortcut_dim, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)		
		self.cat_conv = nn.Sequential(
				nn.Conv2d(model_aspp_outdim+model_shortcut_dim, model_aspp_outdim, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(model_aspp_outdim, momentum=bn_mom),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(model_aspp_outdim, model_aspp_outdim, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(model_aspp_outdim, momentum=bn_mom),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
		)
		self.cls_conv = nn.Conv2d(model_aspp_outdim, num_classes, 1, 1, padding=0)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, SynchronizedBatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		self.backbone = build_backbone(model_backbone, pretrained=False, os=model_output_stride)
		self.backbone_layers = self.backbone.get_layers()

	def forward(self, x):
		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])
		feature_aspp = self.dropout1(feature_aspp)
		feature_aspp = self.upsample_sub(feature_aspp)

		feature_shallow = self.shortcut_conv(layers[0])
		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
		result = self.cat_conv(feature_cat) 
		result = self.cls_conv(result)
		result = self.upsample4(result)
		return result

