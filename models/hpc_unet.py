from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import EncoderDecoder
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn
import torch.nn.functional as F
import torch


class HyperColumnUnet(nn.Module):
    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=1,
            activation='sigmoid',
            center=False,  # usefull for VGG models
    ):
        super(HyperColumnUnet, self).__init__()
        self.encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_shapes,
            decoder_channels=decoder_channels,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
            center=center,
        )
        self.center = center

        self.decoder1 = self.decoder.layer1
        self.decoder2 = self.decoder.layer2
        self.decoder3 = self.decoder.layer3
        self.decoder4 = self.decoder.layer4
        self.decoder5 = self.decoder.layer5

        self.logit = nn.Sequential(nn.Conv2d(496, 32, kernel_size=3, padding=1),
                            nn.ELU(True),
                            nn.Conv2d(32, 1, kernel_size=1, bias=False))

        self.name = 'u-{}'.format(encoder_name)

    def forward(self, x):
        x = self.encoder(x)
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        d1 = self.decoder1([encoder_head, skips[0]])
        d2 = self.decoder2([d1, skips[1]])
        d3 = self.decoder3([d2, skips[2]])
        d4 = self.decoder4([d3, skips[3]])
        d5 = self.decoder5([d4, None])
        
        dcat = torch.cat((
            d5, 
            F.upsample(d4, scale_factor=2, mode='bilinear', align_corners=True),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=True),
            F.upsample(d2, scale_factor=8, mode='bilinear', align_corners=True),
            F.upsample(d1, scale_factor=16, mode='bilinear', align_corners=True)), 1)

        dcat = F.dropout2d(dcat, p=0.4)
        logit = self.logit(dcat)

        return logit

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function (if activation is not `None`) with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x)

        return x