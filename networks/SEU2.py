import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models


class SegNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()

        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ]
        layers += [
                      nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
                      nn.BatchNorm2d(in_channels // 2),
                      nn.ReLU(inplace=True),
                  ] * num_layers
        layers += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class SEU2(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        decoders = list(models.vgg16(pretrained=True).features.children())

        self.dec1 = nn.Sequential(*decoders[:5])
        self.dec2 = nn.Sequential(*decoders[5:10])
        self.dec3 = nn.Sequential(*decoders[10:17])
        self.dec4 = nn.Sequential(*decoders[17:24])
        self.dec5 = nn.Sequential(*decoders[24:])

        # for m in self.modules():
        #   if isinstance(m, nn.Conv2d):
        #      m.requires_grad = False

        self.enc5 = SegNetEnc(512, 512, 1)
        self.enc4 = SegNetEnc(1024, 256, 1)
        self.enc3 = SegNetEnc(512, 128, 1)
        self.enc2 = SegNetEnc(256, 64, 0)
        self.enc1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.enc5xy = nn.Conv2d(1024, 1024, 3, padding=1)
        self.enc4xy = nn.Conv2d(512, 512, 3, padding=1)
        self.enc3xy = nn.Conv2d(256, 256, 3, padding=1)
        self.enc2xy = nn.Conv2d(128, 128, 3, padding=1)
        self.enc1xy = nn.Conv2d(64, 64, 3, padding=1)

        self.final5 = nn.Conv2d(256, num_classes, 3, padding=1)
        self.final4 = nn.Conv2d(128, num_classes, 3, padding=1)
        self.final = nn.Conv2d(64, num_classes, 3, padding=1)

        #self.final = nn.Conv2d(64, num_classes, 3, padding=1)
        self.fuse = nn.Conv2d(6, num_classes, 3, padding=1)

    def forward(self, x, y):
        '''
            Attention, input size should be the 32x.
        '''
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)

        dec1y = self.dec1(y)
        dec2y = self.dec2(dec1y)
        dec3y = self.dec3(dec2y)
        dec4y = self.dec4(dec3y)
        dec5y = self.dec5(dec4y)

        enc5 = self.enc5(dec5) #512
       # print(enc5.size())
       # print((dec4 - dec4y).size())
        enc5xy = self.enc5xy(torch.cat([(dec4 - dec4y), enc5], 1))

        enc4 = self.enc4(enc5xy)
        enc4xy = self.enc4xy(torch.cat([(dec3 - dec3y), enc4], 1))

        enc3 = self.enc3(enc4xy)
        enc3xy = self.enc3xy(torch.cat([(dec2 - dec2y), enc3], 1))

        enc2 = self.enc2(enc3xy)
        enc2xy = self.enc2xy(torch.cat([(dec1 - dec1y), enc2], 1))

        enc1 = self.enc1(enc2xy)
        final = self.final(enc1)
        enc_final = F.upsample_bilinear(final, x.size()[2:])
        return enc_final



