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


class mlDifNet_encxy(nn.Module):

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
        self.enc5xy = SegNetEnc(512, 256, 0)
        self.enc4xy = SegNetEnc(768, 128, 0)
        self.enc3xy = SegNetEnc(384, 64, 0)
        self.enc2xy = SegNetEnc(192, 64, 0)
        #self.enc1xy = SegNetEnc(128, 64, 0)
        self.enc1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.enc1xy = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final5 = nn.Conv2d(256, num_classes, 3, padding=1)
        self.final4 = nn.Conv2d(128, num_classes, 3, padding=1)
        self.final3 = nn.Conv2d(64, num_classes, 3, padding=1)
        self.final2 = nn.Conv2d(64, num_classes, 3, padding=1)
        self.final1 = nn.Conv2d(64, num_classes, 3, padding=1)

        #self.final = nn.Conv2d(64, num_classes, 3, padding=1)
        self.fuse = nn.Conv2d(10, num_classes, 3, padding=1)

    def forward(self, x, y):
        '''
            Attention, input size should be the 32x.
        '''
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)
        #enc5 = self.enc5(dec5)

        #enc4 = self.enc4(torch.cat([dec4, enc5], 1))
        #enc3 = self.enc3(torch.cat([dec3, enc4], 1))
        #enc2 = self.enc2(torch.cat([dec2, enc3], 1))
        #enc1 = self.enc1(torch.cat([dec1, enc2], 1))

        dec1y = self.dec1(y)
        dec2y = self.dec2(dec1y)
        dec3y = self.dec3(dec2y)
        dec4y = self.dec4(dec3y)
        dec5y = self.dec5(dec4y)
        #enc5y = self.enc5(dec5y)

        #enc4y = self.enc4(torch.cat([dec4y, enc5y], 1))
        #enc3y = self.enc3(torch.cat([dec3y, enc4y], 1))
        #enc2y = self.enc2(torch.cat([dec2y, enc3y], 1))
        #enc1y = self.enc1(torch.cat([dec1y, enc2y], 1))

        # difference of multiple layers

        enc5xy = self.enc5xy(abs(dec5 - dec5y))  #256
        enc4xy = self.enc4xy(torch.cat([abs(dec4 - dec4y), enc5xy], 1)) #128
        enc3xy = self.enc3xy(torch.cat([abs(dec3 - dec3y), enc4xy], 1)) #64
        enc2xy = self.enc2xy(torch.cat([abs(dec2 - dec2y), enc3xy], 1)) #64
        # print(enc2xy.size())
        # print(enc2xy.size())
        enc1xy = self.enc1xy(torch.cat([abs(dec1 - dec1y), enc2xy], 1)) #64

        #return F.upsample_bilinear(self.final(enc1xy), x.size()[2:])
        enc5xy_res = F.upsample_bilinear(self.final5(enc5xy), x.size()[2:])
        enc4xy_res = F.upsample_bilinear(self.final4(enc4xy), x.size()[2:])
        enc3xy_res = F.upsample_bilinear(self.final3(enc3xy), x.size()[2:])
        enc2xy_res = F.upsample_bilinear(self.final2(enc2xy), x.size()[2:])
        enc1xy_res = F.upsample_bilinear(self.final1(enc1xy), x.size()[2:])
        enc_final = self.fuse(torch.cat([enc5xy_res, enc4xy_res, enc3xy_res, enc2xy_res, enc1xy_res], 1))
        return enc5xy_res, enc4xy_res, enc3xy_res, enc2xy_res, enc1xy_res, enc_final
