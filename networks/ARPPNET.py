import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import os
from torch.utils import model_zoo
from torchvision import models

class SegNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, scale, num_layers):
        super().__init__()

        layers = [
            nn.Upsample(scale_factor=scale, mode='bilinear'),
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


class ARPPNET(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        my_model = models.vgg16(pretrained=True)
        input_1_new = nn.Conv2d(6, 64, (3, 3), 1, 1)
        my_model.features[0] = input_1_new
        decoders = list(my_model.features.children())

        self.dec1 = nn.Sequential(*decoders[:5])
        self.dec2 = nn.Sequential(*decoders[5:10])
        self.dec3 = nn.Sequential(*decoders[10:17])
        self.dec4 = nn.Sequential(*decoders[17:24])
       # self.dec5 = nn.Sequential(*decoders[24:])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.requires_grad = True
        #block1
        self.AR_dec1_1 = nn.Conv2d(64, 16, 3, padding=1, dilation=1)
        self.AR_dec1_3 = nn.Conv2d(64, 16, 3, padding=3, dilation=3)
        self.AR_dec1_5 = nn.Conv2d(64, 16, 3, padding=5, dilation=5)
        self.AR_dec1_7 = nn.Conv2d(64, 16, 3, padding=7, dilation=7)
        self.AR_dec1_conv = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.Conv2d(8, 8, 3, padding=1),
        )
        #block2
        self.AR_dec2_1 = nn.Conv2d(128, 32, 3, padding=1, dilation=1)
        self.AR_dec2_3 = nn.Conv2d(128, 32, 3, padding=3, dilation=3)
        self.AR_dec2_5 = nn.Conv2d(128, 32, 3, padding=5, dilation=5)
        self.AR_dec2_7 = nn.Conv2d(128, 32, 3, padding=7, dilation=7)
        self.AR_dec2_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.Conv2d(16, 16, 3, padding=1),
        )
        #block3
        self.AR_dec3_1 = nn.Conv2d(256, 64, 3, padding=1, dilation=1)
        self.AR_dec3_3 = nn.Conv2d(256, 64, 3, padding=3, dilation=3)
        self.AR_dec3_5 = nn.Conv2d(256, 64, 3, padding=5, dilation=5)
        self.AR_dec3_7 = nn.Conv2d(256, 64, 3, padding=7, dilation=7)
        self.AR_dec3_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.Conv2d(32, 32, 3, padding=1),
        )
        #block4
        self.AR_dec4_1 = nn.Conv2d(512, 128, 3, padding=1, dilation=1)
        self.AR_dec4_3 = nn.Conv2d(512, 128, 3, padding=3, dilation=3)
        self.AR_dec4_5 = nn.Conv2d(512, 128, 3, padding=5, dilation=5)
        self.AR_dec4_7 = nn.Conv2d(512, 128, 3, padding=7, dilation=7)
        self.AR_dec4_conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
        )
        #deconv
        self.enc4 = SegNetEnc(512, 128, 2, 0)
        self.enc3 = SegNetEnc(256, 64, 2, 0)
        self.enc2 = SegNetEnc(128, 32, 2, 0)
        self.enc1 = SegNetEnc(64, 64, 2, 0)

        self.final = nn.Conv2d(64, 2, 3, padding=1)



    def forward(self, x, y):
        '''
            Attention, input size should be the 32x.
        '''
        ################################fusion of im1######################################
        # all node of layer 1
        concat_xy = torch.cat([x,y],1)
        dec1 = self.dec1(concat_xy)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        #dec5 = self.dec5(dec4)

        AR_dec1_1 = self.AR_dec1_1(dec1)
        AR_dec1_1 = self.AR_dec1_conv(AR_dec1_1)
        AR_dec1_3 = self.AR_dec1_3(dec1)
        AR_dec1_3 = self.AR_dec1_conv(AR_dec1_3)
        AR_dec1_5 = self.AR_dec1_5(dec1)
        AR_dec1_5 = self.AR_dec1_conv(AR_dec1_5)
        AR_dec1_7 = self.AR_dec1_7(dec1)
        AR_dec1_7 = self.AR_dec1_conv(AR_dec1_7)
        AR_dec1_cat = torch.cat([AR_dec1_1, AR_dec1_3, AR_dec1_5, AR_dec1_7], 1)

        AR_dec2_1 = self.AR_dec2_1(dec2)
        AR_dec2_1 = self.AR_dec2_conv(AR_dec2_1)
        AR_dec2_3 = self.AR_dec2_3(dec2)
        AR_dec2_3 = self.AR_dec2_conv(AR_dec2_3)
        AR_dec2_5 = self.AR_dec2_5(dec2)
        AR_dec2_5 = self.AR_dec2_conv(AR_dec2_5)
        AR_dec2_7 = self.AR_dec2_7(dec2)
        AR_dec2_7 = self.AR_dec2_conv(AR_dec2_7)
        AR_dec2_cat = torch.cat([AR_dec2_1, AR_dec2_3, AR_dec2_5, AR_dec2_7], 1)

        AR_dec3_1 = self.AR_dec3_1(dec3)
        AR_dec3_1 = self.AR_dec3_conv(AR_dec3_1)
        AR_dec3_3 = self.AR_dec3_3(dec3)
        AR_dec3_3 = self.AR_dec3_conv(AR_dec3_3)
        AR_dec3_5 = self.AR_dec3_5(dec3)
        AR_dec3_5 = self.AR_dec3_conv(AR_dec3_5)
        AR_dec3_7 = self.AR_dec3_7(dec3)
        AR_dec3_7 = self.AR_dec3_conv(AR_dec3_7)
        AR_dec3_cat = torch.cat([AR_dec3_1, AR_dec3_3, AR_dec3_5, AR_dec3_7], 1)

        AR_dec4_1 = self.AR_dec4_1(dec4)
        AR_dec4_1 = self.AR_dec4_conv(AR_dec4_1)
        AR_dec4_3 = self.AR_dec4_3(dec4)
        AR_dec4_3 = self.AR_dec4_conv(AR_dec4_3)
        AR_dec4_5 = self.AR_dec4_5(dec4)
        AR_dec4_5 = self.AR_dec4_conv(AR_dec4_5)
        AR_dec4_7 = self.AR_dec4_7(dec4)
        AR_dec4_7 = self.AR_dec4_conv(AR_dec4_7)
        AR_dec4_cat = torch.cat([AR_dec4_1, AR_dec4_3, AR_dec4_5, AR_dec4_7], 1)

        enc4 = self.enc4(AR_dec4_cat)
        enc3 = self.enc3(torch.cat([AR_dec3_cat, F.upsample_bilinear(enc4, AR_dec3_cat.size()[2:])], 1))
        enc2 = self.enc2(torch.cat([AR_dec2_cat, F.upsample_bilinear(enc3, AR_dec2_cat.size()[2:])], 1))
        enc1 = self.enc1(torch.cat([AR_dec1_cat, F.upsample_bilinear(enc2, AR_dec1_cat.size()[2:])], 1))
        final = F.upsample_bilinear(self.final(enc1), x.size()[2:])

        return final


#model = models.vgg16(pretrained=False)
#print(model)
#model = TFPCD_middle_fusion(2)
#print(model)
