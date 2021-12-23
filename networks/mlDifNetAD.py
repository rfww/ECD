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


class mlDifNetAD(nn.Module):

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

        ad1 = abs(dec1 - dec1y)
        ad2 = abs(dec2 - dec2y)
        ad3 = abs(dec3 - dec3y)
        ad4 = abs(dec4 - dec4y)
        ad5 = abs(dec5 - dec5y)

        ad1m = torch.mean(ad1,1)
        ad1m = torch.unsqueeze(ad1m,1)
        #print(ad1m.size())
        ad2m = torch.mean(ad2,1)
        ad2m = torch.unsqueeze(ad2m,1)
        ad3m = torch.mean(ad3,1)
        ad3m = torch.unsqueeze(ad3m,1)
        ad4m = torch.mean(ad4,1)
        ad4m = torch.unsqueeze(ad4m,1)
        ad5m = torch.mean(ad5,1)
        ad5m = torch.unsqueeze(ad5m,1)

        enc1 = F.upsample_bilinear(ad1m, x.size()[2:])
        #print(enc1.size())
        enc2 = F.upsample_bilinear(ad2m, x.size()[2:])
        enc3 = F.upsample_bilinear(ad3m, x.size()[2:])
        enc4 = F.upsample_bilinear(ad4m, x.size()[2:])
        enc5 = F.upsample_bilinear(ad5m, x.size()[2:])
        res = torch.mean(torch.cat([enc1, enc2, enc3, enc4, enc5],1),1)
        res1 = torch.mean(res,2)
        print(res1.size())
        res = res >= torch.mean(torch.mean(res,2),1)
        #print(res.size())




        return res
