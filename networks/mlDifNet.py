import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models

class poolxy(nn.Module):
    def __init__(self,in_channels,out_channels,m):
        super().__init__()
        layers = [
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        ]
        layers +=[
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
                 ]*m
        self.pool = nn.Sequential(*layers)
    def forward(self, x):
        return self.pool(x)
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


class mlDifNet(nn.Module):

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
        self.enc4xy = SegNetEnc(512, 128, 0)
        self.enc3xy = SegNetEnc(256, 64, 0)
        self.enc2xy = SegNetEnc(128, 64, 0)
        # self.enc1xy = SegNetEnc(128, 64, 0)
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
        self.pool1 = poolxy(64, 64, 1)
        self.pool2 = poolxy(64, 64, 1)
        self.pool3 = poolxy(64, 128, 1)
        self.pool4 = poolxy(128, 256, 1)
        self.pool5 = poolxy(256, 512,1 )

        self.rf1024 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.rf512 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.rf128 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=str)
        )
        self.class_xy1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.class_xy2 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.class_xy3 = nn.Sequential(
            nn.Conv2d(128, 128, 10, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.class_xy4 = nn.Sequential(
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.dense = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.dp = nn.Softmax()
        self.ct5 = nn.Conv2d(512, 2, 3, 1, 1)
        self.ct4 = nn.Conv2d(256, 2, 3, 1, 1)
        self.ct3 = nn.Conv2d(128, 2, 3, 1, 1)
        self.final5 = nn.Conv2d(256, num_classes, 3, padding=1)
        self.final4 = nn.Conv2d(128, num_classes, 3, padding=1)
        self.final3 = nn.Conv2d(64, num_classes, 3, padding=1)
        self.final2 = nn.Conv2d(64, num_classes, 3, padding=1)
        self.final1 = nn.Conv2d(64, num_classes, 3, padding=1)

        # self.final = nn.Conv2d(64, num_classes, 3, padding=1)
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
        enc5 = self.enc5(dec5)

        enc4 = self.enc4(torch.cat([dec4, enc5], 1))
        enc3 = self.enc3(torch.cat([dec3, enc4], 1))
        enc2 = self.enc2(torch.cat([dec2, enc3], 1))
        enc1 = self.enc1(torch.cat([dec1, enc2], 1))

        dec1y = self.dec1(y)
        dec2y = self.dec2(dec1y)
        dec3y = self.dec3(dec2y)
        dec4y = self.dec4(dec3y)
        dec5y = self.dec5(dec4y)
        enc5y = self.enc5(dec5y)

        enc4y = self.enc4(torch.cat([dec4y, enc5y], 1))
        enc3y = self.enc3(torch.cat([dec3y, enc4y], 1))
        enc2y = self.enc2(torch.cat([dec2y, enc3y], 1))
        enc1y = self.enc1(torch.cat([dec1y, enc2y], 1))

        # difference of multiple layers

        enc5xy = self.enc5xy(abs(enc5 - enc5y))  #256

        enc4xy = self.enc4xy(torch.cat([abs(enc4 - enc4y), enc5xy], 1)) #128

        enc3xy = self.enc3xy(torch.cat([abs(enc3 - enc3y), enc4xy], 1)) #64

        enc2xy = self.enc2xy(torch.cat([abs(enc2 - enc2y), enc3xy], 1)) #64

        # print(enc2xy.size())
        # print(enc2xy.size())
        enc1xy = self.enc1xy(torch.cat([abs(enc1 - enc1y), enc2xy], 1)) #64

        #return F.upsample_bilinear(self.final(enc1xy), x.size()[2:])
        # enc5xy_res = F.upsample_bilinear(self.final5(enc5xy), x.size()[2:])
        # enc4xy_res = F.upsample_bilinear(self.final4(enc4xy), x.size()[2:])
        # enc3xy_res = F.upsample_bilinear(self.final3(enc3xy), x.size()[2:])
        # enc2xy_res = F.upsample_bilinear(self.final2(enc2xy), x.size()[2:])
        # enc1xy_res = F.upsample_bilinear(self.final1(enc1xy), x.size()[2:])
        # enc_final = self.fuse(torch.cat([enc5xy_res, enc4xy_res, enc3xy_res, enc2xy_res, enc1xy_res], 1))
        p1 = self.pool1(enc1xy)
        p2 = self.pool2(enc2xy)
        p3 = self.pool3(enc3xy)
        p4 = self.pool4(enc4xy)
        p5 = self.pool5(enc5xy)
        # classification
        enc = torch.cat((enc5, enc5y), 1)
        xy1 = self.class_xy1(enc)
        xy2 = self.class_xy2(xy1)
        xy3 = self.class_xy3(xy2)
        res = xy3.view(xy3.size(0), -1)
        xy4 = self.class_xy4(res)
        xy = self.dense(xy4)
        dp = self.dp(xy)

        if dp[0][0].item() > dp[0][1].item():  # the result of classification is x

            f5 = torch.cat([p5, enc5], 1)
            f5 = self.rf1024(f5)  # 1024 -> 512

            f4 = torch.cat([enc4, p4], 1)
            f4 = f4 + f5
            f4 = self.rf512(f4)   # 512-> 256

            f3 = torch.cat([enc3, p3], 1)
            f3 = f3+f4
            f3 = self.rf256(f3)  # 256--->128

            f2 = torch.cat([p2, enc2], 1)
            f2 = f2+f3
            # f2 = self.rf128(f2)  # 128-->64
            f1 = torch.cat([p1, enc1], 1)
            f1 = f2+f1
        else:  # the result of classification is y
            f5 = torch.cat([p5, enc5y], 1)
            f5 = self.rf1024(f5)  # 1024 -> 512

            f4 = torch.cat([enc4y, p4], 1)
            f4 = f4 + f5
            f4 = self.rf512(f4)  # 512-> 256

            f3 = torch.cat([enc3y, p3], 1)
            f3 = f3 + f4
            f3 = self.rf256(f3)  # 256--->128

            f2 = torch.cat([p2, enc2y], 1)
            f2 = f2 + f3
            # f2 = self.rf128(f2)  # 128-->64
            f1 = torch.cat([p1, enc1y], 1)
            f1 = f2 + f1
        xy_5 = self.ct5(f5)
        xy_4 = self.ct4(f4)
        xy_3 = self.ct3(f3)
        xy_2 = self.ct3(f2)
        xy_1 = self.ct3(f1)
        xy_1 = F.upsample_bilinear(xy_1, x.size()[2:])
        xy_2 = F.upsample_bilinear(xy_2, x.size()[2:])
        xy_3 = F.upsample_bilinear(xy_3, x.size()[2:])
        xy_4 = F.upsample_bilinear(xy_4, x.size()[2:])
        xy_5 = F.upsample_bilinear(xy_5, x.size()[2:])
        return xy_1, xy_2, xy_3, xy_4, xy_5
