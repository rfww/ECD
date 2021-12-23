import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models


class SegNetEnc(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):  # 512*512*1
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


class SegNetEnc11(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor, num_layers):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
        ]
        layers += [
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ] * num_layers
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class SegNetEnc2(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor, num_layers):
        super().__init__()

        layers = [
            nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ] * num_layers

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class SegNetEnc3(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor, num_layers):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class cross_AD_enhancement(nn.Module):
    def __init__(self):
        super(cross_AD_enhancement,self).__init__()
        self.ADHF2ADLF = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),  # 128->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.ADLF2ADHF = nn.Sequential(
            nn.Conv2d(128,128,3,padding = 1), #128->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.adhl = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1),#256->64
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))

        #self.adlfrf = nn.Sequential(nn.Conv2d(576, 64, 3, padding=1),
         ##                         nn.BatchNorm2d(64),
           #                       nn.ReLU(inplace=True))

        self.adhl_prediction = nn.Conv2d(64, 2, 3, padding=1)#64

        self.hf_prediction = nn.Conv2d(128,2,3,padding=1)#128
        self.lf_prediction = nn.Conv2d(128,2,3,padding=1)

        #initialize_weights(self.enc5, self.enc4, self.enc3, self.enc2, self.enc1, self.hf_af, self.hf_prediction,
        #                   self.hf_xy, self.lfe, self.l_xy_hf_af, self.edge, self.attention, self.final_prediction)

    def forward(self,hft,lft):

        hf_prediction = self.hf_prediction(hft)
        lf_prediction = self.lf_prediction(lft)
        #print(ad_hf.size())
        #print(ad_lf.size())

        adhl = self.adhl(torch.cat([F.upsample_bilinear(hft, lft.size()[2:]), lft], 1))

        adhl_prediction = self.adhl_prediction(adhl)

        ad_hf2lf = self.ADHF2ADLF(F.upsample_bilinear(hft, lft.size()[2:]))
        ad_lf2hf = self.ADLF2ADHF(F.upsample_bilinear(lft, hft.size()[2:]))

        hft1 = hft.mul(ad_lf2hf)
        lft1 = lft.mul(ad_hf2lf)

        return hf_prediction,lf_prediction, adhl_prediction, hft1,lft1


class loadp(nn.Module):
    def __init__(self, backbone='vgg16', pretrained=True, freeze_backbone=False):
        super(loadp, self).__init__()

        vgg = models.vgg16(pretrained)
        features = list(vgg.features.children())
        self.dec1 = nn.Sequential(*features[:5])  # 160
        self.dec2 = nn.Sequential(*features[5:10])  # 80
        self.dec3 = nn.Sequential(*features[10:17])  # 40
        self.dec4 = nn.Sequential(*features[17:24])  # 20
        self.dec5 = nn.Sequential(*features[24:])  # 10

        self.cross_AD_enhancement = cross_AD_enhancement()
        self.enc5 = SegNetEnc(512, 512, 1)  # 20
        self.enc4 = SegNetEnc(1024, 256, 1)  # 40
        self.enc3 = SegNetEnc(512, 128, 1)  # 80
        self.enc2 = SegNetEnc(256, 128, 1)  # 160
        # self.enc1 = SegNetEnc(192, 64, 1)  # 160
        self.enc1 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(192, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # 160
        # self.enc1 = SegNetEnc3(192, 64,1,1) #320
        self.enc5c = SegNetEnc2(512, 256, 2, 1)  # 40
        self.enc4c = SegNetEnc2(256, 128, 2, 1)  # 80
        self.enc3c = SegNetEnc2(128, 64, 2, 1)  # 160
        self.enc2c = SegNetEnc3(128, 64, 1, 1)  # 320
        self.enc5xy = SegNetEnc(512, 256, 1)
        self.enc4xy = SegNetEnc(512, 256, 1)
        self.enc3xy = SegNetEnc(384, 128, 1)
        self.enc2xy = SegNetEnc2(256, 128, 1, 1)
        self.enc1xy = SegNetEnc2(192, 128, 1, 1)  # 40

        # self.hfc = SegNetEnc3(512, 128, 2, 1)#40   384->128
        # self.lfc = SegNetEnc3(384, 128, 2, 1)#40   192->128
        self.p5 = nn.Conv2d(256, 2, 3, padding=1)  # 320
        self.p4 = nn.Conv2d(256, 2, 3, padding=1)
        self.p3 = nn.Conv2d(128, 2, 3, padding=1)
        self.p2 = nn.Conv2d(128, 2, 3, padding=1)
        self.p1 = nn.Conv2d(128, 2, 3, padding=1)
        self.p0 = nn.Conv2d(64, 2, 3, padding=1)
        # change x or y's shape to 4*2*320*320
        # self.change1 = nn.Conv2d(64, 2, 3, 1, 1)
        # self.change23 = nn.Conv2d(128, 2, 3, 1, 1)
        # self.change4 = nn.Conv2d(256, 2, 3, 1, 1)
        # self.change5 = nn.Conv2d(512, 2, 3, 1, 1)

        # After improving:
        # self.rf514 = nn.Conv2d(514, 256, 3, padding=1, stride=1)
        # self.rf386 = nn.Conv2d(386, 128, 3, padding=1, stride=1)
        # self.rf258 = nn.Conv2d(258, 64, 3, padding=1, stride=1)
        # self.rf130 = nn.Conv2d(130, 2, 3, padding=1, stride=1)
        # self.fcat  = nn.Conv2d(2,1,3,1,1)

        # self.add512 = nn.Conv2d(512,2,3,1,1)
        # self.add256 = nn.Conv2d(256,2,3,1,1)
        # self.add128 = nn.Conv2d(128,2,3,1,1)
        # self.add64  = nn.Conv2d(64,2,3,1,1)
        # initialize_weights(self.enc5, self.enc4, self.enc3,
        # self.enc2, self.enc1, self.enc5c,self.enc4c, self.enc3c, self.enc2c)
        # self.lf_af = SegNetEnc2(64, 64, 2, 1)#320
        # self.lf_prediction = nn.Conv2d(64, 2, 3, padding=1)#320
        # self.hf_af = SegNetEnc2(192, 64, 2, 1)#320
        # self.hf_prediction = nn.Conv2d(64, 2, 3, padding=1)
        # self.final_prediction = nn.Conv2d(16,2,3,padding = 1)
        # self.hfp = nn.Conv2d(128, 2, 3, padding=1, stride=1)

      
        

    def forward(self, x, y):
        x_f1 = self.dec1(x)
        x_f2 = self.dec2(x_f1)
        x_f3 = self.dec3(x_f2)
        x_f4 = self.dec4(x_f3)
        x_f5 = self.dec5(x_f4)
        x_enc5 = self.enc5(x_f5)
        x_enc4 = self.enc4(torch.cat([x_f4, x_enc5], 1))
        x_enc3 = self.enc3(torch.cat([x_f3, x_enc4], 1))
        x_enc2 = self.enc2(torch.cat([x_f2, x_enc3], 1))
        x_enc1 = self.enc1(torch.cat([x_f1, x_enc2], 1))

        y_f1 = self.dec1(y)
        y_f2 = self.dec2(y_f1)
        y_f3 = self.dec3(y_f2)
        y_f4 = self.dec4(y_f3)
        y_f5 = self.dec5(y_f4)

        y_enc5 = self.enc5(y_f5)  #
        y_enc4 = self.enc4(torch.cat([y_f4, y_enc5], 1))  #
        y_enc3 = self.enc3(torch.cat([y_f3, y_enc4], 1))  #
        y_enc2 = self.enc2(torch.cat([y_f2, y_enc3], 1))  #
        y_enc1 = self.enc1(torch.cat([y_f1, y_enc2], 1))  #

        enc5xy = self.enc5xy(abs(x_enc5 - y_enc5))
        enc4xy = self.enc4xy(torch.cat([abs(x_enc4 - y_enc4), enc5xy], 1))
        enc3xy = self.enc3xy(torch.cat([abs(x_enc3 - y_enc3), enc4xy], 1))
        enc2xy = self.enc2xy(torch.cat([abs(x_enc2 - y_enc2), enc3xy], 1))
        enc1xy = self.enc1xy(torch.cat([abs(x_enc1 - y_enc1), enc2xy], 1))

        p5 = self.p5(enc5xy)
        p4 = self.p4(enc4xy)
        p3 = self.p3(enc3xy)
        p2 = self.p2(enc2xy)
        p1 = self.p1(enc1xy)

        return p1, p2, p3, p4, p5

        # add_x5 = self.add512(x_enc5)
        # add_x4 = self.add256(x_enc4)
        # add_x3 = self.add128(x_enc3)
        # add_x2 = self.add128(x_enc2)
        # add_x1 = self.add64(x_enc1)

        # add_x5 = F.upsample_bilinear(add_x5,p5.size()[2:])
        # add_x4 = F.upsample_bilinear(add_x4,p4.size()[2:])
        # add_x3 = F.upsample_bilinear(add_x3,p3.size()[2:])
        # # add_x2 = F.upsample_bilinear(add_x2,p2.size()[2:])
        # # add_x1 = F.upsample_bilinear(add_x5,p1.size()[2:])

        # ff5 = add_x5 + p5
        # ff4 = add_x4 + p4
        # ff3 = add_x3 + p3
        # ff2 = add_x2 + p2
        # ff1 = add_x1 + p1
        # upsampling to the target
        # ff1 = F.upsample_bilinear(ff1,x.size()[2:])
        # ff2 = F.upsample_bilinear(ff2,x.size()[2:])
        # ff3 = F.upsample_bilinear(ff3,x.size()[2:])
        # ff4 = F.upsample_bilinear(ff4,x.size()[2:])
        # ff5 = F.upsample_bilinear(ff5,x.size()[2:])

        # return ff1,ff2,ff3,ff4,ff5,dp
        # hf = torch.cat([F.upsample_bilinear(enc5xy, enc4xy.size()[2:]), enc4xy],1)  # 160
        # hf = self.hfc(hf)

        # lf = torch.cat([F.upsample_bilinear(enc3xy, enc2xy.size()[2:]), enc2xy, enc1xy], 1)#160
        # lf = self.lfc(lf)

        # hf_prediction = []
        # lf_prediction= []
        # adhl_prediction = []

        # for i in range(3):
        #     [hf_p,lf_p,ad_p, hft,lft] = self.cross_AD_enhancement(hf,lf)
        #     hf_prediction.append(F.upsample_bilinear(hf_p,x.size()[2:]))
        #     lf_prediction.append(F.upsample_bilinear(lf_p,x.size()[2:]))
        #     adhl_prediction.append(F.upsample_bilinear(ad_p,x.size()[2:]))

        # x_enc1 = self.change1(x_enc1)
        # x_enc2 = self.change23(x_enc2)
        # x_enc3 = self.change23(x_enc3)
        # x_enc4 = self.change4(x_enc4)
        # x_enc5 = self.change5(x_enc5)

        # y_enc1 = self.change1(y_enc1)
        # y_enc2 = self.change23(y_enc2)
        # y_enc3 = self.change23(y_enc3)
        # y_enc4 = self.change4(y_enc4)
        # y_enc5 = self.change5(y_enc5)

        # p1 = F.upsample_bilinear(p1, x.size()[2:])
        # p2 = F.upsample_bilinear(p2, x.size()[2:])
        # p3 = F.upsample_bilinear(p3, x.size()[2:])
        # p4 = F.upsample_bilinear(p4, x.size()[2:])
        # p5 = F.upsample_bilinear(p5, x.size()[2:])

        # x1 = F.upsample_bilinear(x_enc1,x.size()[2:])
        # x2 = F.upsample_bilinear(x_enc2,x.size()[2:])
        # x3 = F.upsample_bilinear(x_enc3,x.size()[2:])
        # x4 = F.upsample_bilinear(x_enc4,x.size()[2:])
        # x5 = F.upsample_bilinear(x_enc5,x.size()[2:])

        # y1 = F.upsample_bilinear(y_enc1,x.size()[2:])
        # y2 = F.upsample_bilinear(y_enc2,x.size()[2:])
        # y3 = F.upsample_bilinear(y_enc3,x.size()[2:])
        # y4 = F.upsample_bilinear(y_enc4,x.size()[2:])
        # y5 = F.upsample_bilinear(y_enc5,x.size()[2:])

        # x = [x1, x2, x3, x4, x5]
        # y = [y1, y2, y3, y4, y5]
        # p = [p1, p2, p3, p4, p5]
        # hf_p = F.upsample_bilinear(hf_p, x.size()[2:])
        # lf_p = F.upsample_bilinear(lf_p, x.size()[2:])
        # ad_p = F.upsample_bilinear(ad_p, x.size()[2:])
        # fp = self.final_prediction(torch.cat([p1,p2,p3,p4,p5,hf_p,lf_p,ad_p],1))
        # class_dp  = self.class_hf(hf)
        # class_dp1 = self.class_hf1(class_dp)
        # class_dp2 = self.class_hf2(class_dp1)
        # class_dp3 = self.class_hf3(class_dp2)

        # res = class_dp3.view(class_dp3.size(0),-1)
        # dpp = self.dense(res)
        # dp = self.dp(dpp)

        # return p, x, y, dp




