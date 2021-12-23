import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision import models
# from utils.spp_layer import spatial_pyramid_pool

class SegNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor,num_layers):
        super().__init__()

        layers = [
            nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
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

    def __init__(self, in_channels, out_channels):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, in_channels*2, 3, padding=1,stride=2),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(in_channels*2, in_channels*4, 3, padding=1,stride=2),
            nn.BatchNorm2d(in_channels*4),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(in_channels*4, in_channels*8, 3, padding=1,stride=2),
            nn.BatchNorm2d(in_channels*8),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class SegNetEnc2(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, in_channels * 2, 1, padding=0),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            
        ]
        layers += [
            nn.Conv2d(in_channels * 2, in_channels * 2, 3, padding=1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
        ]
        # layers += [
        #     nn.Conv2d(in_channels * 4, out_channels,kernel_size=3,padding=1,stride=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class feature_extractor(nn.Module):
    def __init__(self, backbone='vgg16_bn', pretrained=True, freeze_backbone=False):
        super(feature_extractor, self).__init__()
        self.output_num = [4,2,1]
        vgg = models.vgg16_bn(pretrained)
        features = list(vgg.features.children())
        # self.dec1 = nn.Sequential(*features[:5])  # 160
        self.dec2 = nn.Sequential(*features[7:14])  # 80
        self.dec3 = nn.Sequential(*features[14:24])  # 40
        self.dec4 = nn.Sequential(*features[24:34])  # 20
        self.dec5 = nn.Sequential(*features[34:44])  # 10
     
        #self.encb = SegNetEnc2(64, 512, 3)#20
        self.enc5 = SegNetEnc(512, 512, 2,1)#20
        self.enc4 = SegNetEnc(1024, 512, 2,1)#40
        self.enc3 = SegNetEnc(768, 256, 2,1)#80
        self.enc2 = SegNetEnc(384, 128, 2,1)#160
        self.enc1 = SegNetEnc(192, 64, 1,1)  # 160
        # self.enc1 = nn.Sequential(
        #     #nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(192, 128, 3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        # )#160
    
        # self.sgp= SegNetEnc11(64,512)
        self.sgp= SegNetEnc(896,512,1,1)
        self.p5 = nn.Conv2d(256,2,3,1,1)
        self.p4 = nn.Conv2d(128,2,3,1,1)
        self.p3 = nn.Conv2d(128,2,3,1,1)
        self.p2 = nn.Conv2d(64,2,3,1,1)
        self.p1 = nn.Conv2d(64,2,3,1,1)
        self.q5 = nn.Conv2d(512,2,3,1,1)
        self.q4 = nn.Conv2d(256,2,3,1,1)
        self.q3 = nn.Conv2d(128,2,3,1,1)
        self.q2 = nn.Conv2d(128,2,3,1,1)
        self.q1 = nn.Conv2d(64,2,3,1,1)
        self.xp5 = nn.Conv2d(512,2,3,1,1)
        self.xp4 = nn.Conv2d(256,2,3,1,1)
        self.xp3 = nn.Conv2d(128,2,3,1,1)
        self.xp2 = nn.Conv2d(128,2,3,1,1)
        self.xp1 = nn.Conv2d(64,2,3,1,1)

        # self.sp1 = nn.Conv2d(64,2,3,1,1)
        self.sp2 = nn.Conv2d(128,2,3,1,1)
        self.sp3 = nn.Conv2d(128,2,3,1,1)
        self.sp4 = nn.Conv2d(256,2,3,1,1)
        self.sp5 = nn.Conv2d(512,2,3,1,1)
        self.mmp1 = nn.Conv2d(64,2,3,1,1)
        self.mmp2 = nn.Conv2d(128,2,3,1,1)
        self.mmp3 = nn.Conv2d(128,2,3,1,1)
        self.mmp4 = nn.Conv2d(256,2,3,1,1)

        self.mp = nn.Conv2d(512,2,3,1,1)
        self.pf =nn.Sequential(nn.Conv2d(12,2,3,1,1),nn.BatchNorm2d(2),nn.ReLU(inplace=True))


        self.final1 = nn.Sequential(
            nn.Conv2d(130,64,3,1,1),
            # nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.final2 = nn.Sequential(
            nn.Conv2d(258,64,3,1,1),
            # nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.final3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(258,128,3,1,1),
            # nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.final4 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(514,128,3,1,1),
            # nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.final5 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(514,256,3,1,1),
            # nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.s1 = nn.Sequential(
            nn.Conv2d(192,64,3,padding=1,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.s2 = nn.Sequential(
            nn.Conv2d(192,128,3,padding=1,stride=1),
            # nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.s3 = nn.Sequential(
            nn.Conv2d(256,128,3,padding=1,stride=1),
            # nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.s4 = nn.Sequential(
            nn.Conv2d(384,256,3,padding=1,stride=1),
            # nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.s5 = nn.Sequential(
            nn.Conv2d(768,512,3,padding=1,stride=1),
            # nn.MaxPool2d(2,2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(512,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(256,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(256,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.m1 = nn.Sequential(
            nn.Conv2d(128,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mp4 = nn.Sequential(
            nn.Conv2d(512,256,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.mp3 = nn.Sequential(
            nn.Conv2d(512,128,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.mp2 = nn.Sequential(
            nn.Conv2d(512,128,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.mp1 = nn.Sequential(
            nn.Conv2d(512,64,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.seg1 = SegNetEnc(128,64,1,1)
        self.seg2 = SegNetEnc(256,128,1,1)
        self.seg3 = SegNetEnc(384,128,1,1)
        self.seg4 = SegNetEnc(768,256,1,1)
        self.seg5 = SegNetEnc(1024,512,1,1)
        self.cat4 = nn.Sequential(
            nn.Conv2d(768,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.cat3 = nn.Sequential(
            nn.Conv2d(384,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.cat2 = nn.Sequential(
            nn.Conv2d(256,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.encc1 = SegNetEnc2(256)
        self.encc2 = SegNetEnc2(512)
        self.encc3 = SegNetEnc2(1024)
        self.encn1 = SegNetEnc2(256)
        self.encn2 = SegNetEnc2(512)
        self.encn3 = SegNetEnc2(1024)

        self.pr1 = nn.Sequential(
            nn.Conv2d(512,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pr2 = nn.Sequential(
            nn.Conv2d(1024,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pr3 = nn.Sequential(
            nn.Conv2d(2048,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.x1 = nn.Sequential(
            nn.Conv2d(66,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.x2 = nn.Sequential(
            nn.Conv2d(258,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.x3 = nn.Sequential(
            nn.Conv2d(258,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.x4 = nn.Sequential(
            nn.Conv2d(514,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.x5 = nn.Sequential(
            nn.Conv2d(1026,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, y,dp):
        if dp[0][0].item()>dp[0][1].item():
            c_f1 = x
            n_f1 = y
        else:
            c_f1 = y
            n_f1 = x
        # x_f1 = self.dec1(x)
        

        n_f2 = self.dec2(n_f1)
        n_f3 = self.dec3(n_f2)

        c_f2 = self.dec2(c_f1)
        c_f3 = self.dec3(c_f2)
        c_f4 = self.dec4(c_f3)
        c_f5 = self.dec5(c_f4)

        c2 = self.encc1(c_f3)
        c3 = self.encc2(c2)
        c4 = self.encc3(c3)
        n2 = self.encn1(n_f3)
        n3 = self.encn2(n2)
        n4 = self.encn3(n3)
        pr1 = self.pr1(abs(c2-n2))
        pr2 = self.pr2(abs(c3-n3))
        pr3 = self.pr3(abs(c4-n4))
        # print(pr1.shape)
        # print(pr2.shape)
        # print(pr3.shape)
        pr1 = F.upsample_bilinear(pr1, scale_factor=2)
        ccp = self.sgp(torch.cat([
            pr1,
            F.upsample_bilinear(pr2, scale_factor=4),
            F.upsample_bilinear(pr3, pr1.size()[2:])
        ],1))
        # ccp = self.sgp(abs(c_f1-n_f1))
        # mp = self.mp(mp)
        c_enc5 = self.enc5(c_f5)
        c_enc4 = self.enc4(torch.cat([c_f4, c_enc5], 1))
        c_enc3 = self.enc3(torch.cat([c_f3, c_enc4], 1))
        c_enc2 = self.enc2(torch.cat([c_f2, c_enc3], 1))
        c_enc1 = self.enc1(torch.cat([c_f1, c_enc2], 1))
        rmp1 = F.upsample_bilinear(self.mp1(ccp),c_enc1.size()[2:])
        rmp2 = F.upsample_bilinear(self.mp2(ccp),c_enc2.size()[2:])
        rmp3 = F.upsample_bilinear(self.mp3(ccp),c_enc3.size()[2:])
        rmp4 = F.upsample_bilinear(self.mp4(ccp),c_enc4.size()[2:])
        rmp5 = F.upsample_bilinear(ccp,c_enc5.size()[2:])

        p5 = self.seg5(torch.cat([rmp5,c_enc5],1))
        p4 = self.seg4(torch.cat([rmp4,c_enc4],1))
        p3 = self.seg3(torch.cat([rmp3,c_enc3],1))
        p2 = self.seg2(torch.cat([rmp2,c_enc2],1))
        p1 = self.seg1(torch.cat([rmp1,c_enc1],1))
        
        mp = self.mp(ccp) 
        
        x1 = self.x1(torch.cat([p1,F.upsample_bilinear(mp,scale_factor=4)],1))  # 64
        xp1 = self.xp1(x1)
        s2 = self.s2(torch.cat([x1,p2],1))
        x2 = self.x2(torch.cat([s2,p2,xp1],1))
        xp2 = self.xp2(x2)
        s3 = self.s3(torch.cat([F.upsample_bilinear(x2,scale_factor=0.5),p3],1))
        x3 = self.x3(torch.cat([s3,p3,F.upsample_bilinear(xp2,scale_factor=0.5)],1))
        xp3 = self.xp3(x3)
        s4 = self.s4(torch.cat([F.upsample_bilinear(x3,scale_factor=0.5),p4],1))
        x4 = self.x4(torch.cat([s4,p4,F.upsample_bilinear(xp3,scale_factor=0.5)],1))
        xp4 = self.xp4(x4)
        s5 = self.s5(torch.cat([F.upsample_bilinear(x4,scale_factor=0.5),p5],1))
        x5 = self.x5(torch.cat([s5,p5,F.upsample_bilinear(xp4,scale_factor=0.5)],1))
        xp5 = self.xp5(x5)

        q5 = self.q5(p5)
        q4 = self.q4(p4)
        q3 = self.q3(p3)
        q2 = self.q2(p2)
        q1 = self.q1(p1)
        s2 = self.sp2(s2)
        s3 = self.sp3(s3)
        s4 = self.sp4(s4)
        s5 = self.sp5(s5)
        
        q1 = F.upsample_bilinear(q1,scale_factor=2)
        q2 = F.upsample_bilinear(q2,scale_factor=2)
        q3 = F.upsample_bilinear(q3,scale_factor=4)
        q4 = F.upsample_bilinear(q4,scale_factor=8)
        q5 = F.upsample_bilinear(q5,scale_factor=16)
        x1 = F.upsample_bilinear(xp1,scale_factor=2)
        x2 = F.upsample_bilinear(xp2,scale_factor=2)
        x3 = F.upsample_bilinear(xp3,scale_factor=4)
        x4 = F.upsample_bilinear(xp4,scale_factor=8)
        x5 = F.upsample_bilinear(xp5,scale_factor=16)
        
        s2 = F.upsample_bilinear(s2,scale_factor=2)
        s3 = F.upsample_bilinear(s3,scale_factor=4)
        s4 = F.upsample_bilinear(s4,scale_factor=8)
        s5 = F.upsample_bilinear(s5,scale_factor=16)

        mp = F.upsample_bilinear(mp,scale_factor=8)
        pf = self.pf(torch.cat([x1,x2,x3,x4,x5,mp],1))
        return mp,q1,q2,q3,q4,q5,x1,x2,x3,x4,x5,s2,s3,s4,s5,pf
        # return pf
