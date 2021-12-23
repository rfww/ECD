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
            nn.Upsample(scale_factor=2,mode='bilinear'),
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

class Fusion_mld2_fcn(nn.Module):

    def __init__(self, num_classes):

        super().__init__()
        ######mldnet2########################################
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
        self.fuse = nn.Conv2d(18, num_classes, 3, padding=1)

        self.conv_cat1 = nn.Conv2d(4,num_classes,3,padding=1)
        self.conv_cat2 = nn.Conv2d(6, num_classes, 3, padding=1)
        self.conv_cat3 = nn.Conv2d(6, num_classes, 3, padding=1)
        #self.concat2 = nn.Conv2d(4, num_classes, 3, padding=1)

        #########FCN#######################################
        my_model = models.vgg16(pretrained=True)
        input_1_new = nn.Conv2d(6, 64, (3, 3), 1, 1)
        my_model.features[0] = input_1_new
        feats = list(my_model.features.children())
        self.feats = nn.Sequential(*feats[0:10])
        self.feat3 = nn.Sequential(*feats[10:17])
        self.feat4 = nn.Sequential(*feats[17:24])
        self.feat5 = nn.Sequential(*feats[24:31])

        #for m in self.modules():
         #   if isinstance(m, nn.Conv2d):
          #      m.requires_grad = False

        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score_feat3 = nn.Conv2d(256, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)
        self.score_fconn = nn.Conv2d(4096, num_classes, 1)


    def forward(self, x, y):
        #########forward_fcn######################
        concat_xy = torch.cat([x, y], 1)
        feats = self.feats(concat_xy)
        feat3 = self.feat3(feats)
        feat4 = self.feat4(feat3)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)

        score_feat3 = self.score_feat3(feat3)
        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)

        score_1 = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
        score_fuse1 = score_1 + score_feat4
        score_2 = F.upsample_bilinear(score_fuse1, score_feat3.size()[2:])
        score_fuse2 = score_2 + score_feat3

        score_3 = F.upsample_bilinear(score_fuse2, x.size()[2:])
        ##########forward_mldnet2######################
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
        enc5xy = self.enc5xy(abs(enc5 - enc5y))
        enc4xy = self.enc4xy(torch.cat([abs(enc4 - enc4y), enc5xy], 1))
        enc3xy = self.enc3xy(torch.cat([abs(enc3 - enc3y), enc4xy], 1))

        enc5xy_n = F.upsample_bilinear(self.final5(enc5xy), x.size()[2:])
        enc4xy_n = F.upsample_bilinear(self.final4(enc4xy), x.size()[2:])
        enc3xy_n = F.upsample_bilinear(self.final3(enc3xy), x.size()[2:])

        score_final1 = F.upsample_bilinear(score_1, x.size()[2:])
        score_final2 = F.upsample_bilinear(score_2, x.size()[2:])
        score_final3 = score_3

        conv_cat1 = self.conv_cat_1(torch.cat([self.final5(enc5xy),score_1],1))
        conv_cat1_up = F.upsample_bilinear(conv_cat1, score_2.size()[2:])
        conv_cat2 = self.conv_cat_2(torch.cat([self.final4(enc4xy), score_2,conv_cat1_up], 1))
        conv_cat2_up = F.upsample_bilinear(conv_cat2, score_3.size()[2:])
        conv_cat3 = self.conv_cat_3(torch.cat([self.final3(enc4xy), score_3,conv_cat2_up], 1))
        conv_cat3_up = F.upsample_bilinear(conv_cat3, x.size()[2:])

        conv_cat1_n = F.upsample_bilinear(conv_cat1, x.size()[2:])
        conv_cat2_n = F.upsample_bilinear(conv_cat2, x.size()[2:])
        conv_cat3_n = conv_cat3_up


        final = self.fuse(torch.cat([enc5xy_n, enc4xy_n, enc3xy_n, score_final1,
                                     score_final2, score_final3, conv_cat1_n,
                                     conv_cat2_n, conv_cat3_n], 1))

        return {enc5xy_n, enc4xy_n, enc3xy_n, score_final1, score_final2,

            score_final3,conv_cat1_n, conv_cat2_n, conv_cat3_n, final}


#model = Fusion_mld_fcn()
#print(model)
#print(model.score_feat4)
#print(model.feats[0])
#print(model.input_1)

#model_2 = models.vgg16(pretrained=False)
#print(model_2)
#input_1 = model_2.features[0]
#input_1_new = nn.Conv2d(6,64,(3,3),1,1)
#model_2.features[0] = input_1_new
#print(input_1_new)
#print(model_2)