import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision import models
from utils.spp_layer import spatial_pyramid_pool


class CIS_VGGBN(nn.Module):
    def __init__(self, backbone='vgg16_bn', pretrained=True, freeze_backbone=False):
        super(CIS_VGGBN, self).__init__()
        self.output_num = [4,2,1]
        vgg = models.vgg16_bn(pretrained)
        features = list(vgg.features.children())
        self.dec1 = nn.Sequential(*features[:7])  # 160
        # self.dec2 = nn.Sequential(*features[5:10])  # 80
        # self.dec3 = nn.Sequential(*features[10:17])  # 40
        # self.dec4 = nn.Sequential(*features[17:24])  # 20
        # self.dec5 = nn.Sequential(*features[24:])  # 10

        self.cis1 = nn.Sequential(

            nn.Conv2d(128, 64, 3, padding=1,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
   
        self.cis2 = nn.Sequential(
            nn.Linear(1344, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.dp = nn.Softmax()
     
    def forward(self, x, y,bt):
        x_f1 = self.dec1(x)
        y_f1 = self.dec1(y)
        enc  = torch.cat((x_f1, y_f1), 1)
        clc1 = self.cis1(enc)
        spp  = spatial_pyramid_pool(clc1,bt,[clc1.size(2),clc1.size(3)], self.output_num)
        clc2 = self.cis2(spp)
        dp   = self.dp(clc2)
        return x_f1, y_f1, dp
    # def initialize(self):
    #     self.load_state_dict(torch.load('../res/resnet50-19c8e357.pth'), strict=False)
