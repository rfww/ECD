from .fcn import FCN8, FCN16, FCN32
from .erfnet import ERFNet
from .utils import *
from .mlDifNet import mlDifNet
from .mlDifNet_enc import mlDifNet_enc
from .mlDifNet_encxy import mlDifNet_encxy
from .eNet import eNet
from .mlDifNetconv1_4 import mlDifNetconv1_4
from .mlDifNetconv1_3 import mlDifNetconv1_3
from .mlDifNetAD import mlDifNetAD

from .eff_wo_top2dw3_40 import feature_extractor
from .loadp_network import loadp
from .cis import CIS_VGGBN
from .cis_resnet import CIS_ResNet
from .ARPPNET import ARPPNET
from .mlDifNet_2 import mlDifNet_2

net_dic = {'mlDifNet_2':mlDifNet_2, 'ARPPNET':ARPPNET, 'cis':CIS_VGGBN, 'cis_resnet':CIS_ResNet,'SC2D': feature_extractor, 'loadp': loadp,'mlDifNetconv1_4': mlDifNetconv1_4,'mlDifNet': mlDifNet, 'mlDifNet_enc': mlDifNet_enc,'mlDifNet_encxy': mlDifNet_encxy}


def get_model(num_classes, model_name):

    Net = net_dic[model_name]
    model = Net(num_classes)
    # model.apply(weights_init)
    return model
