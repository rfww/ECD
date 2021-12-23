import os
import time
import torch
from options.test_options_CIS import TestOptions
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from dataloader.transform import Transform_test
from dataloader.dataset import NeoData_test
from networks import get_model

from eval import *
import argparse


def print_model(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print("Total number of parameters:{}".format(num_params))


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    imagedir = os.path.join(args.datadir, 'image.txt')
    image2dir = os.path.join(args.datadir, 'image2.txt')
    labeldir = os.path.join(args.datadir, 'label.txt')
                                         
    transform = Transform_test(args.size)
    dataset_test = NeoData_test(imagedir, image2dir, labeldir, transform)
    loader = DataLoader(dataset_test, num_workers=2, batch_size=1, shuffle=False) #test data loader

    model = get_model(args.num_classes, args.cis_model)
    if args.cuda:
        model = model.cuda()

    checkpoint = torch.load(args.cis_model_dir)
    model.load_state_dict(checkpoint)
    model.eval()
    count = 0
    countp0 = 0
    countp1 = 0
    all_time = 0
    for step, colign in enumerate(loader):
        
        img = colign[4].squeeze(0).numpy()       #image-numpy,original image
        img2 = colign[5].squeeze(0).numpy()
        images = colign[0]                       #image-tensor
        classi = colign[1]
        images2 = colign[2]
        label = colign[3]                        #label-tensor
        file_name = colign[6]
        image_name = file_name[0].split("/")[-1]
        folder_name = file_name[0].split("/")[-3]
        if args.cuda:
            images = images.cuda()
            images2 = images2.cuda()
            classi = classi.cuda()
        inputs = Variable(images, volatile=True)
        inputs2 = Variable(images2, volatile=True)
        
        stime = time.time()
        c,n,dp=model(inputs,inputs2,1)
        all_time += time.time()-stime
        count += 1
        
        if classi.item() == 0 and dp[0][0].item() > dp[0][1].item():
            countp0 += 1
        elif classi.item() == 1 and dp[0][0].item() < dp[0][1].item():
            countp1 += 1
        
        print("This is the {}th of image!".format(count))
    print(countp0)
    print(countp1)
    print(count)
    print("ACC:{}".format((countp0+countp1)/count))
    print("Running time:{}".format(all_time/332))
    print_model(model)

if __name__ == '__main__':
    parser = TestOptions().parse()
    main(parser)


