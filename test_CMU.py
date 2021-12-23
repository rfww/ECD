import os
import time
import torch
from options.test_options_CMU import TestOptions
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
# from utils.label2Img import label2rgb
from dataloader.transform import Transform_test
from dataloader.dataset import NeoData_test
from networks import get_model

from eval import *
import argparse

def main(args):
    despath = args.savedir
    if not os.path.exists(despath):
        os.mkdir(despath)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    imagedir = os.path.join(args.datadir, 'image.txt')
    image2dir = os.path.join(args.datadir, 'image2.txt')
    labeldir = os.path.join(args.datadir, 'label.txt')
                                         
    transform = Transform_test(args.size)
    dataset_test = NeoData_test(imagedir, image2dir, labeldir, transform)
    loader = DataLoader(dataset_test, num_workers=2, batch_size=1, shuffle=False) #test data loader

    model = get_model(args.num_classes, args.cd_model)
    model_cis = get_model(args.num_classes, args.cis_model)
    if args.cuda:
        model = model.cuda()
        model_cis = model_cis.cuda()

    checkpoint1 = torch.load(args.cd_model_dir)
    model.load_state_dict(checkpoint1,strict=False)
    model.eval()

    checkpoint2 = torch.load(args.cis_model_dir)
    model_cis.load_state_dict(checkpoint2)
    model_cis.eval()

    count = 0
    sum = 0
    total = 0 
    for step, colign in enumerate(loader):
        t1 = time.time()
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
        
        c,n,dp=model_cis(inputs,inputs2,1)
        pr, p1, p2, p3, p4, p5, x1, x2, x3, x4, x5, s2, s3, s4, s5, pf = model(c,n,dp)
     
        etime = time.time()
        sum += (etime-stime)
        count += 1
        print("This is the {}th of image!".format(count),"// Avg time/img: %.4f s" % (etime-stime))
       
        out_pr = pr[0].cpu().max(0)[1].data.squeeze(0).byte().numpy()  # index of max-channel
        out_pf = pf[0].cpu().max(0)[1].data.squeeze(0).byte().numpy()  # index of max-channel

        Image.fromarray(out_pr * 255).save(despath + 'RSS_' + folder_name + '_' + image_name.split(".")[0] + '_pr.png')
        Image.fromarray(out_pf * 255).save(despath + 'RSS_' + folder_name + '_' + image_name.split(".")[0] + '_pf.png')


if __name__ == '__main__':
    parser = TestOptions().parse()
    main(parser)


