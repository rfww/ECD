import os
import time
import math
import torch
import numpy as np
from torchvision import models
from eval import *
import torch.nn as nn
from utils import evalIoU
from networks import get_model
from torch.autograd import Variable
from dataloader.dataset import NeoData
from torch.utils.data import DataLoader
from dataloader.transform import MyTransform
from torchvision.transforms import ToPILImage
from options.train_options_CIS import TrainOptions
from torch.optim import SGD, Adam, lr_scheduler
from criterion.criterion import CrossEntropyLoss2d
import argparse
NUM_CHANNELS = 3


def get_loader(args):
    # add the weight of each class (1/ln(c+Pclass))
    # calculate the weights of each class

    # weight[0]=1.45
    # weight[1]=54.38
    # weight[2] = 428.723
    imagepath_train = os.path.join(args.datadir, 'train/image.txt')
    imagepath_train2 = os.path.join(args.datadir, 'train/image2.txt')
    labelpath_train = os.path.join(args.datadir, 'train/label.txt')
    imagepath_val = os.path.join(args.datadir, 'test/image.txt')
    imagepath_val2 = os.path.join(args.datadir, 'test/image2.txt')
    labelpath_val = os.path.join(args.datadir, 'test/label.txt')

    train_transform = MyTransform(reshape_size=(320, 320), crop_size=(320, 320),  # 500,350 448 320
                                  augment=True)  # data transform for training set with data augmentation, including resize, crop, flip and so on
    val_transform = MyTransform(reshape_size=(320, 320), crop_size=(320, 320),
                                augment=False)  # data transform for validation set without data augmentation

    dataset_train = NeoData(imagepath_train, imagepath_train2,
                            labelpath_train, train_transform)  # DataSet
    dataset_val = NeoData(imagepath_val, imagepath_val2,
                          labelpath_val, val_transform)

    loader = DataLoader(dataset_train, num_workers=args.num_workers,
                        batch_size=args.batch_size, shuffle=True, drop_last=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size,
                            shuffle=False, drop_last=True)

    return loader, loader_val


def train(args, model):
    NUM_CLASSES = args.num_classes  # pascal=21, cityscapes=20
    batchs = args.batch_size

    savedir = args.savedir
    weight = torch.ones(NUM_CLASSES)
    loader, loader_val = get_loader(args)

    if args.cuda:
        criterion = CrossEntropyLoss2d(weight).cuda()
        classiffication = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = CrossEntropyLoss2d(weight)
        # classiffication = torch.nn.CrossEntropyLoss()
        # save log
    automated_log_path = savedir + "/automated_log.txt"
    if (not os.path.exists(automated_log_path)):  # dont add first line if it exists
        with open(automated_log_path, "a") as myfile:
            myfile.write(
                "Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")
    paras = dict(model.named_parameters())
    paras_new = []

    for k, v in paras.items():

        if 'bias' in k:
            if 'dec' in k:
                paras_new += [{'params': [v], 'lr': 0.02 *
                               args.lr, 'weight_decay': 0}]
            else:
                paras_new += [{'params': [v], 'lr': 0.2 *
                               args.lr, 'weight_decay': 0}]
        else:
            if 'dec' in k:
                paras_new += [{'params': [v], 'lr': 0.01 *
                               args.lr, 'weight_decay': 0.00004}]
            else:
                paras_new += [{'params': [v], 'lr': 0.1 *
                               args.lr, 'weight_decay': 0.00004}]
    optimizer = Adam(paras_new, args.lr, (0.9, 0.999),
                     eps=1e-08, weight_decay=1e-4)

    def lambda1(epoch): return pow((1 - ((epoch - 1) / args.num_epochs)), 0.9)
    # learning rate changed every epoch
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    start_epoch = 1

    for epoch in range(start_epoch, args.num_epochs + 1):
        print("----- TRAINING - EPOCH", epoch, "-----")
        scheduler.step(epoch)
        epoch_loss = []
        time_train = []

        # confmatrix for calculating IoU
        confMatrix = evalIoU.generateMatrixTrainId(evalIoU.args)
        perImageStats = {}
        nbPixels = 0
        usedLr = 0
        # for param_group in optimizer.param_groups:
        for param_group in optimizer.param_groups:
            # print("LEARNING RATE:", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.cuda().train()
        count = 1
        for step, (images, classi, images2, labels) in enumerate(loader):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                images2 = images2.cuda()
                labels = labels.cuda()
                classi = torch.tensor(classi).cuda()

            inputs = Variable(images)
            inputs2 = Variable(images2)

            targets = Variable(labels)
            classi = Variable(classi)

            # p1, p2, p3, p4, p5 = modelp(inputs, inputs2)

            x, y, dp = model(inputs, inputs2, args.batch_size)
            loss = classiffication(dp, classi)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print('loss: {} (epoch: {}, step: {})'.format(average, epoch, step),
                      "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        iouAvgStr, iouTrain, classScoreList = cal_iou(evalIoU, confMatrix)
        print("EPOCH IoU on TRAIN set: ", iouAvgStr)

        # calculate eval-loss and eval-IoU
        #average_epoch_loss_val, iouVal = eval(args, model, loader_val, criterion, epoch)
        average_epoch_loss_val = 0
        iouVal = 0
        # save model every X epoch
        if epoch % args.epoch_save == 0:
            torch.save(model.state_dict(), '{}_{}.pth'.format(
                os.path.join(args.savedir, args.model), str(epoch)))
        t = time.strftime("%D:%H:%m:%S")
        # save log
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%s\t\t%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (
                t,epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr))

    return (model)


def main(args):
    '''
        Train the model and record training options.
    '''
    savedir = '{}'.format(args.savedir)
    modeltxtpath = os.path.join(savedir, 'model.txt')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    with open(savedir + '/opts.txt', "w") as myfile:  # record options
        myfile.write(str(args))

    # initialize the network
    model = get_model(args.num_classes, args.cis_model)  # load model
    decoders = list(models.vgg16_bn(pretrained=True).features.children())
    model.dec1 = nn.Sequential(*decoders[:7])
    #checkpoint = torch.load(args.pretrained)
    #model.load_state_dict(checkpoint)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.requires_grad = True

    with open(modeltxtpath, "w") as myfile:  # record model
        myfile.write(str(model))

    if args.cuda:
        model = model.cuda()
        print("---------cuda--------")
    # checkpoint = torch.load(args.pretrained)
    # model.load_state_dict(checkpoint)
    model.train()

    num_epochs = args.num_epochs
    print("========== TRAINING ===========")
    stime = time.time()
    model = train(args, model)

    print("========== TRAINING FINISHED ===========")
    print("%g epochs completed in %.3f hours.\n" %
          (num_epochs, (time.time()-stime)/3600))


if __name__ == '__main__':
    parser = TrainOptions().parse()
    main(parser)
