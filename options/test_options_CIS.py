#-*- coding:utf-8 -*-
import argparse
import os

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--cuda', action='store_true', default=True)
        self.parser.add_argument('--cis-model', default="cis", help='model to train,options:...')
        self.parser.add_argument('--cis-model-dir', default="./save_models2021/cis_cmu/cis_20.pth", help='path to stored-model')
        self.parser.add_argument('--num-classes', type=int, default=2)
        self.parser.add_argument('--datadir', default="./data_CMU/test/", help='path where image2.txt and label.txt lies')
        self.parser.add_argument('-size', default=(480, 480), help='resize the test image')
        self.parser.add_argument('--stored', default=True, help='whether or not store the result')
        self.parser.add_argument('--savedir', type=str, default='', help='without saving')
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
