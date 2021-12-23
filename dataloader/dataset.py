import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.read_image_label import read_image_class,read_label
EXTENSIONS = ['.jpg', '.png','.JPG','.PNG']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return os.path.join(root, '{}{}'.format(basename,extension))

def image_path_city(root, name):
    return os.path.join(root, '{}'.format(name))

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class NeoData(Dataset):
    def __init__(self, imagepath=None, imagepath2=None, labelpath=None, transform=None):
        #  make sure label match with image 
        self.transform = transform 
        assert os.path.exists(imagepath), "{} not exists !".format(imagepath)
        assert os.path.exists(imagepath2), "{} not exists !".format(imagepath2)
        assert os.path.exists(labelpath), "{} not exists !".format(labelpath)                                  
        
        image,classi   = read_image_class(imagepath)
        image2,classi2 = read_image_class(imagepath2)
        label = read_label(labelpath)
        self.train_set = (
            image,
            classi,
            image2,
            classi2,
            label
        )
       

    def __getitem__(self, index):
        filename   = self.train_set[0][index]
        classi     = self.train_set[1][index]
        filename2  = self.train_set[2][index]
        classi2    = self.train_set[3][index]
        filenameGt = self.train_set[4][index]
        
        with open(filename, 'rb') as f: 
            image = load_image(f).convert('RGB')
        with open(filename2, 'rb') as f:
            image2 = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')
        # print("--------------------------------------")
        if self.transform is not None:#########################
            image, image2, label = self.transform(image, image2, label)

        return image, classi, image2, label

    def __len__(self):
        return len(self.train_set[0])
    
class NeoData_test(Dataset):
    def __init__(self, imagepath=None, imagepath2=None, labelpath=None, transform=None):
        self.transform = transform 
        
        assert os.path.exists(imagepath), "{} not exists !".format(imagepath)
        assert os.path.exists(imagepath2), "{} not exists !".format(imagepath2)
        assert os.path.exists(labelpath), "{} not exists !".format(labelpath)
        
        image,classi   = read_image_class(imagepath)
        image2,classi2 = read_image_class(imagepath2)
        label = read_label(labelpath)
        self.test_set = (
            image,
            classi,
            image2,
            classi2,
            label
        )
        print("Length of test data is {}".format(len(self.test_set[0])))
    def __getitem__(self, index):
        filename   = self.test_set[0][index]
        classi     = self.test_set[1][index]
        filename2  = self.test_set[2][index]
        #classi2    = self.test_set[3][index]
        filenameGt = self.test_set[4][index]
        
        with open(filename, 'rb') as f: # advance
            image = load_image(f).convert('RGB')
        with open(filename2,'rb') as f:
            image2 = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')

        if self.transform is not None:
            image_tensor, image_tensor2, label_tensor, img,img2 = self.transform(image,image2, label)

        return (image_tensor,classi, image_tensor2, label_tensor, np.array(img),np.array(img2),filenameGt)  #return original image, in order to show segmented area in origin

    def __len__(self):
        return len(self.test_set[0])

