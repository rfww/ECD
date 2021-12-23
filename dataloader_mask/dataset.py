import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset

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
    def __init__(self, imagepath=None, imagepath2=None, maskpath = None, labelpath=None, transform=None):
        #  make sure label match with image 
        self.transform = transform 
        assert os.path.exists(imagepath), "{} not exists !".format(imagepath)
        assert os.path.exists(imagepath2), "{} not exists !".format(imagepath2)
        assert os.path.exists(maskpath), "{} not exists !".format(maskpath)
        assert os.path.exists(labelpath), "{} not exists !".format(labelpath)                                  
        self.image = []
        self.image2 = []
        self.mask = []
        self.label= [] 
        with open(imagepath,'r') as f:
            for line in f:
                self.image.append(line.strip())
        with open(imagepath2,'r') as f:
            for line in f:
                self.image2.append(line.strip())
        with open(maskpath,'r')as f:
            for line in f:
                self.mask.append(line.strip())
        with open(labelpath,'r') as f:
            for line in f:
                self.label.append(line.strip())

    def __getitem__(self, index):
        filename = self.image[index]
        filename2 = self.image2[index]
        filenamemask = self.mask[index]
        filenameGt = self.label[index]
        
        with open(filename, 'rb') as f: 
            image = load_image(f).convert('RGB')
        with open(filename2, 'rb') as f:
            image2 = load_image(f).convert('RGB')
        with open(filenamemask,'rb') as f:
            mask = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')
        if self.transform is not None:#########################
            image, image2, mask, label = self.transform(image, image2, mask, label)

        return image,image2, mask, label

    def __len__(self):
        return len(self.image)
    
class NeoData_test(Dataset):
    def __init__(self, imagepath=None, imagepath2=None, maskpath = None, labelpath=None, transform=None):
        self.transform = transform 
        
        assert os.path.exists(imagepath), "{} not exists !".format(imagepath)
        assert os.path.exists(imagepath2), "{} not exists !".format(imagepath2)
        assert os.path.exists(maskpath), "{} not exists !".format(maskpath)
        assert os.path.exists(labelpath), "{} not exists !".format(labelpath)
        
        self.image = []
        self.image2 = []
        self.mask = []
        self.label= [] 
        with open(imagepath,'r') as f:
            for line in f:
                self.image.append(line.strip())
        with open(imagepath2,'r') as f:
            for line in f:
                self.image2.append(line.strip())
        with open(maskpath, 'r') as f:
            for line in f:
                self.mask.append(line.strip())
        with open(labelpath,'r') as f:
            for line in f:
                self.label.append(line.strip())
        print("Length of test data is {}".format(len(self.image)))
    def __getitem__(self, index):
        filename = self.image[index]
        filename2 = self.image2[index]
        filenamemask = self.mask[index]
        filenameGt = self.label[index]
        
        with open(filename, 'rb') as f: # advance
            image = load_image(f).convert('RGB')
        with open(filename2,'rb') as f:
            image2 = load_image(f).convert('RGB')
        with open(filenamemask, 'rb') as f:
            mask = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')

        if self.transform is not None:
            image_tensor, image_tensor2, mask_tensor, label_tensor, img,img2, mask = self.transform(image,image2, mask, label)

        return (image_tensor, image_tensor2, mask_tensor, label_tensor, np.array(img),np.array(img2), np.array(mask), filenameGt)  #return original image, in order to show segmented area in origin

    def __len__(self):
        return len(self.image)

