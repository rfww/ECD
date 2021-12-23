import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy import ndimage

class BaseDataSet(Dataset):
    def __init__(self, root, split, mean, std, base_size=None, augment=True, val=False,
                crop_size=321, scale=True, flip=True, rotate=False, blur=False, return_id=False):
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop_size = crop_size
        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur
        self.val = val
        self.files = []
        self.files2 = []
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.return_id = return_id

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError
    
    def _load_data(self, index):
        raise NotImplementedError

    def _val_augmentation(self, image,image2, e_label, label):
        if self.crop_size:
            h, w = label.shape
            # Scale the smaller side to crop size
            if h < w:
                h, w = (self.crop_size, int(self.crop_size * w / h))
            else:
                h, w = (int(self.crop_size * h / w), self.crop_size)

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            image2 = cv2.resize(image2,(w,h), interpolation=cv2.INTER_LINEAR)
            e_label = Image.fromarray(e_label).resize((w, h), resample=Image.NEAREST)
            e_label = np.asarray(e_label, dtype=np.int32)

            label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.int32)

            # Center Crop
            h, w = label.shape
            start_h = (h - self.crop_size )// 2
            start_w = (w - self.crop_size )// 2
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            image2 = image2[start_h:end_h, start_w:end_w]
            e_label = e_label[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]
        return image, image2, e_label, label

    def _augmentation(self, image, image2, e_label, label):
        h, w, _ = image.shape
        # Scaling, we set the bigger to base size, and the smaller 
        # one is rescaled to maintain the same ratio, if we don't have any obj in the image, re-do the processing
        if self.base_size:
            if self.scale:
                longside = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
            else:
                longside = self.base_size
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            image2 = cv2.resize(image2,(w,h), interpolation=cv2.INTER_LINEAR)
            e_label = cv2.resize(e_label, (w, h), interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    
        h, w, _ = image.shape
        # Rotate the image with an angle between -10 and 10
        if self.rotate:
            angle = random.randint(-10, 10)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)#, borderMode=cv2.BORDER_REFLECT)
            image2 = cv2.warpAffine(image2, rot_matrix, (w, h),
                                   flags=cv2.INTER_LINEAR)  # , borderMode=cv2.BORDER_REFLECT)
            e_label = cv2.warpAffine(e_label, rot_matrix, (w, h),
                                   flags=cv2.INTER_NEAREST)  # ,  borderMode=cv2.BORDER_REFLECT)
            label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)#,  borderMode=cv2.BORDER_REFLECT)

        # Padding to return the correct crop size
        if self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT,}
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
                image2 = cv2.copyMakeBorder(image2, value=0, **pad_kwargs)
                e_label = cv2.copyMakeBorder(e_label, value=0, **pad_kwargs)
                label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)
            
            # Cropping 
            h, w, _ = image.shape
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            image2 = image2[start_h:end_h, start_w:end_w]
            e_label = e_label[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

        # Random H flip
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                image2 = np.fliplr(image2).copy()
                e_label = np.fliplr(e_label).copy()
                label = np.fliplr(label).copy()

        # Gaussian Blud (sigma between 0 and 1.5)
        if self.blur:
            sigma = random.random()
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
            image2 = cv2.GaussianBlur(image2, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
                                     borderType=cv2.BORDER_REFLECT_101)
        return image, image2, e_label, label
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, image2, e_label, label, image_id = self._load_data(index)
        if self.val:
            image, image2, e_label, label = self._val_augmentation(image, image2, e_label, label)
        elif self.augment:
            image, image2, e_label, label = self._augmentation(image, image2, e_label, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        e_label = torch.from_numpy(np.array(e_label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image))
        image2 = Image.fromarray(np.uint8(image2))
        if self.return_id:
            return  self.normalize(self.to_tensor(image)), self.normalize(self.to_tensor(image2)), e_label, label, image_id
        return self.normalize(self.to_tensor(image)), self.normalize(self.to_tensor(image2)), e_label, label

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

