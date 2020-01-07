import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from unet.unet import UnetLite
from os import listdir
from os import getcwd
import numpy as np
from glob import glob
import torch

from PIL import Image


class ProcessData():
    def __init__(self, imgs_dir, masks_dir,num_of_samples=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [file.split('.')[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

        if num_of_samples is not None:
            self.ids = self.ids[:num_of_samples]


    def __preprocess__(self, pil_img):
        w, h = pil_img.size
        pil_img = pil_img.resize((220, 140))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans


    def getData(self):
        imgs = []
        masks = []
        for idx in self.ids:
            
            mask_file = glob(self.masks_dir + idx + '*')[0]
            img_file = glob(self.imgs_dir + idx + '*')[0]

            mask = Image.open(mask_file)
            img = Image.open(img_file)

            imgs.append(self.__preprocess__(img))
            masks.append(self.__preprocess__(mask))

        return torch.tensor(imgs), torch.tensor(masks)
