from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
from PIL import Image
import numpy as np
import torch
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from MnistData import load_mnist

logger = logging.getLogger(__name__)


class Mnist(Dataset):
    def __init__(self, cfg, phase):
        self.num_category = 10
        self.cfg = cfg
        self.phase = phase # train or test
        if phase == 'test':
            phase = 't10k'
        self.images, self.labels = load_mnist(kind=phase)
        self.resolution = cfg.resolution
        self.transforms = transforms.ToTensor()
        self.db = self._get_db()
        logger.info('=> Loading {} images from {}'.format(self.phase, self.cfg.root))
        logger.info('=> num_images: {}'.format(len(self.db['x'])))
    
    def _get_db(self):
        db = {}
        db['x'] = self.images
        db['y'] = self.labels

        return db
    
    def __len__(self,):
        return len(self.db['x'])
    
    def __getitem__(self, idx):
        img = copy.deepcopy(self.db['x'][idx]).astype(np.uint8)
        lbl = copy.deepcopy(self.db['y'][idx])

        img = np.resize(img, (28, 28)).astype(np.uint8)
        img = Image.fromarray(img)
        if self.resolution != 28:
            img = img.resize((self.resolution, self.resolution))
        img = np.array(img)

        img = np.resize(img, (self.resolution, self.resolution, 1))
        img = self.transforms(img)
        
        return img, lbl, idx
    

if __name__ == "__main__":
    x, y = load_mnist()
    img = x[0]
    img = np.resize(img, (28, 28)).astype(np.uint8)
    img = Image.fromarray(img)
    img = img.resize((224, 224))
    img.show()
    # img = np.array(img)
    # img = np.resize(img, (128, 128, 1))
    # transform = transforms.ToTensor()
    # img = transform(img)
    # print(img.shape)