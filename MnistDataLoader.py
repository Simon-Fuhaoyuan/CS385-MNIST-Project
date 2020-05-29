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

        if self.cfg.exp1:
            assert self.cfg.in_channel == 1, 'In exp1, the input channel should be 1, but get %d' % (self.cfg.in_channel)
            buffer = []
            block_num = 4
            assert self.resolution % block_num == 0, 'Resolution is not dividable by block number (%d)' % (block_num)
            block_size = self.resolution // block_num
            for i in range(block_num):
                for j in range(block_num):
                    buffer.append(img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size])
            random.shuffle(buffer)
            img_new = np.zeros((self.resolution, self.resolution, 1))
            for i in range(block_num):
                for j in range(block_num):
                    img_new[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = buffer[block_num * i + j]
            img = img_new

        elif self.cfg.exp2:
            assert self.cfg.in_channel == 3, 'In exp1, the input channel should be 3, but get %d' % (self.cfg.in_channel)
            y = np.arange(self.resolution)
            y = np.tile(y, (self.resolution, self.resolution))
            x = y.T
            x = np.resize(x, (self.resolution, self.resolution, 1))
            y = np.resize(y, (self.resolution, self.resolution, 1))
            img = np.concatenate((img, x, y), axis=2)
            np.random.shuffle(img)

        img = self.transforms(img)
        img = img.float()

        one_hot = torch.zeros(10)
        one_hot[lbl] = 1
        one_hot = one_hot.float()
        
        return img, lbl, one_hot, idx
    

if __name__ == "__main__":
    # x, y = load_mnist()
    # img = x[0]
    # img = np.resize(img, (28, 28)).astype(np.uint8)
    # img = Image.fromarray(img)
    # img = img.resize((224, 224))
    # img.show()
    # img = np.array(img)
    # img = np.resize(img, (128, 128, 1))
    # transform = transforms.ToTensor()
    # img = transform(img)
    # print(img.shape)
    resolution = 28
    a = np.arange(resolution)
    a = np.tile(a, (resolution, resolution))
    b = a.T
    a = np.resize(a, (resolution, resolution, 1))
    b = np.resize(b, (resolution, resolution, 1))
    c = np.concatenate((a, b), axis=2)
    print(c.shape)