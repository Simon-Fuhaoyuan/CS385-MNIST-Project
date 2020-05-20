import copy
import logging
import random

import struct
import cv2
import numpy as np
import torch
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def load_mnist(path='MNIST', kind='train'):
    """Load MNIST data from path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def visualize(path='MNIST'):
    X_train, y_train = load_mnist(path=path)
    print(X_train.shape)
    print(y_train.shape)
    fig, ax = plt.subplots(
        nrows=5,
        ncols=5,
        sharex=True,
        sharey=True, )

    ax = ax.flatten()
    for i in range(25):
        img = X_train[y_train == 7][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    # plt.show()
    plt.savefig('images/test.png')

def statistic(path='MNIST'):
    x_train, y_train = load_mnist(path=path)
    x_test, y_test = load_mnist(path=path, kind='t10k')
    print(x_train[0])
    train_sta = 0
    test_sta = 0
    for i in range(10):
        a = (y_train == i).sum()
        b = (y_test == i).sum()
        train_sta += a
        test_sta += b
        print('Train set[%d]:' % i, a)
        print('Test set[%d]:' % i, b)
    print('Train total:', train_sta)
    print('Test total:', test_sta)

def load_mnist_by_class(digit, count, path='MNIST', kind='train'):
    x, y = load_mnist(path=path, kind=kind)
    x = x[y == digit][:int(count)]
    y = y[y == digit][:int(count)]

    return x, y

if __name__ == "__main__":
    visualize()
    statistic()