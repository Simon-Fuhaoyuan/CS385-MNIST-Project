import os
import sys
import argparse
import logging
import numpy as np
from time import time
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from sklearn import manifold

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim

from MnistDataLoader import Mnist
from evaluate import accuracy
import models
from MnistData import load_mnist

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])

def train(config, net, device, train_loader, crit, optimizer, epoch):
    loss_train = 0.0
    for i, (img, label, one_hot, _) in enumerate(train_loader):
        img = img.to(device)
        if config.loss == 'crossentropy':
            gt = label.long().to(device)
        else:
            gt = one_hot.to(device)
        output = net(img)
        loss = crit(output, gt)
        loss_train += loss.item()

        if i % config.print_freq == 0:
            logging.info(
                f'Epoch[{epoch}][{i}/{len(train_loader)}], Train Loss: {loss.item():.3f}({loss_train / (i + 1):.3f})'
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_train /= len(train_loader)
    return loss_train

def test(config, net, device, test_loader, epoch):
    total_correct = 0
    total_cnt = 0
    for i, (img, label, _1, _2) in enumerate(test_loader):
        img = img.to(device)
        pred = net(img)
        n_correct, cnt = accuracy(pred, label)
        total_correct += n_correct
        total_cnt += cnt
        if i % config.print_freq == 0:
            logging.info(
                f'Epoch[{epoch}][{i}/{len(test_loader)}], Test accuracy: {n_correct / cnt:.3f}({total_correct / total_cnt:.3f})'
            )
    
    return total_correct / total_cnt

def get_criterion(config):
    if config.loss == 'crossentropy':
        crit = nn.CrossEntropyLoss()
    elif config.loss == 'mse':
        crit = nn.MSELoss(reduction='mean')
    else:
        logging.info(f'Error: No match criterion')
        exit()

    return crit

color = ['r', 'g', 'b', 'y']

def tSNE(x, y, digits, n_components=2):
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(x)

    print("Org data dimension is {}. Embedded data dimension is {}".format(x.shape[-1], X_tsne.shape[-1]))

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # normalize
    plt.figure()
    
    for i, (digit) in enumerate(digits):
        X = X_norm[y == digit][: , 0]
        Y = X_norm[y == digit][: , 1]
        plt.scatter(X, Y, s=20, c=color[i], label='digit %d'%digit)
    
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.savefig('images/%dand%d.png' % (digits[0], digits[1]))


if __name__ == "__main__":
    x, y = load_mnist(kind='t10k')

    x_4 = x[y==4]
    y_4 = y[y==4]
    x_9 = x[y==9]
    y_9 = y[y==9]

    x_group = np.concatenate((x_4, x_9), axis=0)
    y_group = np.concatenate((y_4, y_9), axis=0)

    # print(y_group)
    tSNE(x_group, y_group.astype(np.uint8), [4,9])