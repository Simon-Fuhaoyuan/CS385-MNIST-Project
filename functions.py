import os
import sys
import argparse
import logging
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from MnistDataLoader import Mnist
from evaluate import accuracy
import models


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])

def train(config, net, device, train_loader, crit, optimizer, epoch):
    loss_train = 0.0
    for i, (img, label, _) in enumerate(train_loader):
        img = img.to(device)
        label = label.long().to(device)
        output = net(img)
        loss = crit(output, label)
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
    for i, (img, label, _) in enumerate(test_loader):
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
    