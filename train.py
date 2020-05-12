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

def parser_args():
    parser = argparse.ArgumentParser(description='Train Mnist dataset')
    parser.add_argument('--epoch', help='Total epoches', default=50, type=int)
    parser.add_argument('--batch_size', help='Batch size', default=8, type=int)
    parser.add_argument('--resolution', help='Resolution of sparse tensor', default=224, type=int)
    parser.add_argument('--in_channel', help='Input image channel', default=1, type=int)
    parser.add_argument('--lr', help='The learning rate', default=0.001, type=float)
    parser.add_argument('--root', help='The initial dataset root', default='./Mnist', type=str)
    parser.add_argument('--weight', help='The weight folder', default='./weights', type=str)
    parser.add_argument('--model', help='The architecture of CNN', default='vgg11', type=str)
    parser.add_argument('--workers', help='Number of workers when loading data', default=4, type=int)
    parser.add_argument('--print_freq', help='Number of iterations to print', default=50, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    args = parser.parse_args()

    return args

def train(net, dataloader, device, config):
    train_loader, test_loader = dataloader[0], dataloader[1]
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    crit = nn.CrossEntropyLoss().to(device)
    if not os.path.isdir(config.weight):
        os.makedirs(config.weight)
    checkpoint = os.path.join(config.weight, config.model + '.pth')

    logging.info(f'LR: {scheduler.get_lr()}')
    best_acc = 0
    for epoch in range(config.epoch):
        ########## TRAIN ##########
        start = time()
        net.train()
        loss_train = 0.0
        for i, (img, label, _) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device)
            output = net(img)
            loss = crit(output, label)
            loss_train += loss.item()
            optimizer.step()
            loss_train += loss.item()

            if i % config.print_freq == 0:
                logging.info(
                    f'Epoch[{epoch}][{i}/{len(train_loader)}], \
                    Train Loss: {loss.item():.3f}({loss_train / (i + 1):.3f})'
                )

        loss_train /= len(train_loader)
        end = time() - start
        logging.info(
            f'=> Epoch[{epoch}], Average train Loss: {loss_train:.3f}, Tot Time: {end:.3f}'
        )

        ########## TEST ###########
        start = time()
        net.eval()
        total_correct = 0
        total_cnt = 0
        for i, (img, label, _) in enumerate(test_loader):
            img.to(device)
            pred = net(img)
            n_correct, cnt = accuracy(pred, label)
            total_correct += n_correct
            total_cnt += cnt
            if i % config.print_freq == 0:
                logging.info(
                    f'Epoch[{epoch}][{i}/{len(test_loader)}], \
                    Test accuracy: {n_correct / cnt:.3f}({total_correct / total_cnt:.3f})'
                )
        end = time() - start
        acc = total_correct / total_cnt
        logging.info(
            f'=> Epoch[{epoch}], Final accuracy: {acc:.3f}, Tot Time: {end:.3f}\n'
        )
        if acc > best_acc:
            torch.save(net.state_dict(), checkpoint)
            logging.info(f'Saving best model checkpoint at {checkpoint}')
            best_acc = acc
        scheduler.step()
        logging.info(f'LR: {scheduler.get_lr()}')


if __name__ == '__main__':
    config = parser_args()
    logging.info(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = eval('models.' + config.model + '.get_CNN')(config.in_channel)
    net.to(device)

    logging.info(net)
    train_dataset = Mnist(config, 'train')
    test_dataset = Mnist(config, 'test')
    train_loader = Data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.workers, 
        pin_memory=True)
    test_loader = Data.DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.workers, 
        pin_memory=True)
    loader = (train_loader, test_loader)
    train(net, loader, device, config)