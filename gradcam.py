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
from functions import GradCAM


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])


def parser_args():
    parser = argparse.ArgumentParser(description='Train Mnist dataset')
    parser.add_argument('--epoch', help='Total epoches', default=50, type=int)
    parser.add_argument('--batch_size', help='Batch size', default=32, type=int)
    parser.add_argument('--resolution', help='Resolution of sparse tensor', default=224, type=int)
    parser.add_argument('--in_channel', help='Input image channel', default=1, type=int)
    parser.add_argument('--loss', help='The loss function', default='crossentropy', type=str)
    parser.add_argument('--final_layer', help='The final layer of CNN', default='softmax', type=str)
    parser.add_argument('--lr', help='The learning rate', default=0.01, type=float)
    parser.add_argument('--root', help='The initial dataset root', default='./Mnist', type=str)
    parser.add_argument('--weight', help='The weight folder', default='./weights', type=str)
    parser.add_argument('--model', help='The architecture of CNN', default='smallnet', type=str)
    parser.add_argument('--workers', help='Number of workers when loading data', default=4, type=int)
    parser.add_argument('--print_freq', help='Number of iterations to print', default=200, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    # experiment options
    parser.add_argument(
        '--exp1', 
        help='Partition input image into a few blocks, then random shuffle each block', 
        action='store_true',
        )
    parser.add_argument(
        '--exp2', 
        help='Add x and y coordinates to each pixel to let image be 28 * 28 * 3, then random shuffle each pixel', 
        action='store_true',
        )

    args = parser.parse_args()
    return args

def main(net, loader, device, config):
    checkpoint = os.path.join(config.weight, config.model + '.pth')
    net.load_state_dict(torch.load(checkpoint))

    total_correct = 0
    total_cnt = 0
    for i, (img, label, _1, _2) in enumerate(loader):
        img = img.to(device)
        pred, fmaps = net(img)
        n_correct, cnt = accuracy(pred, label)
        total_correct += n_correct
        total_cnt += cnt
        
        gradcam = GradCAM(i, config, checkpoint, img, pred, fmaps, label)
        gradcam.generate()
    
    print('acc: %.4f' % (total_correct / total_cnt))
        
        
if __name__ == '__main__':
    config = parser_args()
    logging.info(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = eval('models.' + config.model + '.get_CNN')(config)
    net.to(device)

    logging.info(net)
    test_dataset = Mnist(config, 'test')
    test_loader = Data.DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.workers, 
        pin_memory=True)

    main(net, test_loader, device, config)