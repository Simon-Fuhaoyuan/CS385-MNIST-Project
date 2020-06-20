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
from sklearn.decomposition import PCA
import cv2

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

color = ['pink', 'red', 'orange', 'gold', 'lime', 'green', 'cyan', 'blue', 'violet', 'purple']

def tSNE(x, y, digits, name, n_components=2):
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
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    if not os.path.isdir('images'):
        os.makedirs('images')
    plt.tight_layout()
    plt.savefig('images/' + name + '.png')
    # plt.show()

def pca(x, y, digits, name, n_components=2):
    pca_ = PCA(n_components=3)
    X_pca = pca_.fit_transform(x)

    print("Org data dimension is {}. Embedded data dimension is {}".format(x.shape[-1], X_pca.shape[-1]))

    x_min, x_max = X_pca.min(0), X_pca.max(0)
    X_norm = (X_pca - x_min) / (x_max - x_min)  # normalize
    fig = plt.figure()
    
    ax = fig.gca(projection='3d')
    for i, (digit) in enumerate(digits):
        X = X_norm[y == digit][: , 0]
        Y = X_norm[y == digit][: , 1]
        Z = X_norm[y == digit][: , 2]
        ax.scatter(X, Y, Z, s=20, c=color[i], label='digit %d'%digit)
    
    plt.xticks([])
    plt.yticks([])
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    if not os.path.isdir('images'):
        os.makedirs('images')
    plt.tight_layout()
    plt.savefig('images/' + name + '_pca.png')
    # plt.show()


class GradCAM(object):
    def __init__(self, i, cfg, model_dict, input, output, fmaps, target):
        self.cfg = cfg
        self.init_image_size = cfg.resolution
        self.output_dir = 'images'
        self.fmap_size = fmaps.size()[2:]
        self.fmaps = fmaps.cpu().detach().numpy()
        self.input = input.cpu().detach().numpy() * 255
        self.input = np.swapaxes(self.input, 1, 2)
        self.input = np.swapaxes(self.input, 2, 3)
        self.output = output.cpu().detach().numpy()
        self.grads = self._get_grads(model_dict).cpu().detach().numpy()
        self.target = target.cpu().detach().numpy()
        self.group = i
    
    def _get_grads(self, model_dict):
        model = torch.load(model_dict)
        grads = model['classifier.weight']
        
        return grads
    
    def save(self, gcam, raw_image, id, category_id):
        gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
        gcam = gcam.astype(np.float) # + raw_image.astype(np.float)
        if(gcam.max() != 0):
           gcam = gcam / gcam.max() * 255.0
        output_dir = os.path.join(self.output_dir, self.cfg.model, 'gradcam')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        filename = '/group%d_num_%d_cat%d.png' %(self.group, id, category_id)
        filename_init = '/group%d_num_%d.png' %(self.group, id)
        cv2.imwrite(output_dir + filename, np.uint8(gcam))
        cv2.imwrite(output_dir + filename_init, np.uint8(raw_image))
        # np.save(output_dir + filename, gcam)
        # np.save(output_dir + filename_init, raw_image)

    def generate(self):
        for i in range(self.fmaps.shape[0]): # batch size
            gcam = np.zeros((self.fmap_size[0], self.fmap_size[1]))
            category_id = np.argmax(self.output[i])
            target_id = self.target[i]
            if not category_id == target_id:
                continue
            raw_image = self.input[i]
            for j in range(self.fmaps.shape[1]):
                fmap = self.fmaps[i][j]
                grad = self.grads[category_id][j]
                gcam += fmap * grad

            gcam -= gcam.min()
            if(gcam.max() != 0):
                gcam /= gcam.max()
            gcam = cv2.resize(gcam, (self.init_image_size, self.init_image_size))
            self.save(gcam, raw_image, i, category_id)
        print('Group %d finish!' % self.group)


if __name__ == "__main__":
    x, y = load_mnist(kind='t10k')

    x_4 = x[y==4]
    y_4 = y[y==4]
    x_9 = x[y==9]
    y_9 = y[y==9]

    x_group = np.concatenate((x_4, x_9), axis=0)
    y_group = np.concatenate((y_4, y_9), axis=0)

    # print(y_group)
    tSNE(x_group, y_group.astype(np.uint8), [4, 9])
