import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

import numpy as np
import argparse


class Generator(nn.Module):
    def __init__(self, opt, img_shape):
        super(Generator, self).__init__()

        self.opt = opt
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Mnist dataset')
    parser.add_argument('--latent_dim', default=100, type=int)
    opt = parser.parse_args()

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    G = Generator(opt, (1, 28, 28)).cuda()
    z = Variable(Tensor(np.random.normal(0, 1, (2, opt.latent_dim))))
    gen_img = G(z)
    print(gen_img.shape)
