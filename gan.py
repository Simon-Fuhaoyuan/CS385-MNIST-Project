import os
import sys
import argparse
import logging
import numpy as np
from time import time

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from torch.autograd import Variable

from MnistDataLoader import Mnist
from models.GAN import Generator, Discriminator


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])


def parser_args():
    parser = argparse.ArgumentParser(description='Train GAN on Mnist dataset')
    parser.add_argument('--epoch', help='Total epoches', default=50, type=int)
    parser.add_argument('--batch_size', help='Batch size', default=32, type=int)
    parser.add_argument('--resolution', help='Resolution of sparse tensor', default=28, type=int)
    parser.add_argument('--in_channel', help='Input image channel', default=1, type=int)
    parser.add_argument("--latent_dim", help="dimensionality of the latent space", default=100, type=int)
    parser.add_argument("--img_size", help="size of each image dimension", default=28, type=int)    
    parser.add_argument('--lr', help='The learning rate', default=0.0002, type=float)
    parser.add_argument('--weight', help='The weight folder', default='./weights', type=str)
    parser.add_argument('--workers', help='Number of workers when loading data', default=4, type=int)
    parser.add_argument('--print_freq', help='Number of iterations to print', default=200, type=int)
    parser.add_argument('--root', help='The initial dataset root', default='./Mnist', type=str)
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
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

def save_image(imgs, epoch, index):
    imgs = imgs.detach().cpu().numpy()
    imgs = (imgs + 1) / 2
    imgs = imgs * 255
    fig, ax = plt.subplots(
        nrows=5,
        ncols=5,
        sharex=True,
        sharey=True, )

    ax = ax.flatten()
    for i in range(25):
        img = imgs[i].reshape(28, 28).astype(np.uint8)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    if not os.path.isdir('images'):
        os.makedirs('images')
    plt.savefig('images/synthetic_%d_%d.png' % (epoch, index))

def main(generator, discriminator, dataloader, device, config):
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    adversarial_loss = nn.BCELoss().to(device)
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=(config.b1, config.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.b1, config.b2))

    for epoch in range(config.epoch):
        for i, (imgs, _1, _2, _3) in enumerate(dataloader):
            imgs = (imgs - 0.5) / 0.5
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
            real_imgs = Variable(imgs.type(Tensor))
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], config.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i % config.print_freq == 0:
                logging.info(
                    f'Epoch [{epoch}][{i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]'
                )
            if i % 100 == 0:
                save_image(gen_imgs.data[:25], epoch, i)


if __name__ == "__main__":
    config = parser_args()
    logging.info(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_shape = (config.in_channel, config.img_size, config.img_size)
    generator = Generator(config, img_shape).to(device)
    discriminator = Discriminator(config, img_shape).to(device)

    logging.info(generator)
    logging.info(discriminator)

    dataset = Mnist(config, 'train')
    loader = Data.DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.workers, 
        pin_memory=True)

    main(generator, discriminator, loader, device, config)
