import numpy as np
from MnistData import load_mnist
import matplotlib.pyplot as plt
import random
import copy

imgs, _ = load_mnist()
imgs = imgs[:15]

fig, ax = plt.subplots(
    nrows=3,
    ncols=5,
    sharex=True,
    sharey=True, )

ax = ax.flatten()
for k in range(15):
    img = imgs[k].reshape(28, 28, 1)
    buffer = []
    block_num = 4
    block_size = 28 // block_num
    for i in range(block_num):
        for j in range(block_num):
            block = copy.deepcopy(img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size])
            buffer.append(block)
    random.shuffle(buffer)
    # img = np.zeros((28,28))
    for i in range(block_num):
        for j in range(block_num):
            img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = buffer[block_num * i + j]
    
    img = img.reshape(28,28)
    ax[k].imshow(img, cmap='Greys', interpolation='nearest')

# img = imgs[0].reshape(28, 28)
# buffer = []
# block_num = 4
# block_size = 28 // block_num
# print(block_size)
# for i in range(block_num):
#     for j in range(block_num):
#         buffer.append(img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size])

# for i in range(16):
#     ax[i].imshow(buffer[i], cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.show()
plt.savefig('images/reshape.png')
