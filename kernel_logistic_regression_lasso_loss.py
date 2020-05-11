import numpy as np
from math import log, exp
from MnistData import load_mnist_by_class
import os

TRAIN_SAMPLES = 1000
TEST_SAMPLES = 1000
PIXELS = 28 * 28

def train(x, kernel, y, c, regular, lr, epoch, num_samples=TRAIN_SAMPLES):
    assert x.shape[0] == c.shape[0]
    gradient = np.zeros(c.shape)
    for i in range(num_samples):
        value = (c * kernel[i]).sum()
        gradient = gradient + (-1 * y[i] * kernel[i]) / (1 + exp(y[i] * value))
    lasso_term = (c > 0).astype(np.int32)
    lasso_term = lasso_term * 2 - 1
    gradient = gradient + regular * lasso_term
    c = c - lr * gradient

    weight = np.zeros(x.shape[1])
    for i in range(num_samples):
        weight = weight + c[i] * x[i]

    return c, weight

def test(x, y, weight, epoch, num_samples=TEST_SAMPLES):
    num_correct = 0
    for i in range(num_samples):
        value = (x[i] * weight).sum()
        probability = 1 / (1 + exp(-1 * value))
        if probability > 0.5 and y[i] == 1:
            num_correct += 1
        elif probability <= 0.5 and y[i] == -1:
            num_correct += 1
    
    accuracy = num_correct / num_samples
    # print('Epoch[%d]'%epoch, accuracy)
    return accuracy

def multi_test(x, y, weights, num_samples=TEST_SAMPLES):
    num_correct = 0
    for i in range(num_samples):
        best_p = 0.0
        pred_digit = 0
        for j in range(10):
            value = (x[i] * weights[j]).sum()
            probability = 1 / (1 + exp(-1 * value))
            if probability > best_p:
                pred_digit = j
                best_p = probability
        if pred_digit == y[i]:
            num_correct += 1
    
    return num_correct / num_samples

def construct_kernel_matrix(x, kernel_function='rbf', sigma=1, d=2):
    n_samples = x.shape[0]
    kernel = np.zeros((n_samples, n_samples))
    if kernel_function == 'rbf':
        for i in range(n_samples):
            for j in range(i, n_samples):
                norm = np.linalg.norm(x[i] - x[j]) ** 2
                value = exp(-1 * norm / (2 * (sigma ** 2)))
                kernel[i][j] = value
                kernel[j][i] = value
    
    elif kernel_function == 'poly':
        for i in range(n_samples):
            for j in range(i, n_samples):
                norm = (x[i] * x[j]).sum()
                value = norm ** d
                kernel[i][j] = value
                kernel[j][i] = value

    elif kernel_function == 'cos':
        for i in range(n_samples):
            for j in range(i, n_samples):
                norm1 = np.linalg.norm(x[i])
                norm2 = np.linalg.norm(x[j])
                value = (x[i] * x[j]).sum() / (norm1 * norm2)
                kernel[i][j] = value
                kernel[j][i] = value
    
    return kernel

def construct_kernel_x(x, kernel_function='cos'):
    n_samples = x.shape[0]
    for i in range(n_samples):
        x[i] = x[i] / np.linalg.norm(x[i])

    return x

def main():
    # load data
    x_train = np.zeros((TRAIN_SAMPLES, PIXELS))
    y_train = np.zeros((TRAIN_SAMPLES))
    x_test = np.zeros((TEST_SAMPLES, PIXELS))
    y_test = np.zeros((TEST_SAMPLES))
    for digit in range(10):
        x, y = load_mnist_by_class(digit, TRAIN_SAMPLES / 10)
        x_train[digit * TRAIN_SAMPLES // 10: (digit + 1) * TRAIN_SAMPLES // 10] = x
        y_train[digit * TRAIN_SAMPLES // 10: (digit + 1) * TRAIN_SAMPLES // 10] = y
        x, y = load_mnist_by_class(digit, TEST_SAMPLES / 10, kind='t10k')
        x_test[digit * TEST_SAMPLES // 10: (digit + 1) * TEST_SAMPLES // 10] = x
        y_test[digit * TEST_SAMPLES // 10: (digit + 1) * TEST_SAMPLES // 10] = y

    # normalize
    x_train = (x_train / 255).astype(np.float32)
    x_test = (x_test / 255).astype(np.float32)
    kernel_matrix = construct_kernel_matrix(x_train, kernel_function='cos')
    x_train = construct_kernel_x(x_train)
    x_test = construct_kernel_x(x_test)
    print('Data preparation complete!')

    # hyper parameters
    lr = 0.001
    epoch = 50
    print_freq = 10
    regular = 0.5
    model_prefix = 'weights/kernel_regression'
    if not os.path.isdir(model_prefix):
        os.makedirs(model_prefix)

    weights = []
    # train each single digit
    for j in range(10):
        print('Training for digit %d' % j)
        weight_file = os.path.join(model_prefix, '%d.npy'%j)
        c = np.zeros(kernel_matrix.shape[0])
        weight = np.zeros(x_train.shape[1])
        best_accuracy = 0
        best_weight = weight
        # deal with soft label
        target_digit = j
        label_train = (y_train == target_digit).astype(np.int32)
        label_test = (y_test == target_digit).astype(np.int32)
        label_train = label_train * 2 - 1
        label_test = label_test * 2 - 1

        for i in range(epoch):
            c, weight = train(x_train, kernel_matrix, label_train, c, regular, lr, i)
            accuracy = test(x_test, label_test, weight, i)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weight = weight
            if i > 0 and i % print_freq == 0:
                print('Epoch[%d/%d]'%(i, epoch), 'current accuracy: %.4f'%accuracy)
        
        weights.append(best_weight)
        np.save(weight_file, best_weight)
        print('Finish training for digit %d, best accuracy %.4f\n' % (j, best_accuracy))
    
    final_accuracy = multi_test(x_test, y_test, weights)
    print('Final accuracy:', final_accuracy)


if __name__ == "__main__":
    main()