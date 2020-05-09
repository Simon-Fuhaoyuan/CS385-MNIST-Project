import numpy as np
from math import log, exp
from MnistData import load_mnist
import os

TRAIN_SAMPLES = 60000
TEST_SAMPLES = 10000
PIXELS = 28 * 28

def train(x, y, weight, lr, epoch, num_samples=TRAIN_SAMPLES):
    log_likelihood = 0.0
    gradient = np.zeros(PIXELS)
    for i in range(num_samples):
        value = (x[i] * weight).sum()
        log_likelihood += y[i] * value - log(1 + exp(value))
        probability = 1 / (1 + exp(-1 * value))
        gradient = gradient + (y[i] - probability) * x[i]

    # print('Epoch[%d]'%epoch, log_likelihood)
    weight = weight + lr * gradient
    return weight

def test(x, y, weight, epoch, num_samples=TEST_SAMPLES):
    num_correct = 0
    for i in range(num_samples):
        value = (x[i] * weight).sum()
        probability = 1 / (1 + exp(-1 * value))
        if probability > 0.5 and y[i] == 1:
            num_correct += 1
        elif probability < 0.5 and y[i] == 0:
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

def main():
    x_train, y_train = load_mnist()
    x_test, y_test = load_mnist(kind='t10k')
    # normalize
    x_train = (x_train / 255).astype(np.float32)
    x_test = (x_test / 255).astype(np.float32)
    
    # hyper parameters
    lr = 0.0001
    epoch = 50
    print_freq = 10
    model_prefix = 'weights/logistic_regression'
    if not os.path.isdir(model_prefix):
        os.makedirs(model_prefix)

    weights = []
    # train each single digit
    for j in range(10):
        print('Training for digit %d' % j)
        weight_file = os.path.join(model_prefix, '%d.npy'%j)
        weight = np.zeros(PIXELS)
        best_accuracy = 0
        best_weight = np.zeros(PIXELS)
        # deal with soft label
        target_digit = j
        label_train = (y_train == target_digit).astype(np.uint8)
        label_test = (y_test == target_digit).astype(np.uint8)

        for i in range(epoch):
            weight = train(x_train, label_train, weight, lr, i)
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