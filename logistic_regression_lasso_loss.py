import numpy as np
from math import log, exp
from MnistData import load_mnist
import os

TRAIN_SAMPLES = 60000
TEST_SAMPLES = 10000
PIXELS = 28 * 28

def train(x, y, weight, regular, lr, epoch, num_samples=TRAIN_SAMPLES):
    gradient = np.zeros(PIXELS)
    for i in range(num_samples):
        value = (x[i] * weight).sum()
        gradient = gradient + (-1 * y[i] * x[i]) / (1 + exp(y[i] * value))
    lasso_term = (weight > 0).astype(np.int32)
    lasso_term = lasso_term * 2 - 1
    gradient = gradient + regular * lasso_term

    weight = weight - lr * gradient
    return weight

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

def main():
    x_train, y_train = load_mnist()
    x_test, y_test = load_mnist(kind='t10k')
    # normalize
    x_train = (x_train / 255).astype(np.float32)
    x_test = (x_test / 255).astype(np.float32)
    
    # hyper parameters
    lr = 0.0001
    regular = 0.5
    epoch = 50
    print_freq = 10
    batch_size = 60000
    model_prefix = 'weights/logistic_regression_lasso_loss'
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
        # deal with soft label, label is +1 or -1
        target_digit = j
        label_train = (y_train == target_digit).astype(np.int32)
        label_test = (y_test == target_digit).astype(np.int32)
        label_train = label_train * 2 - 1
        label_test = label_test * 2 - 1

        for i in range(epoch):
            for batch in range(TRAIN_SAMPLES // batch_size):
                weight = train(
                    x_train[batch * batch_size:], 
                    label_train[batch * batch_size:], 
                    weight, 
                    regular, 
                    lr, 
                    i,
                    num_samples=batch_size)
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