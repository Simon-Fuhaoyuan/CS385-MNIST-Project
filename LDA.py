import numpy as np
from math import log, exp
from MnistData import load_mnist
import os

TRAIN_SAMPLES = 60000
TEST_SAMPLES = 10000
PIXELS = 28 * 28


def test(x, y, weight, bias, num_samples=TEST_SAMPLES):
    pred = np.dot(x, weight.T) + bias
    pred = (pred > 0).astype(np.uint8)
    num_correct = (pred == y).sum()
    ## recall
    recall = 1 - (pred < y).sum() / y.sum()

    return num_correct / num_samples, recall

def multi_test(x, y, weights, bias, num_samples=TEST_SAMPLES):
    num_correct = 0
    for i in range(num_samples):
        best_score = 0.0
        pred_digit = 0
        for j in range(10):
            score = (x[i] * weights[j]).sum() + bias[j]
            if score > best_score:
                pred_digit = j
                best_score = score
        if pred_digit == y[i]:
            num_correct += 1
    
    return num_correct / num_samples

def cal_LDA_bias(digit, x, y, weight, lamb=0):
    x_pos = x[y == digit]
    x_neg = x[y != digit]
    pred_pos = np.dot(x_pos, weight.T)
    pred_neg = np.dot(x_neg, weight.T)

    mean_pos = pred_pos.mean()
    mean_neg = pred_neg.mean()
    std_pos = np.std(pred_pos)
    std_neg = np.std(pred_neg)

    if lamb < 0:
        lamb = std_neg / (std_neg + std_pos)
    bias = -1 * (lamb * mean_pos + (1 - lamb) * mean_neg)

    return bias

def cal_LDA_weight(digit, x, y):
    pos_samples = x[y == digit]
    neg_samples = x[y != digit]
    n_pos = pos_samples.shape[0]
    n_neg = neg_samples.shape[0]
    mean_pos = np.mean(pos_samples, axis=0)
    mean_neg = np.mean(neg_samples, axis=0)
    var_pos = np.var(pos_samples, axis=0) * n_pos
    var_neg = np.var(neg_samples, axis=0) * n_neg
    Sw = np.zeros((PIXELS, PIXELS))
    for i in range(PIXELS):
        value = var_pos[i] + var_neg[i]
        if value > 0:
            Sw[i][i] = 1 / value
    weight = np.dot(Sw, (mean_pos - mean_neg).T)

    return weight

def main():
    x_train, y_train = load_mnist()
    x_test, y_test = load_mnist(kind='t10k')
    # normalize
    x_train = (x_train / 255).astype(np.float32)
    x_test = (x_test / 255).astype(np.float32)
    
    # hyper parameters
    model_prefix = 'weights/LDA'
    if not os.path.isdir(model_prefix):
        os.makedirs(model_prefix)

    weights = []
    biases = []
    # train each single digit
    for j in range(10):
        print('Training for digit %d' % j)
        weight_file = os.path.join(model_prefix, '%d.npy'%j)
        # deal with soft label
        target_digit = j
        label_test = (y_test == target_digit).astype(np.uint8)
        weight = cal_LDA_weight(target_digit, x_train, y_train)
        bias = cal_LDA_bias(target_digit, x_train, y_train, weight, lamb=0.6)
        accuracy, recall = test(x_test, label_test, weight, bias)
        # print('acc', accuracy)
        print('rec', recall)

        weights.append(weight)
        biases.append(bias)
        np.save(weight_file, weight)
    
    final_accuracy = multi_test(x_test, y_test, weights, biases)
    print('Final accuracy:', final_accuracy)

if __name__ == "__main__":
    main()