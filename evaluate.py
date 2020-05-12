import numpy as np

def accuracy(output, target, meta=None, fp=None):
    '''
    Calculate accuracy.
    '''
    n_correct = 0
    cnt = 0

    for i in range(len(acc)):
        one_hot = output[i]
        max_cat = np.argmax(one_hot)
        cnt += 1
        if max_cat == target[i]:
            n_correct = n_correct + 1

    return n_correct, cnt