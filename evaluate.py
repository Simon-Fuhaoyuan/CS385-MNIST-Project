import numpy as np

def accuracy(output, target, meta=None, fp=None):
    '''
    Calculate accuracy.
    '''
    n_correct = 0
    cnt = 0
    output = output.cpu().detach().numpy()

    for i in range(target.shape[0]):
        one_hot = output[i]
        max_cat = np.argmax(one_hot)
        cnt += 1
        if max_cat == target[i]:
            n_correct = n_correct + 1

    return n_correct, cnt