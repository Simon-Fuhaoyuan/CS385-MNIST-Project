from sklearn import svm
import numpy as np
from MnistData import load_mnist

x_train, y_train = load_mnist()
x_test, y_test = load_mnist(kind='t10k')
train_num = 10000
test_num = 1000


predictor = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
predictor.fit(x_train[:train_num], y_train[:train_num])
result = predictor.predict(x_test[:test_num])
accurancy = np.sum(np.equal(result, y_test[:test_num])) / test_num
print(accurancy)