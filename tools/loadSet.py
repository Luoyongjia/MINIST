import numpy as np


def onehot(targets, num):
    result = np.zeros((num, 10))
    for i in range(num):
        result[i][targets[i]] = 1
    return result


def load_data(path = '/Users/luoyongjia/Program/py/MNIST/mnist/mnist.npz'):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()

    x_train = x_train.reshape(60000, 28*28) / 255.
    x_test = x_test.reshape(10000, 28*28) / 255.

    y_train = onehot(y_train, 60000)
    y_test = onehot(y_test, 10000)

    return x_train, y_train, x_test, y_test