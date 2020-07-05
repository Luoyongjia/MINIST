import tools
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def save(parameters, save_as):
    dic={}
    for i in range(len(parameters)):
        dic[str[i]] = parameters[i].data
    np.savez(save_as, **dic)


def load(parameters, file):
    params  = np.load(file)
    for i in range(len(parameters)):
        parameters[i].data = params[str(i)]


def train(net, loss_fun, x_train, y_train, batch_size, optimizer, load_file, save_as, times = 1, retrain=False):
    X = x_train
    Y = y_train
    loss_val = []
    acc_val = []
    data_size = X.shape[0]
    if not retrain and os.path.isfile(load_file): load(net.parameters, load_file)
    for loop in range(times):
        i = 0
        while i <= data_size - batch_size:
            x = X[i:i+batch_size]
            y = Y[i:i+batch_size]
            i += batch_size

            output = net.forward(x)
            batch_acc, batch_loss = loss_fun(output, y)
            eta = loss_fun.gradient()
            net.backward(eta)
            optimizer.update()
            loss_val.append(batch_loss)
            acc_val.append(batch_acc)
            if i%50 == 0:
                print("loop: %d, batch: %5d, batch acc: %2.1f, batch loss: %.2f" % \
                      (loop, i, batch_acc * 100, batch_loss))

        pass
    if save_as is not None:
        save(net.parameters, save_as)

    return acc_val,loss_val



def test(net, x_test, y_test, loss_fun):
    X = x_test
    Y = y_test
    output = net.forward(X)
    acc, _ = loss_fun(output, Y)

    print(acc)


if __name__ == "__main__":
    netStructure=[
        {'type': 'Linear', 'shape': (784, 200)},
        {'type': 'Relu'},
        {'type': 'Linear', 'shape': (200, 100)},
        {'type': 'Relu'},
        {'type': 'Linear', 'shape': (100, 10)}
    ]
    x_train, y_train, x_test, y_test = tools.load_data()
    loss_fn = tools.loss.crossEntropyLoss()
    lr = 0.001
    batch_size = 128
    net = tools.Net(netStructure)
    optimizer = tools.optim.SGD(net.parameters, lr)
    acc, loss = train(net, loss_fn, x_train, y_train, batch_size, optimizer, None, None, times=3, retrain=True)

    plt.plot(range(len(loss)), loss)
    plt.show()
    plt.plot(range(len(acc)), acc)
    plt.show()

    test(net, x_test, y_test, loss_fn)