import numpy as np
import tools
import tools.optim as opt

def train(net, loss_fn, x_train, y_train, batch_size, optimizer, times = 1):
    data_size = x_train.shape[0]
    for loop in range(times):
        i = 0
        while i <= data_size - batch_size:
            x = x_train[i : i + batch_size]
            y = y_train[i : i + batch_size]
            i += batch_size

            output = net.forward(x)
            batch_acc, batch_loss = loss_fn(output, y)
            eta = loss_fn.gradient()
            net.backward(eta)
            optimizer.update()

if __name__== "__main__":
    layers = [
        {'type':'linear','shape':(784, 200)},
        {'type':'relu'},
        {'type':'linear', 'shape':(200,100)},
        {'type':'relu'},
        {'type':'linear','shape':(100, 10)}
    ]
    loos_fn = tools.crossEntropyLoss()
    