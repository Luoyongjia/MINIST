import tools
import numpy as np
import matplotlib.pyplot as plt
import os
import time

#存储网络中的参数
def save(parameters, save_as):
    dic={}
    for i in range(len(parameters)):
        dic[str[i]] = parameters[i].data
    np.savez(save_as, **dic)

#读取参数
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
    #读取
    if not retrain and os.path.isfile(load_file): load(net.parameters, load_file)
    starttime = time.time()

    for loop in range(times):
        i = 0
        while i <= data_size - batch_size:
            x = X[i:i+batch_size]
            y = Y[i:i+batch_size]
            i += batch_size

            #训练部分
            output = net.forward(x)
            batch_acc, batch_loss = loss_fun(output, y)
            eta = loss_fun.gradient()
            net.backward(eta)
            optimizer.update()
            loss_val.append(batch_loss)
            acc_val.append(batch_acc)
            if i%50 == 0:
                print("loop: %d, batch: %5d, batch acc: %2.1f, batch loss: %.2f" % \
                      (loop + 1, i, batch_acc * 100, batch_loss))

        pass
    #记录训练时间
    endtime = time.time()
    print("Used ", round(endtime - starttime, 2)," secs")
    if save_as is not None:
        save(net.parameters, save_as)

    return acc_val, loss_val



def test(net, x_test, y_test, loss_fun):
    X = x_test
    Y = y_test
    output = net.forward(X)
    acc, _ = loss_fun(output, Y)

    print(acc)


if __name__ == "__main__":
    netSum = []     #所有的网络合集

    netStructure_Sigmoid = [
        {'type': 'Linear', 'shape': (784, 200)},
        {'type': 'Sigmoid'},
        {'type': 'Linear', 'shape': (200, 100)},
        {'type': 'Sigmoid'},
        {'type': 'Linear', 'shape': (100, 10)}
    ]
    netStructure_Tanh = [
        {'type': 'Linear', 'shape': (784, 200)},
        {'type': 'Tanh'},
        {'type': 'Linear', 'shape': (200, 100)},
        {'type': 'Tanh'},
        {'type': 'Linear', 'shape': (100, 10)}
    ]
    netStructure_Relu=[
        {'type': 'Linear', 'shape': (784, 200)},
        {'type': 'Relu'},
        {'type': 'Linear', 'shape': (200, 100)},
        {'type': 'Relu'},
        {'type': 'Linear', 'shape': (100, 10)}
    ]
    netSum.append(netStructure_Sigmoid)
    netSum.append(netStructure_Tanh)
    netSum.append(netStructure_Relu)

    #读取npz中的数据
    x_train, y_train, x_test, y_test = tools.load_data()

    #loss, lr, batch size设置
    loss_fn = tools.loss.crossEntropyLoss()
    lr = 0.001
    batch_size = 128

    for net in netSum:
        net = tools.Net(net)
        optimizer = tools.optim.SGD(net.parameters, lr)
        acc, loss = train(net, loss_fn, x_train, y_train, batch_size, optimizer, None, None, times=1, retrain=True)

        plt.plot(range(len(loss)), loss)
        plt.show()
        plt.plot(range(len(acc)), acc)
        plt.show()

        test(net, x_test, y_test, loss_fn)