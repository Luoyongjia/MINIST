from .layers import *


class Net(Layer):
    def __init__(self, layer_configures):
        self.layers = []
        self.parameters = []

        #构建网络，初始化
        for config in layer_configures:
            self.layers.append(self.createLayer(config))

    def createLayer(self, config):
        return self.getDefaultLayer(config)

    #创建不同种类的单层网络
    def getDefaultLayer(self, config):
        t = config['type']
        if t == 'Linear':
            layer = Linear(**config)
            self.parameters.append(layer.W)
            if layer.b is not None: self.parameters.append(layer.b)
        elif t == 'Relu':
            layer = Relu()
        elif t == 'Softmax':
            layer = Softmax()
        elif t == 'Sigmoid':
            layer = Sigmoid()
        elif t == 'Tanh':
            layer = Tanh()
        else:
            raise TypeError
        return layer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, eta):
        for layer in self.layers[::-1]:
            eta = layer.backward(eta)
        return eta