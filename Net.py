from tools.layers.softmax import softmax
class Net(Layer):
    def __init__(self, layer_configures):
        self.layers = []
        for config in layer_configures:
            self.layers.append(self.createLayer(config))
        
    def createLayer(self, config):
        return self.getDefaultLayer(config)

    def getDefaultLayer(self, config):
        t = config['type']
        if t=='linear':
            layer = Linear(**config)
            self.parameters.append(layer.W)
            if layer.b is not None: self.parameters.append(layer.b)
        elif t == 'Relu':
            layer = ReLU()
        elif t == 'softmax':
            layer = softmax()
        else:
            raise TypeError
        return layer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, eta):
        for layer in self.layers:
            eta = layer.backward(eta)
        return eta