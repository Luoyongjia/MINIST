from .Layer import Layer
import numpy as np

class Relu(Layer):
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, eta):
        eta[self.x <= 0] = 0
        return eta

class Softmax(Layer):
    def forward(self, x):
        v = np.exp(x - x.max(axis=-1, keepdims=True))
        return v / v.sum(axis=-1, keepdims=True)

    def backward(self, y):
        pass