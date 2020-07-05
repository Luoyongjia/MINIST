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


class Sigmoid(Layer):
    def forward(self, x):
        self.y = 1/(1+np.exp(-x))
        return self.y

    def backward(self, eta):
        return np.einsum('...,...,...->...', self.y, 1 - self.y, eta, optimize=True)


class Tanh(Layer):
    def forward(self, x):
        ex = np.exp(x)
        esx = np.exp(-x)
        self.y = (ex - esx) / (ex + esx)
        return self.y

    def backward(self, eta):
        return np.einsum('...,...,...->...', 1 - self.y, 1 + self.y, eta, optimize=True)