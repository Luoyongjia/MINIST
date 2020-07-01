from .Layer import Layer
import numpy as np

class softmax(Layer):
    def forward(self, x):
        v = np.exp(x - x.max(axis = -1, keepdims=True))
        self.a = v / v.sum(axis = -1, keepdims = True)
        return self.a
    
    def backward(self, eta):
        return self.a*(eta - np.einsum('ij, ij->i', eta, self.a, optimize = True))

