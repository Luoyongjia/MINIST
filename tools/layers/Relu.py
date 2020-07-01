from .Layer import Layer
import numpy as np

class Relu(Layer):
    def forward(self, x):
        self.x = x
        return np.maximum(0,x)
    
    def backward(self, eta):
        eta[self.x < 0] = 0
        return eta