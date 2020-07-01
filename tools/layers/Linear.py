from .Layer import Layer
from .Parameter import Parameter
import numpy as np

class Linear(Layer):
    def _init_(self, shape, require_grad=True, bias = True, **kwargs):
        self.W = Parameter(shape, require_grad)
        self.b = Parameter(shape[-1], require_grad) if bias else None
        self.require_grad = require_grad

    def forward(self, x):
        if self.require_grad: self.x = x
        out = np.dot(x, self.W.data)
        if self.b is not None: out = out + self.b.data
        return out

    def backward(self, eta):
        if self.require_grad:
            self.W.grad = np.dot(self.x.T, eta)
            if self.b is not None: self.b.grad = eta
        return np.dot(eta, self.W.T)
