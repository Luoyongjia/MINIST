from .Layer import Layer
from .Parameter import Parameter
import numpy as np


class Linear(Layer):
    def __init__(self, shape, requires_grad=True, bias=True, **kwargs):
        W = np.random.randn(*shape) * (2 / shape[0] ** 0.5)
        self.W = Parameter(W, requires_grad)
        if bias:
            self.b = Parameter(np.zeros(shape[-1]), requires_grad)
        self.requires_grad = requires_grad

    def forward(self, x):
        if self.requires_grad:
            self.x = x
            out = np.dot(x, self.W.data)
            if self.b is not None:
                out = out + self.b.data
        return out

    def backward(self, eta):
        if self.requires_grad:
            self.W.grad = np.dot(self.x.T, eta)
            if self.b is not None:
                self.b.grad = np.einsum('i...->...', eta, optimize=True) / eta.shape[0]
        return np.dot(eta, self.W.T)
