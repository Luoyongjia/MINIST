import numpy as np
from .softmax import softmax

class crossEntropy():
    def __init__(self):
        self.classifier = softmax()
    
    def backward(self):
        return self.classifier.a - self.y
        
    def __call__(self, a, y):
        a = self.classifier.forward(a)
        self.y = y
        loss = np.einsum('ij, ij->', y,np.log(a),optimize=True) / y.shape[0]
        return -loss
