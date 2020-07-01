import numpy as np

class Parameter(object):
    def _init_(self, shape, requires_grade):
        if isinstance(shape, int):
            self.data = np.zeros(shape)
        elif len(shape) == 2:
            self.data = np.random.randn(*shape) *2 / shape[0]
        
        self.grad = None
        self.requires_grade = requires_grade


    @property
    def T(self):
        return self.data.T