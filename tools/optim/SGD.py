class SGD(object):
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr
    
    def update(self):
        for p in self.parameters:
            if not p.requires_grad:continue
            p.data -= self.lr*p.grad