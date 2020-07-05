class SGD(object):
    def __init__(self, parameters, lr):
        self.parameters = []
        for p in parameters:
            if p.requires_grade:
                self.parameters.append(p)
        # self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr

    def update(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad