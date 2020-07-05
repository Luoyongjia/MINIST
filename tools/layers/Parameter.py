class Parameter(object):
    def __init__(self, data, requires_grade, skip_decay=False):
        self.data = data
        self.grad = None
        self.skip_decay = skip_decay
        self.requires_grade = requires_grade

    @property
    def T(self):
        return self.data.T
