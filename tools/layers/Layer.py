from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def backward(self, *args):
        pass
