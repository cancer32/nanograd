__all__ = ["Scalar"]

import math


class Scalar(object):
    def __init__(self, data, label=None, _children=None, _op=None):
        self.data = data
        self.label = label or ''
        self._children = _children or ()
        self._op = _op or ''
        self.grad = 0.0

    def __repr__(self):
        return f"Scalar({self.data})"

    @staticmethod
    def new(data):
        """Returns a Scalar wrapper of the data

        :param data: scalar value
        :type data: object
        :return: Scalar wrapper
        :rtype: nanograd.Scalar
        """
        if not isinstance(data, Scalar):
            data = Scalar(data)
        return data

    def __eq__(self, other):
        other = self.new(other)
        return self.data == other.data

    def __lt__(self, other):
        other = self.new(other)
        return self.data < other.data

    def __le__(self, other):
        other = self.new(other)
        return self.data <= other.data

    def __add__(self, other):
        other = self.new(other)
        ret = Scalar(self.data + other.data,
                     _children=(self, other),
                     _op='+')
        return ret

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self.new(other)
        ret = Scalar(self.data * other.data,
                     _children=(self, other),
                     _op='*')
        return ret

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = self.new(other)
        ret = Scalar(self.data / other.data,
                     _children=(self, other),
                     _op='/')
        return ret

    def __rtruediv__(self, other):
        other = self.new(other)
        return other / self

    def exp(self):
        ret = Scalar(math.exp(self.data),
                     _children=(self,),
                     _op='exp')
        return ret

    def tanh(self):
        ret = Scalar((math.exp(2*self.data) - 1)/(math.exp(2*self.data) + 1),
                     _children=(self,),
                     _op='tanh')
        return ret

    def item(self):
        """Returns the original scalar value
        """
        return self.data
