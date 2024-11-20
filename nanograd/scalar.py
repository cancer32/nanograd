__all__ = ["Scalar"]

import math

from . import gradient
from . import vizualize


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

    @staticmethod
    def grad_fn(node):
        pass

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
        ret.grad_fn = gradient.add_backward
        return ret

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self.new(other)
        ret = Scalar(self.data * other.data,
                     _children=(self, other),
                     _op='*')
        ret.grad_fn = gradient.multipy_backward
        return ret

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = self.new(other)
        ret = Scalar(self.data / other.data,
                     _children=(self, other),
                     _op='/')
        ret.grad_fn = gradient.truediv_backward
        return ret

    def __rtruediv__(self, other):
        other = self.new(other)
        return other / self

    def exp(self):
        ret = Scalar(math.exp(self.data),
                     _children=(self,),
                     _op='exp')
        ret.grad_fn = gradient.exp_backward
        return ret

    def __pow__(self, power):
        power = self.new(power)
        ret = Scalar(self.data**power.data,
                     _children=(self, power),
                     _op='**')
        ret.grad_fn = gradient.pow_backward
        return ret

    def tanh(self):
        ret = Scalar((math.exp(2*self.data) - 1)/(math.exp(2*self.data) + 1),
                     _children=(self,),
                     _op='tanh')
        ret.grad_fn = gradient.tanh_backward
        return ret

    def item(self):
        """Returns the original scalar value
        """
        return self.data

    def nodes(self):
        """Retuns all the expression nodes from parent to child

        :return: list of nodes
        :rtype: list
        """
        nodes = []
        found = set()

        def traverse(node):
            if node in found:
                return
            found.add(node)
            for child in node._children:
                traverse(child)
            nodes.append(node)

        return reversed(nodes)

    def backward(self):
        """Backward function to calculate the gradient
        """
        self.grad = 1.0
        for child in self.nodes():
            child.grad_fn(child)

    def zero_grad(self):
        """Make grad zero for current and all the child nodes
        """
        self.grad = 0.0
        for child in self.nodes():
            child.grad = 0.0

    def draw(self):
        vizualize.draw_dot(self)
