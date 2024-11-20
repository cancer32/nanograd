import math


def add_backward(node):
    """Backpropogation function for addition

    :param node: Scalar node
    :type node: nanograd.Scalar
    """
    for child in node._children:
        child.grad += 1 * node.grad


def multipy_backward(node):
    """Backpropogation function for multiplication

    :param node: Scalar node
    :type node: nanograd.Scalar
    """
    ch1, ch2 = node._children
    ch1.grad += ch2.data * node.grad
    ch2.grad += ch1.data * node.grad


def truediv_backward(node):
    """Backpropogation function for division

    :param node: Scalar node
    :type node: nanograd.Scalar
    """
    ch1, ch2 = node._children
    ch1.grad += (1/ch2.data) * node.grad
    ch2.grad += (-ch1.data/(ch2.data**2)) * node.grad


def tanh_backward(node):
    """Backpropogation function for tanh

    :param node: Scalar node
    :type node: nanograd.Scalar
    """
    child, = node._children
    child.grad += (1 - node.data**2) * node.grad


def exp_backward(node):
    """Backpropogation function for tanh

    :param node: Scalar node
    :type node: nanograd.Scalar
    """
    child, = node._children
    child.grad += node.data