import math


def add_backward(node):
    """Backpropogation function for addition

    :param node: Scalar node
    :type node: nanograd.Scalar
    """
    for child in node._children:
        child.grad += 1 * node.grad


def subtract_backward(node):
    """Backpropogation function for subtraction

    :param node: Scalar node
    :type node: nanograd.Scalar
    """
    ch1, ch2 = node._children
    ch1.grad += 1 * node.grad
    ch2.grad += -1 * node.grad


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
    """Backpropogation function for exponent

    :param node: Scalar node
    :type node: nanograd.Scalar
    """
    child, = node._children
    child.grad += node.data * node.grad


def pow_backward(node):
    """Backpropogation function for power

    :param node: Scalar node
    :type node: nanograd.Scalar
    """
    child, power = node._children
    child.grad += power.data * (child.data ** (power.data-1)) * node.grad


def relu_backward(node):
    """Backpropogation function for relu activation function

    :param node: Scalar node
    :type node: nanograd.Scalar
    """
    child, = node._children
    child.grad += (child.data > 0 and 1 or 0) * node.grad


def log_backward(node):
    """Backpropogation function for log

    :param node: Scalar node
    :type node: nanograd.Scalar
    """
    child, = node._children
    child.grad += (1/child.data) * node.grad


def abs_backward(node):
    """Backpropogation function for absolute value

    :param node: Scalar node
    :type node: nanograd.Scalar
    """
    child, = node._children
    child.grad += (child.data >= 0 and 1 or -1) * node.grad
