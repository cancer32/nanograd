import math


def add_backward(node):
    for child in node._children:
        child.grad += 1 * node.grad

def multipy_backward(node):
    ch1, ch2 = node._children
    ch1.grad += ch2.data * node.grad
    ch2.grad += ch1.data * node.grad

def tanh_backward(node):
    child, = node._children
    child.grad += (1 - node.data**2) * node.grad
