import unittest

import nanograd
from nanograd import gradient


class GradientTest(unittest.TestCase):
    def test_add_backward(self):
        a = nanograd.Scalar(1.5)
        b = nanograd.Scalar(3.14)
        c = a + b
        c.grad = 2.0
        gradient.add_backward(c)
        self.assertEqual((a.grad, b.grad), (2, 2),
                         'Failed add backward test')

    def test_mul_backward(self):
        a = nanograd.Scalar(1.5)
        b = nanograd.Scalar(3.14)
        c = a * b
        c.grad = 2.0
        gradient.multipy_backward(c)
        self.assertEqual((a.grad, b.grad), (6.28, 3.0),
                         'Failed mul backward test')

    def test_truediv_backward(self):
        a = nanograd.Scalar(5)
        b = nanograd.Scalar(2)
        c = a / b
        c.grad = 2.0
        gradient.truediv_backward(c)
        self.assertEqual((a.grad, b.grad), (1.0, -2.5),
                         'Failed mul backward test')

    def test_tanh_backward(self):
        a = nanograd.Scalar(100.0)
        b = a.tanh()
        b.grad = 1.0
        gradient.tanh_backward(b)
        self.assertEqual(a.grad, (1-(b.item()**2)),
                         'Failed tanh backward test')

    def test_exp_backward(self):
        a = nanograd.Scalar(100.0)
        b = a.exp()
        b.grad = 1.0
        gradient.exp_backward(b)
        self.assertEqual(a.grad, b,
                         'Failed exp backward test')

    def test_pow_backward(self):
        a = nanograd.Scalar(1.314)
        b = a ** 3
        b.grad = 2.0
        gradient.pow_backward(b)
        self.assertEqual(a.grad, 3 * (a.data ** 2) * 2,
                         'Failed pow backward test')
