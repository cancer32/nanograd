import random

from .scalar import Scalar


class Neuron:
    def __init__(self, n_inputs):
        self.weights = [Scalar(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Scalar(random.uniform(-1, 1))

    def __call__(self, x):
        logits = sum((xi*wi for xi, wi in zip(x, self.weights)), self.bias)
        return logits.tanh()

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs=n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_inputs, n_layers):
        _n_layers = [n_inputs] + n_layers
        self.layers = [Layer(n_inputs=_n_layers[i], n_outputs=_n_layers[i+1])
                       for i in range(len(n_layers))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0