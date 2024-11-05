# NanoGrad

A tiny Autograd engine inspired by [Micrograd](https://github.com/karpathy/micrograd). Implements backpropagation over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification. Potentially useful for educational purposes.


### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from nanograd import Tensor

a = Tensor(-4.0)
b = Tensor(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

### License

Apache 2.0