import unittest
import nanograd

class TensorTest(unittest.TestCase):
    def test_attributes(self):
        # data attribute test
        data = -2.0
        t1 = nanograd.Tensor(data)
        self.assertEqual(t1.data, data,
                         'Failed Tensor.data attribute test')

        # data attribute test
        data = -2.0
        t1 = nanograd.Tensor(data)
        self.assertEqual(t1.data, data,
                         'Failed Tensor.data attribute test')

    def test_item(self):
        data = 3.14
        t1 = nanograd.Tensor(data)
        self.assertEqual(t1.item(), data,
                         'Failed item() method test')

    def test__repr__(self):
        # Repr test
        data = -3.14
        return_str = f'Tensor({data})'
        t1 = nanograd.Tensor(data)
        self.assertEqual(t1.__repr__(), return_str,
                         'Failed Tensor.__repr__() test')
        
    def test_add(self):
        # Adding two Tensors
        a = nanograd.Tensor(3.0)
        b = nanograd.Tensor(2.0)
        self.assertEqual(a + b, nanograd.Tensor(5.0),
                         'Failed addition test')
        # Adding Tensor with scalar value
        a = nanograd.Tensor(3.0)
        b = 2.0
        self.assertEqual(a + b, nanograd.Tensor(5.0),
                         'Failed addition with scalar test')

    def test_muliply(self):
        # Muliplying two Tensors
        a = nanograd.Tensor(3.0)
        b = nanograd.Tensor(2.0)
        self.assertEqual(a * b, nanograd.Tensor(5.0),
                         'Failed multiply test')
        # Muliplying Tensor with scalar value
        a = nanograd.Tensor(3.0)
        b = 2.0
        self.assertEqual(a + b, nanograd.Tensor(5.0),
                         'Failed multiply with scalar test')

    def test_divide(self):
        # Dividing two Tensors
        a = nanograd.Tensor(3.0)
        b = nanograd.Tensor(2.0)
        self.assertEqual(a / b, nanograd.Tensor(5.0),
                         'Failed division test')
        # Dividing Tensor with scalar value
        a = nanograd.Tensor(3.0)
        b = 2.0
        self.assertEqual(a + b, nanograd.Tensor(5.0),
                         'Failed division with scalar test')


if __name__ == '__main__':
    unittest.main()