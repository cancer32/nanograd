import unittest
import nanograd

class TensorTest(unittest.TestCase):
    def test_data(self):
        data = -2.0
        t1 = nanograd.Tensor(data)
        self.assertEqual(t1.data, data,
                         'Failed tensor.data attribute test')

    def test__repr__(self):
        data = -3.14
        return_str = f'Tensor({data})'
        t1 = nanograd.Tensor(data)
        self.assertEqual(t1.__repr__(), return_str,
                         'Failed tensor.__repr__() test')


if __name__ == '__main__':
    unittest.main()