import unittest
import nanograd


class ScalarTest(unittest.TestCase):
    def test_attributes(self):
        # data attribute test
        data = -2.0
        t1 = nanograd.Scalar(data)
        self.assertEqual(t1.data, data,
                         'Failed Scalar.data attribute test')

        # data attribute test
        data = -2.0
        t1 = nanograd.Scalar(data)
        self.assertEqual(t1.data, data,
                         'Failed Scalar.data attribute test')

    def test_item(self):
        data = 3.14
        t1 = nanograd.Scalar(data)
        self.assertEqual(t1.item(), data,
                         'Failed item() method test')

    def test__repr__(self):
        # Repr test
        data = -3.14
        return_str = f'Scalar({data})'
        t1 = nanograd.Scalar(data)
        self.assertEqual(t1.__repr__(), return_str,
                         'Failed Scalar.__repr__() test')
        
    def test_add(self):
        # Adding two Scalars
        a = nanograd.Scalar(3.0)
        b = nanograd.Scalar(2.0)
        self.assertEqual(a + b, nanograd.Scalar(5.0),
                         'Failed addition test')
        # Adding Scalar with scalar value
        a = nanograd.Scalar(3.0)
        b = 2.0
        self.assertEqual(a + b, nanograd.Scalar(5.0),
                         'Failed addition with scalar test')

    def test_muliply(self):
        # Muliplying two Scalars
        a = nanograd.Scalar(3.0)
        b = nanograd.Scalar(2.0)
        self.assertEqual(a * b, nanograd.Scalar(5.0),
                         'Failed multiply test')
        # Muliplying Scalar with scalar value
        a = nanograd.Scalar(3.0)
        b = 2.0
        self.assertEqual(a + b, nanograd.Scalar(5.0),
                         'Failed multiply with scalar test')

    def test_divide(self):
        # Dividing two Scalars
        a = nanograd.Scalar(3.0)
        b = nanograd.Scalar(2.0)
        self.assertEqual(a / b, nanograd.Scalar(5.0),
                         'Failed division test')
        # Dividing Scalar with scalar value
        a = nanograd.Scalar(3.0)
        b = 2.0
        self.assertEqual(a + b, nanograd.Scalar(5.0),
                         'Failed division with scalar test')

    def test_backward(self):
        # Addition test
        a = nanograd.Scalar(3.0, label='a')
        b = a + a; b.label = 'b'
        b.backward()
        self.assertEqual(b.grad, 2)

        # Addition/Mutiplication test
        a = nanograd.Scalar(-2.0, label='a')
        b = nanograd.Scalar(3.0, label='b')
        d = a * b ; d.label = 'd'
        e = a + b; e.label = 'e'
        f = d * e; f.label = 'f'
        f.backward()
        self.assertEqual(a.grad, -3.0)
        self.assertEqual(b.grad, -8.0)



if __name__ == '__main__':
    unittest.main()