import math
import unittest

import nanograd


class ScalarTest(unittest.TestCase):
    def test_attributes(self):
        # attributes test
        data = -2.0
        _op = "+"
        label = "lable_1"
        ch1 = nanograd.Scalar(3.14)
        x = nanograd.Scalar(data, _children=(ch1,), _op=_op, label=label)
        self.assertEqual(x.data, data,
                         "Failed Scalar.data attribute test")
        self.assertEqual(x._children, (ch1,),
                         "Failed Scalar._children attribute test")
        self.assertEqual(x._op, _op,
                         "Failed Scalar._op attribute test")
        self.assertEqual(x.label, label,
                         "Failed Scalar.label attribute test")

    def test_item(self):
        data = 3.14
        t1 = nanograd.Scalar(data)
        self.assertEqual(t1.item(), data,
                         "Failed item() method test")

    def test__repr__(self):
        # Repr test
        data = -3.14
        return_str = f"Scalar({data})"
        t1 = nanograd.Scalar(data)
        self.assertEqual(t1.__repr__(), return_str,
                         "Failed Scalar.__repr__() test")

    def test_new(self):
        x = nanograd.Scalar.new(3.19)
        print(x)
        self.assertTrue(isinstance(x, nanograd.Scalar),
                        "Failed new staticmethod test")

    def test_equal(self):
        # Checking equality
        self.assertEqual(nanograd.Scalar(3.0), nanograd.Scalar(3.0),
                         "Failed equality test")
        self.assertNotEqual(nanograd.Scalar(3.0), nanograd.Scalar(2.0),
                            "Failed non equality test")

        a = nanograd.Scalar(3.0)
        b = 3.0
        self.assertEqual(a, b,
                         "Failed equality test with scalar")
        self.assertEqual(b, a,
                         "Failed requality test with scalar")

    def test_less_greater(self):
        # Checking if value is lesser than other
        self.assertLess(nanograd.Scalar(1.2), nanograd.Scalar(3.0),
                        "Failed less than test")
        self.assertLessEqual(nanograd.Scalar(1.2), nanograd.Scalar(1.2),
                             "Failed less than equal test")
        self.assertGreater(nanograd.Scalar(3.0), nanograd.Scalar(2.0),
                           "Failed greater than test")
        self.assertGreaterEqual(nanograd.Scalar(3.0), nanograd.Scalar(3.0),
                                "Failed greater than equal test")

        self.assertLess(nanograd.Scalar(3.5), 6,
                        "Failed less than test with scalar")
        self.assertGreater(10.2, nanograd.Scalar(3.0),
                           "Failed greater than with scalar")

    def test_add(self):
        # Adding two Scalars
        a = nanograd.Scalar(3.0)
        b = nanograd.Scalar(2.0)
        self.assertEqual(a + b, nanograd.Scalar(5.0),
                         "Failed addition test")

        # Adding Scalar with scalar value
        a = nanograd.Scalar(3.0)
        b = 2.0
        self.assertEqual(a + b, nanograd.Scalar(5.0),
                         "Failed addition test with scalar ")
        self.assertEqual(b + a, nanograd.Scalar(5.0),
                         "Failed raddition test with scalar")

    def test_subtract(self):
        # Adding two Scalars
        a = nanograd.Scalar(3.0)
        b = nanograd.Scalar(2.0)
        self.assertEqual(a - b, nanograd.Scalar(1.0),
                         "Failed subtraction test")

        # Adding Scalar with scalar value
        a = nanograd.Scalar(3.0)
        b = 2.0
        self.assertEqual(a - b, nanograd.Scalar(1.0),
                         "Failed subtraction test with scalar ")
        self.assertEqual(b - a, nanograd.Scalar(-1.0),
                         "Failed rsubtraction test with scalar")

    def test_muliply(self):
        # Muliplying two Scalars
        a = nanograd.Scalar(3.0)
        b = nanograd.Scalar(2.0)
        self.assertEqual(a * b, nanograd.Scalar(6.0),
                         "Failed multiply test")

        # Muliplying Scalar with scalar value
        a = nanograd.Scalar(3.0)
        b = 2.0
        self.assertEqual(a * b, nanograd.Scalar(6.0),
                         "Failed multiply with scalar test")
        self.assertEqual(b * a, nanograd.Scalar(6.0),
                         "Failed rmultiply with scalar test")

    def test_divide(self):
        # Dividing two Scalars
        a = nanograd.Scalar(3.0)
        b = nanograd.Scalar(2.0)
        self.assertEqual(a / b, nanograd.Scalar(1.5),
                         "Failed division test")
        # Dividing Scalar with scalar value
        a = nanograd.Scalar(3.0)
        b = 2.0
        self.assertEqual(a / b, nanograd.Scalar(1.5),
                         "Failed division with scalar test")
        self.assertEqual(b / a, nanograd.Scalar(0.6666666666666666),
                         "Failed rdivision with scalar test")

    def test_tanh(self):
        x = nanograd.Scalar(3.14)
        self.assertAlmostEqual(x.tanh().item(), math.tanh(x.item()),
                               'Failed tanh test')

    def test_log(self):
        x = nanograd.Scalar(3.14)
        self.assertAlmostEqual(x.log().item(), math.log(x.item()),
                               'Failed log test')

    def test_relu(self):
        x = nanograd.Scalar(3.14)
        self.assertAlmostEqual(x.relu().item(), 3.14,
                               'Failed relu test 1')
        x1 = nanograd.Scalar(0)
        self.assertAlmostEqual(x1.relu().item(), 0,
                               'Failed relu test 2')
        x2 = nanograd.Scalar(-2)
        self.assertAlmostEqual(x2.relu().item(), 0,
                               'Failed relu test 2')

    def test_exp(self):
        x = nanograd.Scalar(3.14)
        self.assertAlmostEqual(x.exp().item(), math.exp(x.item()),
                               'Failed exp test')

    def test_pow(self):
        x = nanograd.Scalar(10)
        y = x ** 3
        self.assertEqual(y.item(), (x.item() ** 3),
                         'Failed pow test')

    def test_backward(self):
        # Addition test
        a = nanograd.Scalar(3.0, label="a")
        b = a + a
        b.label = "b"
        b.backward()
        self.assertEqual(a.grad, 2.0)

        # Addition/Mutiplication test
        a = nanograd.Scalar(-2.0, label="a")
        b = nanograd.Scalar(3.0, label="b")
        d = a * b
        d.label = "d"
        e = a + b
        e.label = "e"
        f = d * e
        f.label = "f"
        f.backward()
        self.assertEqual(a.grad, -3.0)
        self.assertEqual(b.grad, -8.0)


if __name__ == "__main__":
    unittest.main()
