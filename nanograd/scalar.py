__all__ = ["Scalar"]


class Scalar(object):
    def __init__(self, data, label=None, _children=None, _op=None):
        self.data = data
        self.label = label or ''
        self._children = _children or ()
        self._op = _op or ''

    def __repr__(self):
        return f"Scalar({self.data})"

    @staticmethod
    def new(data):
        """Returns a Scalar wrapper of the data

        :param data: scalar value
        :type data: object
        :return: Scalar wrapper
        :rtype: nanograd.Scalar
        """
        if not isinstance(data, Scalar):
            data = Scalar(data)
        return data

    def __add__(self, other):
        return self.__class__(self.data + other.data)


    def item(self):
        """Returns the original scalar value
        """
        return self.data