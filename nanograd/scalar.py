__all__ = ["Scalar"]


class Scalar(object):
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Scalar({self.data})"

    def __add__(self, other):
        return self.__class__(self.data + other.data)
