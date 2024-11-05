__all__ = ['Tensor']


class Tensor(object):
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f'Tensor({self.data})'