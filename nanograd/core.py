
class Value:
    def __init__(self, value, label=None, _children=()):
        self.value = value
        self.label = label
        self._children = _children

    @staticmethod
    def new(data):
        if not isinstance(data, Value):
            data = Value(data)
        return data

    def __add__(self, other):
        other = self.new(other)
        return self.data + other.data
