
import abc

from gbkfit.utils import parseutils


class ParamDesc(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def type():
        pass

    @classmethod
    def load(cls, info):
        return cls(**parseutils.parse_class_args(cls, info))

    @abc.abstractmethod
    def dump(self):
        pass

    def __init__(self, name, size):
        self._name = name
        self._size = size

    def name(self):
        return self._name

    def size(self):
        return self._size

    def is_scalar(self):
        return not self.is_vector()

    @abc.abstractmethod
    def is_vector(self):
        pass


class ParamScalarDesc(ParamDesc):

    @staticmethod
    def type():
        return 'scalar'

    def dump(self):
        return dict(type=self.type(), name=self.name())

    def __init__(self, name):
        super().__init__(name, 1)

    def is_vector(self):
        return False

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name()!r})'


class ParamVectorDesc(ParamDesc):

    @staticmethod
    def type():
        return 'vector'

    def dump(self):
        return dict(type=self.type(), name=self.name(), size=self.size())

    def __init__(self, name, size):
        super().__init__(name, size)

    def is_vector(self):
        return True

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name()!r}, {self.size()!r})'


parser = parseutils.TypedParser(ParamDesc)

parser.register(ParamScalarDesc)
parser.register(ParamVectorDesc)
