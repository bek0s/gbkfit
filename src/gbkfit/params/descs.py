
import abc

from gbkfit.utils import parseutils


__all__ = ['ParamScalarDesc', 'ParamVectorDesc']


class ParamDesc(parseutils.TypedParserSupport, abc.ABC):

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'parameter description')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def __init__(self, name, size):
        self._name = name
        self._size = size

    def name(self):
        return self._name

    def size(self):
        return self._size


class ParamScalarDesc(ParamDesc):

    @staticmethod
    def type():
        return 'scalar'

    def dump(self):
        return dict(type=self.type(), name=self.name())

    def __init__(self, name):
        super().__init__(name, 1)

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

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name()!r}, {self.size()!r})'


def load_descs_dict(info):
    descs = {}
    for key, val in info.items():
        descs[key] = pdesc_parser.load(dict(name=key) | val)
    return descs


def dump_descs_dict(descs):
    info = {key: pdesc_parser.dump(val) for key, val in descs.items()}
    for val in info.values():
        del val['name']
    return info


pdesc_parser = parseutils.TypedParser(ParamDesc)
pdesc_parser.register(ParamScalarDesc)
pdesc_parser.register(ParamVectorDesc)
