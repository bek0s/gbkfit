
import abc

from gbkfit.utils import parseutils


__all__ = ['ParamScalarDesc', 'ParamVectorDesc']


class ParamDesc(parseutils.TypedParserSupport, abc.ABC):

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'parameter description')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        info = dict(type=self.type(), name=self.name(), size=self.size())
        if self.desc() is not None:
            info.update(desc=self.desc())
        if self.minimum() is not None:
            info.update(
                minimum=self.minimum(),
                exclusive_minimum=self.exclusive_minimum())
        if self.maximum() is not None:
            info.update(
                maximum=self.maximum(),
                exclusive_maximum=self.exclusive_maximum())
        return info

    def __init__(
            self, name, size, desc=None, default=None,
            minimum=None, exclusive_minimum=False,
            maximum=None, exclusive_maximum=False):
        self._name = name
        self._size = size
        self._desc = desc
        self._default = default
        self._minimum = minimum
        self._maximum = maximum
        self._exclusive_minimum = exclusive_minimum
        self._exclusive_maximum = exclusive_maximum

    def name(self):
        return self._name

    def size(self):
        return self._size

    def desc(self):
        return self._desc

    def default(self):
        return self._default

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum

    def exclusive_maximum(self):
        return self._exclusive_minimum

    def exclusive_minimum(self):
        return self._exclusive_maximum


class ParamScalarDesc(ParamDesc):

    @staticmethod
    def type():
        return 'scalar'

    def dump(self):
        info = super().dump()
        del info['size']
        return info

    def __init__(
            self, name, desc=None, default=None,
            minimum=None, exclusive_minimum=False,
            maximum=None, exclusive_maximum=False):
        super().__init__(
            name, 1, desc, default,
            minimum, exclusive_minimum,
            maximum, exclusive_maximum)


class ParamVectorDesc(ParamDesc):

    @staticmethod
    def type():
        return 'vector'

    def __init__(
            self, name, size, desc=None,
            minimum=None, exclusive_minimum=False,
            maximum=None, exclusive_maximum=False):
        super().__init__(
            name, size, desc,
            minimum, exclusive_minimum,
            maximum, exclusive_maximum)


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
