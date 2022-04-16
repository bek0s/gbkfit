
import abc
import collections.abc

import numpy as np

from gbkfit.utils import parseutils


__all__ = ['ParamDescDict', 'ParamScalarDesc', 'ParamVectorDesc']


def explode_peram_desc(desc, override_name=None):
    pass


def explode_param_descs(descs, override_names=None):
    pass


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
        if self.default() is not None:
            info.update(default=self.default())
        if self.minimum() is not None and np.isfinite(self.minimum()):
            info.update(minimum=self.minimum())
        if self.maximum() is not None and np.isfinite(self.maximum()):
            info.update(maximum=self.maximum())
        return info

    def __init__(self, name, size, desc, default, minimum, maximum):
        assert minimum <= maximum
        self._name = name
        self._size = size
        self._desc = desc
        self._default = default
        self._minimum = minimum
        self._maximum = maximum

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


class ParamScalarDesc(ParamDesc):

    @staticmethod
    def type():
        return 'scalar'

    def dump(self):
        info = super().dump()
        del info['size']
        return info

    def __init__(
            self, name, desc=None,
            default=None, minimum=-np.inf, maximum=+np.inf):
        super().__init__(name, 1, desc, default, minimum, maximum)


class ParamVectorDesc(ParamDesc):

    @staticmethod
    def type():
        return 'vector'

    def __init__(
            self, name, size, desc=None,
            default=None, minimum=-np.inf, maximum=+np.inf):
        super().__init__(name, size, desc, default, minimum, maximum)


class ParamDescDict(collections.abc.MutableMapping):

    def __init__(self, descs):
        self._dict = dict()
        self.update(descs)

    def __getitem__(self, key):
        return self._dict.__getitem__(key)

    def __setitem__(self, key, value):
        assert isinstance(value, ParamDesc)
        assert key == value.name()
        self._dict.__setitem__(key, value)

    def __delitem__(self, key):
        self._dict.__delitem__(key)

    def __iter__(self):
        return self._dict.__iter__()

    def __len__(self):
        return self._dict.__len__()

    def __repr__(self):
        return self._dict.__repr__()

    def __str__(self):
        return self._dict.__str__()

    def exploded_names(self):
        from gbkfit.params.paramutils import explode_param_names
        return explode_param_names(self)


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
