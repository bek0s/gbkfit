
import abc
import collections.abc
import copy

import numpy as np

from gbkfit.utils import parseutils


__all__ = [
    'ParamDescDict',
    'ParamScalarDesc',
    'ParamVectorDesc',
    'load_pdescs_dict',
    'dump_pdescs_dict',
    'pdesc_parser'
]


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
        if np.isfinite(self.minimum()):
            info.update(minimum=self.minimum())
        if np.isfinite(self.maximum()):
            info.update(maximum=self.maximum())
        return info

    def __init__(self, name, size, desc, default, minimum, maximum):
        minimum = -np.inf if minimum is None else minimum
        maximum = +np.inf if maximum is None else maximum
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
            default=None, minimum=None, maximum=None):
        super().__init__(name, 1, desc, default, minimum, maximum)


class ParamVectorDesc(ParamDesc):

    @staticmethod
    def type():
        return 'vector'

    def __init__(
            self, name, size, desc=None,
            default=None, minimum=None, maximum=None):
        super().__init__(name, size, desc, default, minimum, maximum)


class ParamDescDict(collections.abc.Mapping):

    def __init__(self, pdescs):
        self._pdescs = copy.deepcopy(pdescs)

    def __getitem__(self, key):
        return self._pdescs.__getitem__(key)

    def __delitem__(self, key):
        self._pdescs.__delitem__(key)

    def __iter__(self):
        return self._pdescs.__iter__()

    def __len__(self):
        return self._pdescs.__len__()

    def __repr__(self):
        return self._pdescs.__repr__()

    def __str__(self):
        return self._pdescs.__str__()


def load_pdescs_dict(info):
    pdescs = {}
    for key, val in info.items():
        pdescs[key] = pdesc_parser.load(dict(name=key) | val)
    return pdescs


def dump_pdescs_dict(pdescs):
    info = {key: pdesc_parser.dump(val) for key, val in pdescs.items()}
    for val in info.values():
        del val['name']
    return info


pdesc_parser = parseutils.TypedParser(ParamDesc)
pdesc_parser.register(ParamScalarDesc)
pdesc_parser.register(ParamVectorDesc)
