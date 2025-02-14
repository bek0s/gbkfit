
import abc
import copy

from gbkfit.utils import numutils, parseutils


__all__ = [
    'NWMode',
    'NWModeRelative',
    'NWModeRelative1',
    'NWModeRelative2',
    'nwmode_parser'
]


class NWMode(parseutils.TypedSerializable, abc.ABC):

    def dump(self):
        return dict(type=self.type())

    def transform(self, param, in_place=True):
        return self._transform_in_place(
            param if in_place else copy.deepcopy(param))

    @abc.abstractmethod
    def _transform_in_place(self, param):
        pass


class NWModeRelative(NWMode, abc.ABC):

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'node-wise mode')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        return super().dump() | dict(origin=self.origin())

    def __init__(self, origin: int):
        super().__init__()
        self._origin = origin

    def origin(self):
        return self._origin


class NWModeRelative1(NWModeRelative):

    @staticmethod
    def type():
        return 'relative1'

    def _transform_in_place(self, param):
        print(param)
        origin_idx = self.origin()
        origin_val = param[origin_idx]
        param += param[origin_idx]
        param[origin_idx] = origin_val
        return param


class NWModeRelative2(NWModeRelative):

    @staticmethod
    def type():
        return 'relative2'

    def _transform_in_place(self, param):
        return numutils.cumsum(param, self.origin(), out=param)


nwmode_parser = parseutils.TypedParser(NWMode, [
    NWModeRelative1,
    NWModeRelative2])
