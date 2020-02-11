
import abc

from gbkfit.data import Data
from gbkfit.utils import parseutils


class Dataset(abc.ABC):

    @classmethod
    def load(cls, info):
        return Dataset({k: Data.load(v) for k, v in info.items()})

    def dump(self, prefix=''):
        return {k: v.dump(prefix + k + '_') for k, v in self.items()}

    def __init__(self, data):
        self._data = data

    def __contains__(self, item):
        return item in self._data

    def __getitem__(self, item):
        return self._data[item]

    def __iter__(self):
        return iter(self._data)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def get(self, item):
        return self._data.get(item)

    @property
    def npix(self):
        return next(iter(self._data.values())).npix

    @property
    def size(self):
        return next(iter(self._data.values())).size

    @property
    def step(self):
        return next(iter(self._data.values())).step

    @property
    def zero(self):
        return next(iter(self._data.values())).zero


parser = parseutils.SimpleParser(Dataset)
