
import abc
import collections

from gbkfit.utils import parseutils


class Dataset(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def type():
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, info):
        pass

    def dump(self, prefix=''):
        info = {'type': self.type()}
        for key, value in self.items():
            info[key] = value.dump(prefix + key + '_')
        return info

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
