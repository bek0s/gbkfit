
import abc

from gbkfit.utils import parseutils


class Dataset(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def load(cls, info, *args, **kwargs):
        pass

    @abc.abstractmethod
    def dump(self, *args, **kwargs):
        pass

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
    def npixs(self):
        return tuple(data.npix for data in self.values())

    @property
    def sizes(self):
        return tuple(data.size for data in self.values())

    @property
    def steps(self):
        return tuple(data.step for data in self.values())

    @property
    def cvals(self):
        return tuple(data.cval for data in self.values())


parser = parseutils.TypedParser(Dataset)
