
import abc
import copy

from gbkfit.utils import parseutils
from . import _detail


class Dataset(parseutils.TypedParserSupport, abc.ABC):

    def __init__(self, data):
        desc = parseutils.make_typed_desc(self.__class__, 'dataset')
        # At least one data item must be defined
        if not data:
            raise RuntimeError(f"{desc} contains no data items")
        # All data items must have the same dtype
        _detail.ensure_same_attrib_value(data, 'dtype', desc)
        # We need to copy the data to ensure they are kept intact
        self._data = copy.deepcopy(data)

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
    def zeros(self):
        return tuple(data.zero for data in self.values())

    @property
    def rpixs(self):
        return tuple(data.rpix for data in self.values())

    @property
    def rvals(self):
        return tuple(data.rval for data in self.values())

    @property
    def rotas(self):
        return tuple(data.rota for data in self.values())

    @property
    def dtype(self):
        return next(iter(self.values())).dtype()


dataset_parser = parseutils.TypedParser(Dataset)
