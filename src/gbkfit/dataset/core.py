
import abc
import copy

from gbkfit.dataset.data import Data
from gbkfit.utils import iterutils, parseutils


__all__ = [
    'Dataset',
    'dataset_parser'
]


def _ensure_same_attrib_value(data, method):
    attr = {k: getattr(v, method)() for k, v in data.items()}
    if len(set(attr.values())) > 1:
        raise RuntimeError(
            f"dataset contains data items of different {method}: {str(attr)}")


class Dataset(parseutils.TypedSerializable, abc.ABC):

    def __init__(self, data):
        # At least one data item must be defined
        if not data:
            raise RuntimeError("dataset contains no data items")
        # All data items must be of the right type
        invalid_data = [k for k, v in data.items() if not isinstance(v, Data)]
        if invalid_data:
            raise RuntimeError(
                f"dataset contains valid data items: {invalid_data}")
        # All data items must have the same properties
        _ensure_same_attrib_value(data, 'size')
        _ensure_same_attrib_value(data, 'step')
        _ensure_same_attrib_value(data, 'rpix')
        _ensure_same_attrib_value(data, 'rval')
        _ensure_same_attrib_value(data, 'rota')
        _ensure_same_attrib_value(data, 'dtype')
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

    def get(self, item, default=None):
        return self._data.get(item, default)

    # All data items have the same properties (see __init__())
    def _first(self):
        return next(iter(self.values()))

    def npix(self):
        return self._first().npix()

    def size(self):
        return self._first().size()

    def step(self):
        return self._first().step()

    def zero(self):
        return self._first().zero()

    def rpix(self):
        return self._first().rpix()

    def rval(self):
        return self._first().rval()

    def rota(self):
        return self._first().rota()

    def dtype(self):
        return self._first().dtype()


class DatasetTypedParser(parseutils.TypedParser):

    def __init__(self):
        super().__init__(Dataset)

    def dump_many(self, x, *args, **kwargs):
        # Ensure that a unique prefix for each dataset is provided
        # in order to avoid datasets overwriting each other
        prefix = iterutils.listify(kwargs.get('prefix'), False)
        if len(x) != len(set(prefix)):
            raise RuntimeError(
                "when dumping multiple datasets,"
                "a unique prefix for each dataset must be provided")
        return super().dump_many(x, *args, **kwargs)


dataset_parser = DatasetTypedParser()
