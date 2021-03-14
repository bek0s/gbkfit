
import abc
import copy

from gbkfit.utils import parseutils


def _ensure_same_attrib_value(data, method, class_desc):
    attr = {k: getattr(v, method)() for k, v in data.items()}
    if len(set(attr.values())) > 1:
        raise RuntimeError(
            f"{class_desc} contains data items of different {method}: "
            f"{str(attr)}")


class Dataset(parseutils.TypedParserSupport, abc.ABC):

    def __init__(self, data):
        desc = parseutils.make_typed_desc(self.__class__, 'dataset')
        # At least one data item must be defined
        if not data:
            raise RuntimeError(f"{desc} contains no data items")
        # All data items must have the same properties
        _ensure_same_attrib_value(data, 'size', desc)
        _ensure_same_attrib_value(data, 'step', desc)
        _ensure_same_attrib_value(data, 'rpix', desc)
        _ensure_same_attrib_value(data, 'rval', desc)
        _ensure_same_attrib_value(data, 'rota', desc)
        _ensure_same_attrib_value(data, 'dtype', desc)
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
    def npix(self):
        return next(iter(self.values())).npix

    @property
    def size(self):
        return next(iter(self.values())).size

    @property
    def step(self):
        return next(iter(self.values())).step

    @property
    def zero(self):
        return next(iter(self.values())).zero

    @property
    def rpix(self):
        return next(iter(self.values())).rpix

    @property
    def rval(self):
        return next(iter(self.values())).rval

    @property
    def rota(self):
        return next(iter(self.values())).rota

    @property
    def dtype(self):
        return next(iter(self.values())).dtype()


dataset_parser = parseutils.TypedParser(Dataset)
