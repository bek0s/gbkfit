
from gbkfit.dataset.core import Dataset
from . import _detail


__all__ = [
    'DatasetLSlit'
]


class DatasetLSlit(Dataset):

    # The axes of a long-slit spectrum: the position along the slit
    # and the spectral axis
    _ndim = 2
    _spectral_axis = 1

    @staticmethod
    def type():
        return 'lslit'

    @classmethod
    def load(cls, info, **kwargs):
        # The options of its one data item are given flat
        names = ['lslit']
        opts = _detail.load_dataset_common(
            cls, _detail.nest_single_item(info, 'lslit'), names, **kwargs)
        return cls(**opts)

    def dump(self, **kwargs):
        return _detail.flatten_single_item(
            _detail.dump_dataset_common(self, **kwargs), 'lslit')

    def __init__(self, lslit):
        super().__init__(dict(lslit=lslit))
