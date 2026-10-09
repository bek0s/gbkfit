
from gbkfit.dataset.core import Dataset
from . import _detail


__all__ = [
    'DatasetSCube'
]


class DatasetSCube(Dataset):

    # The axes of a spectral cube: x, y and the spectral axis
    _ndim = 3
    _spectral_axis = 2

    @staticmethod
    def type():
        return 'scube'

    @classmethod
    def load(cls, info, **kwargs):
        # The options of its one data item are given flat
        names = ['scube']
        opts = _detail.load_dataset_common(
            cls, _detail.nest_single_item(info, 'scube'), names, **kwargs)
        return cls(**opts)

    def dump(self, **kwargs):
        return _detail.flatten_single_item(
            _detail.dump_dataset_common(self, **kwargs), 'scube')

    def __init__(self, scube):
        super().__init__(dict(scube=scube))
