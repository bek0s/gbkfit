
from gbkfit.dataset.core import Dataset
from . import _detail


__all__ = [
    'DatasetLSlit'
]


class DatasetLSlit(Dataset):

    @staticmethod
    def type():
        return 'lslit'

    @classmethod
    def load(cls, info, **kwargs):
        names = ['lslit']
        opts = _detail.load_dataset_common(cls, info, names, **kwargs)
        return cls(**opts)

    def dump(self, **kwargs):
        return _detail.dump_dataset_common(self, **kwargs)

    def __init__(self, lslit):
        # TODO: validate wcs
        super().__init__(dict(lslit=lslit))
