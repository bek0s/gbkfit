
from gbkfit.dataset import Dataset
from . import _detail

__all__ = ['DatasetSCube']


class DatasetSCube(Dataset):

    @staticmethod
    def type():
        return 'scube'

    @classmethod
    def load(cls, info):
        names = ['scube']
        opts = _detail.load_dataset_common(cls, info, names)
        return cls(**opts)

    def dump(self, prefix=''):
        return _detail.dump_dataset_common(self, prefix)

    def __init__(self, scube):
        # TODO: validate wcs
        super().__init__(dict(scube=scube))
