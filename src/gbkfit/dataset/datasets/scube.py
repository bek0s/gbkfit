
from gbkfit.dataset.core import Dataset
from . import _detail


__all__ = [
    'DatasetSCube'
]


class DatasetSCube(Dataset):

    @staticmethod
    def type():
        return 'scube'

    @classmethod
    def load(cls, info, **kwargs):
        names = ['scube']
        opts = _detail.load_dataset_common(cls, info, names, **kwargs)
        return cls(**opts)

    def dump(self, **kwargs):
        return _detail.dump_dataset_common(self, **kwargs)

    def __init__(self, scube):
        # TODO: validate wcs
        super().__init__(dict(scube=scube))
