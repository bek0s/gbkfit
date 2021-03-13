
from gbkfit.dataset.core import Dataset
from . import _detail


__all__ = ['DatasetMMaps']


class DatasetMMaps(Dataset):

    @staticmethod
    def type():
        return 'mmaps'

    @classmethod
    def load(cls, info, *args, **kwargs):
        names = [f'mmap{i}' for i in range(7)]
        opts = _detail.load_dataset_common(cls, info, names)
        return cls(**opts)

    def dump(self, prefix=''):
        return _detail.dump_dataset_common(self, prefix)

    def __init__(
            self,
            mmap0=None, mmap1=None, mmap2=None, mmap3=None,
            mmap4=None, mmap5=None, mmap6=None, mmap7=None):
        mmaps = locals().copy()
        mmaps.pop('self')
        mmaps.pop('__class__')
        # TODO: validate wcs
        super().__init__({k: v for k, v in mmaps.items() if v})
