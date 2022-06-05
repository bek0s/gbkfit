
from gbkfit.dataset.core import Dataset
from . import _detail


__all__ = ['DatasetImage']


class DatasetImage(Dataset):

    @staticmethod
    def type():
        return 'image'

    @classmethod
    def load(cls, info):
        names = ['image']
        opts = _detail.load_dataset_common(cls, info, names)
        return cls(**opts)

    def dump(self, prefix='', dump_full_path=True):
        return _detail.dump_dataset_common(
            self, prefix=prefix, dump_full_path=dump_full_path)

    def __init__(self, image):
        # TODO: validate wcs
        super().__init__(dict(image=image))
