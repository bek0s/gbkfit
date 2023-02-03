
from gbkfit.dataset.core import Dataset
from . import _detail


__all__ = [
    'DatasetImage'
]


class DatasetImage(Dataset):

    @staticmethod
    def type():
        return 'image'

    @classmethod
    def load(cls, info, **kwargs):
        names = ['image']
        opts = _detail.load_dataset_common(cls, info, names, 2, **kwargs)
        return cls(**opts)

    def dump(self, **kwargs):
        return _detail.dump_dataset_common(self, **kwargs)

    def __init__(self, image):
        # TODO: validate wcs
        super().__init__(dict(image=image))
