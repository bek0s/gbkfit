
from gbkfit.dataset.core import Dataset
from . import _detail


__all__ = [
    'DatasetImage'
]


class DatasetImage(Dataset):

    # An image has no spectral axis
    _ndim = 2
    _spectral_axis = None

    @staticmethod
    def type():
        return 'image'

    @classmethod
    def load(cls, info, **kwargs):
        # The options of its one data item are given flat
        names = ['image']
        opts = _detail.load_dataset_common(
            cls, _detail.nest_single_item(info, 'image'), names, **kwargs)
        return cls(**opts)

    def dump(self, **kwargs):
        return _detail.flatten_single_item(
            _detail.dump_dataset_common(self, **kwargs), 'image')

    def __init__(self, image):
        super().__init__(dict(image=image))
