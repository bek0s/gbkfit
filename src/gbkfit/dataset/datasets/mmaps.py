
from gbkfit.dataset import Dataset, _detail as dataset_detail, make_dataset_desc
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
        mmaps_args = locals()
        mmaps_args.pop('self')
        mmaps_args.pop('__class__')
        mmaps = {k: v for k, v in mmaps_args.items() if v}
        # All moment maps must have the same attributes
        desc = make_dataset_desc(self.__class__)
        dataset_detail.ensure_same_attrib_value(mmaps, 'size', desc)
        dataset_detail.ensure_same_attrib_value(mmaps, 'step', desc)
        dataset_detail.ensure_same_attrib_value(mmaps, 'cval', desc)
        dataset_detail.ensure_same_attrib_value(mmaps, 'rota', desc)
        # TODO: validate wcs
        super().__init__(mmaps)

    @property
    def npix(self):
        return self.npixs[0]

    @property
    def size(self):
        return self.sizes[0]

    @property
    def step(self):
        return self.steps[0]

    @property
    def cpix(self):
        return self.cpixs[0]

    @property
    def cval(self):
        return self.cvals[0]

    @property
    def rota(self):
        return self.rotas[0]
