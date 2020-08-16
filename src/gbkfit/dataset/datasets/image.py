
from gbkfit.dataset import Dataset
from . import _detail


__all__ = ['DatasetImage']


class DatasetImage(Dataset):

    @staticmethod
    def type():
        return 'image'

    @classmethod
    def load(cls, info, *args, **kwargs):
        names = ['image']
        opts = _detail.load_dataset_common(cls, info, names)
        return cls(**opts)

    def dump(self, prefix=''):
        return _detail.dump_dataset_common(self, prefix)

    def __init__(self, image):
        # TODO: validate wcs
        super().__init__(dict(image=image))

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
