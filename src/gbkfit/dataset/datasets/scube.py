
from gbkfit.dataset import Dataset
from . import _detail


__all__ = ['DatasetSCube']


class DatasetSCube(Dataset):

    @staticmethod
    def type():
        return 'scube'

    @classmethod
    def load(cls, info, *args, **kwargs):
        names = ['scube']
        opts = _detail.load_dataset_common(cls, info, names)
        return cls(**opts)

    def dump(self, prefix=''):
        return _detail.dump_dataset_common(self, prefix)

    def __init__(self, scube):
        # TODO: validate wcs
        super().__init__(dict(scube=scube))

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
    def zero(self):
        return self.zeros[0]

    @property
    def rpix(self):
        return self.rpixs[0]

    @property
    def rval(self):
        return self.rvals[0]

    @property
    def rota(self):
        return self.rotas[0]


