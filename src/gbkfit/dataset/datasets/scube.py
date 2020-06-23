
import gbkfit.dataset.data
import gbkfit.dataset.dataset

from gbkfit.utils import parseutils


class DatasetSCube(gbkfit.dataset.dataset.Dataset):

    @staticmethod
    def type():
        return 'scube'

    @classmethod
    def load(cls, info, *args, **kwargs):
        cls_args = parseutils.parse_class_args(cls, info)
        cls_args.update(dict(
            scube=gbkfit.dataset.data.parser.load_one(
                cls_args['scube'],
                step=info.get('step'),
                cval=info.get('cval'))))
        return cls(**cls_args)

    def dump(self, *args, **kwargs):
        scube = self['scube']
        return dict(
            scube=scube.dump(prefix=kwargs.get('prefix', '')),
            step=scube.step,
            cval=scube.cval)

    def __init__(self, scube):
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
    def cval(self):
        return self.cvals[0]
