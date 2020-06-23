
import gbkfit.dataset.data
import gbkfit.dataset.dataset

from gbkfit.utils import parseutils


class DatasetLSlit(gbkfit.dataset.dataset.Dataset):

    @staticmethod
    def type():
        return 'lslit'

    @classmethod
    def load(cls, info, *args, **kwargs):
        cls_args = parseutils.parse_class_args(cls, info)
        cls_args.update(dict(
            lslit=gbkfit.dataset.data.parser.load_one(
                cls_args['lslit'],
                step=info.get('step'),
                cval=info.get('cval'))))
        return cls(**cls_args)

    def dump(self, *args, **kwargs):
        lslit = self['lslit']
        return dict(
            lslit=lslit.dump(prefix=kwargs.get('prefix', '')),
            step=lslit.step,
            cval=lslit.cval)

    def __init__(self, lslit):
        super().__init__(dict(lslit=lslit))

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
