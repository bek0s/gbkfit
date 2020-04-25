
import gbkfit.dataset.data
import gbkfit.dataset.dataset

from gbkfit.utils import parseutils


class DatasetLSlit(gbkfit.dataset.dataset.Dataset):

    @staticmethod
    def type():
        return 'lslit'

    @classmethod
    def load(cls, info):
        args = parseutils.parse_class_args(cls, info)
        args.update(dict(
            lslit=gbkfit.dataset.data.parser.load_one(
                args['lslit'], step=info.get('step'), cval=info.get('cval'))))
        return cls(**args)

    def dump(self, **kwargs):
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
