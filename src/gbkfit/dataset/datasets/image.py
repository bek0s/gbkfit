
import gbkfit.dataset.data
import gbkfit.dataset.dataset

from gbkfit.utils import parseutils


class DatasetImage(gbkfit.dataset.dataset.Dataset):

    @staticmethod
    def type():
        return 'image'

    @classmethod
    def load(cls, info):
        args = parseutils.parse_class_args(cls, info)
        args.update(dict(
            image=gbkfit.dataset.data.parser.load_one(
                args['image'], step=info.get('step'), cval=info.get('cval'))))
        return cls(**args)

    def dump(self, **kwargs):
        image = self['image']
        return dict(
            image=image.dump(prefix=kwargs.get('prefix', '')),
            step=image.step,
            cval=image.cval)

    def __init__(self, image):
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
    def cval(self):
        return self.cvals[0]
