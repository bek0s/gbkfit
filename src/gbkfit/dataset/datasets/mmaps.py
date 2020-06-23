
import re

import gbkfit.dataset.data
import gbkfit.dataset.dataset


class DatasetMMaps(gbkfit.dataset.dataset.Dataset):

    @staticmethod
    def type():
        return 'mmaps'

    @classmethod
    def load(cls, info, *args, **kwargs):
        mmaps = {}
        for key, data in info.items():
            match = re.findall(r'^mmap([0-9]|[1-9][0-9]*)$', key)
            if match:
                mmaps[int(match[0])] = gbkfit.dataset.data.parser.load_one(
                    data, step=info.get('step'), cval=info.get('cval'))
        if not mmaps:
            raise RuntimeError("at least one moment map must be provided")
        return dict(sorted(mmaps.items()))

    def dump(self, *args, **kwargs):
        prefix = kwargs.get('prefix', '')
        info = dict((k, v.dump(prefix=prefix)) for k, v in self.items())
        info.update(dict(
            step=self[0].step,
            cval=self[0].cval))
        return info

    def __init__(self, mmaps):
        super().__init__({f'mmap{order}': map for order, map in mmaps.items()})

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
