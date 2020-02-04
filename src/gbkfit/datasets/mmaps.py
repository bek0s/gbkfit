
import gbkfit.data
import gbkfit.dataset


class MMaps(gbkfit.dataset.Dataset):

    @staticmethod
    def type():
        return 'mmaps'

    @classmethod
    def load(cls, info):

        for key, value in info.items():
            pass
        bmap_data = gbkfit.data.Data.load(info['mmap0'])
        vmap_data = gbkfit.data.Data.load(info['mmap1'])
        dmap_data = gbkfit.data.Data.load(info['mmap2'])
        return cls(bmap_data, vmap_data, dmap_data)

    def __init__(self, bmap_data, vmap_data, dmap_data):
        super().__init__({
            'bmap': bmap_data,
            'vmap': vmap_data,
            'dmap': dmap_data
        })
