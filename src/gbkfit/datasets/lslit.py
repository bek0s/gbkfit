
import gbkfit.data
import gbkfit.dataset


class LSlit(gbkfit.dataset.Dataset):

    @staticmethod
    def type():
        return 'lslit'

    @classmethod
    def load(cls, info):
        lslit_data = gbkfit.data.Data.load(info['lslit'])
        return cls(lslit_data)

    def __init__(self, lslit_data):
        super().__init__({
            'lslit': lslit_data
        })
