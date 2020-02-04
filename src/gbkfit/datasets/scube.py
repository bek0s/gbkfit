
import gbkfit.data
import gbkfit.dataset


class SCube(gbkfit.dataset.Dataset):

    @staticmethod
    def type():
        return 'scube'

    @classmethod
    def load(cls, info):
        scube_data = gbkfit.data.Data.load(info['scube'])
        return cls(scube_data)

    def __init__(self, scube_data):
        super().__init__({
            'scube': scube_data
        })
