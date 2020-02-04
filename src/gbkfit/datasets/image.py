
import gbkfit.data
import gbkfit.dataset


class Image(gbkfit.dataset.Dataset):

    @staticmethod
    def type():
        return 'image'

    @classmethod
    def load(cls, info):
        image_data = gbkfit.data.Data.load(info['image'])
        return cls(image_data)

    def __init__(self, image_data):
        super().__init__({
            'image': image_data
        })
