
from gbkfit.model.dmodels import *


def test_dmodel_image():

    size = (51, 51)
    step = (0.5, 0.5)

    dmodel = DModelImage(size, step)


if __name__ == '__main__':
    test_dmodel_image()
