
import numpy as np

from gbkfit.psflsf import psfs


def test_psfs():
    psf = psfs.PSFGauss(1, 1, 0)
    data = psf.asarray((1, 1))
    assert np.sum(data) == 1
