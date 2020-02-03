
from gbkfit.psflsfs import *


def test_psfs():
    psf = PSFGauss(1, 1, 0)
    data = psf.asarray((1, 1))
    assert np.sum(data) == 1
