
import math

import numpy as np
import pytest

import gbkfit.math
from gbkfit.psflsf import *
from gbkfit.psflsf.lsfs import *
from gbkfit.psflsf.psfs import *


@pytest.mark.parametrize(
    "psf_type, psf_class, psf_params", [
        ('point', PSFPoint, dict()),
        ('gauss', PSFGauss, dict(
            sigma=5.0, ratio=1.0, posa=0.0)),
        ('ggauss', PSFGGauss, dict(
            alpha=5.0, beta=1.0, ratio=1.0, posa=0.0)),
        ('lorentz', PSFLorentz, dict(
            gamma=5.0, ratio=1.0, posa=0.0)),
        ('moffat', PSFMoffat, dict(
            alpha=5.0, beta=1.0, ratio=1.0, posa=0.0))
    ]
)
def test_psf_analytic(psf_type, psf_class, psf_params):
    step = (0.5, 0.2)
    offset = (0, 0)
    # Creation tests
    psf = psf_class(**psf_params)
    psf_arr = psf.asarray(step)
    psf_size = psf.size(step, offset)
    arr_max_index = np.unravel_index(np.argmax(psf_arr), psf_arr.shape)
    arr_max_index = tuple(i.item() for i in arr_max_index)
    arr_max_index = arr_max_index[::-1]
    assert psf_arr.shape[::-1] == psf_size
    assert gbkfit.math.is_odd(np.all(psf_size))
    assert math.isclose(np.sum(psf_arr), 1.0, abs_tol=1e-9)
    assert arr_max_index == (psf_size[0] // 2, psf_size[1] // 2)
    # Dump tests
    dumped_psf_info = psf.dump()
    psf_info = dict(
        type=psf_type,
        **psf_params)
    assert dumped_psf_info == psf_info
    # Load tests
    loaded_psf = psf_parser.load(dumped_psf_info)
    assert vars(loaded_psf) == vars(psf)


@pytest.mark.parametrize(
    "lsf_type, lsf_class, lsf_params", [
        ('point', LSFPoint, dict()),
        ('gauss', LSFGauss, dict(
            sigma=5.0)),
        ('ggauss', LSFGGauss, dict(
            alpha=5.0, beta=1.0)),
        ('lorentz', LSFLorentz, dict(
            gamma=5.0)),
        ('moffat', LSFMoffat, dict(
            alpha=5.0, beta=1.0))
    ]
)
def test_lsf_analytic(lsf_type, lsf_class, lsf_params):
    step = 0.5
    offset = 0
    # Creation tests
    lsf = lsf_class(**lsf_params)
    lsf_arr = lsf.asarray(step)
    lsf_size = lsf.size(step, offset)
    arr_max_index = np.unravel_index(np.argmax(lsf_arr), lsf_arr.shape)
    arr_max_index = tuple(i.item() for i in arr_max_index)
    arr_max_index = arr_max_index[::-1][0]
    assert lsf_arr.shape[0] == lsf_size
    assert gbkfit.math.is_odd(np.all(lsf_size))
    assert math.isclose(np.sum(lsf_arr), 1.0, abs_tol=1e-9)
    assert arr_max_index == lsf_size // 2
    # Dump tests
    dumped_lsf_info = lsf.dump()
    lsf_info = dict(
        type=lsf_type,
        **lsf_params)
    assert dumped_lsf_info == lsf_info
    # Load tests
    loaded_lsf = lsf_parser.load(dumped_lsf_info)
    assert vars(loaded_lsf) == vars(lsf)


if __name__ == '__main__':
    test_psf_analytic()
    test_lsf_analytic()
