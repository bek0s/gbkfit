"""
Tests for DCube: its low- and high-res grids, and its weight handling:
when the cube is padded for the PSF convolution, the high-res weights
must always be cropped/downscaled to the low-res weight cube, and only
smoothed if smooth_weights is set.
"""

import numpy as np
import pytest

from gbkfit.observation.observables._dcube import DCube, cube_extra
from gbkfit.instrument import LSFGauss
from gbkfit.instrument import PSFGauss


@pytest.mark.parametrize('psf, lsf', [
    (None, None), (PSFGauss(2), LSFGauss(5))])
def test_grids(driver, psf, lsf):
    # The high-res cube has oversampling times the pixels of the low-res cube,
    # and the padding (edge) of the convolution on both sides (along z,
    # only that of the LSF if its FFT convolves separately). Its first
    # pixel is the first high-res pixel of the first low-res pixel, minus
    # the padding.
    size = (21, 30, 41)
    step = (0.5, 1.5, 2.0)
    rpix = (5.0, 6.5, 7.5)
    rval = (1.0, 1.5, 2.0)
    oversampling = (1, 2, 3)
    dcube = DCube(
        size=size, step=step, rpix=rpix, rval=rval, rota=0, rest=None,
        medium=None, oversampling=oversampling, primary_beam=None, psf=psf,
        lsf=lsf, smooth_weights=False,
        dtype=np.dtype(np.float32), fft_dtype=np.dtype(np.float64))
    plan = dcube.plan(driver, has_weights=True)
    # The spatial axes are measured from the reference pixel, and the
    # spectral axis from its world value there
    zero = np.array([0, 0, rval[2]]) - np.multiply(rpix, step)
    step_hi = np.divide(step, oversampling)
    size_hi = np.multiply(size, oversampling)
    edge_hi = np.zeros(3, int)
    if psf:
        lsf_size = lsf.size(step_hi[2])
        kernel_size = psf.size(tuple(step_hi[:2])) + (lsf_size,)
        fft = driver.fft(np.float64)
        size_z = size_hi[2] + lsf_size - 1
        size_hi, edge_hi = fft.fft_convolution_shape(
            tuple(size_hi), kernel_size)
        if fft.separable:
            size_hi = tuple(size_hi[:2]) + (size_z,)
            edge_hi = tuple(edge_hi[:2]) + (lsf_size // 2,)
    zero_hi = zero - np.divide(step, 2) - (np.array(edge_hi) - 0.5) * step_hi
    assert dcube.size() == size
    assert dcube.step() == step
    np.testing.assert_allclose(dcube.zero(), zero)
    grid_hi = plan.scratch_grid()
    assert grid_hi.size == tuple(size_hi)
    assert plan.scratch_edge() == tuple(edge_hi)
    np.testing.assert_allclose(grid_hi.coords.step, step_hi)
    np.testing.assert_allclose(grid_hi.zero(), zero_hi)
    for cube in (plan.dcube(), plan.wcube()):
        assert cube.shape == size[::-1]
    for cube in (plan.scratch_dcube(), plan.scratch_wcube()):
        assert cube.shape == tuple(size_hi)[::-1]
    assert plan.scratch_dcube() is not plan.dcube()
    assert plan.scratch_wcube() is not plan.wcube()


def test_downscaling_keeps_a_uniform_cube(driver):
    dcube = DCube(
        size=(21, 30, 41), step=(0.5, 1.5, 2.0), rpix=(5.0, 6.5, 7.5),
        rval=(1.0, 1.5, 2.0), rota=0, rest=None, medium=None,
        oversampling=(2, 3, 4),
        primary_beam=None, psf=None, lsf=None,
        smooth_weights=False,
        dtype=np.dtype(np.float32), fft_dtype=np.dtype(np.float64))
    plan = dcube.plan(driver, has_weights=False)
    driver.mem_fill(plan.scratch_dcube(), 42)
    plan.evaluate(None, cube_extra, cube_extra)
    np.testing.assert_allclose(driver.mem_copy_d2h(plan.dcube()), 42)


def _evaluate_weights(driver, smooth_weights):
    """
    Evaluate a DCube with a PSF, after writing a known pattern into its
    (padded) high-res weight cube. Return the part of the pattern that
    corresponds to the low-res cube, and the resulting low-res weights.
    """
    dcube = DCube(
        size=(24, 20, 1), step=(1, 1, 1), rpix=(11.5, 9.5, 0),
        rval=(0, 0, 0), rota=0, rest=None, medium=None,
        oversampling=(1, 1, 1), primary_beam=None, psf=PSFGauss(1.5),
        lsf=None, smooth_weights=smooth_weights,
        dtype=np.float32,
        fft_dtype=np.float64)
    plan = dcube.plan(driver, has_weights=True)

    z, y, x = np.indices(plan.scratch_grid().size[::-1])
    pattern = (1 + 0.5 * np.sin(0.4 * x) * np.cos(0.3 * y))
    pattern = pattern.astype(np.float32)
    driver.mem_copy_h2d(pattern, plan.scratch_wcube())
    driver.mem_fill(plan.scratch_dcube(), 0)
    plan.evaluate(None, cube_extra, cube_extra)

    edge_z, edge_y, edge_x = plan.scratch_edge()[::-1]
    size_z, size_y, size_x = dcube.size()[::-1]
    expected = pattern[
        edge_z:edge_z + size_z,
        edge_y:edge_y + size_y,
        edge_x:edge_x + size_x]
    weights = np.asarray(driver.mem_copy_d2h(plan.wcube()))
    return expected, weights


def test_weights_without_smoothing(driver):
    expected, weights = _evaluate_weights(driver, smooth_weights=False)
    np.testing.assert_array_equal(weights, expected)


def test_weights_with_smoothing(driver):
    unsmoothed, weights = _evaluate_weights(driver, smooth_weights=True)
    # Smoothing reduces the variations but keeps the average
    assert weights.std() < 0.8 * unsmoothed.std()
    assert abs(weights.mean() - unsmoothed.mean()) < 0.02


def test_lsf_only_keeps_the_image_on_a_non_square_cube(driver):
    # An LSF smooths each spectrum only, so the image (the cube summed
    # over its spectral axis) must stay the same. Without a PSF, DCube
    # uses a point PSF, which must have the shape of a non-square image.
    from gbkfit.instrument import LSFGauss
    images = []
    for lsf in (None, LSFGauss(2.0)):
        dcube = DCube(
            size=(24, 16, 40), step=(1, 1, 1), rpix=(11.5, 7.5, 19.5),
            rval=(0, 0, 0), rota=0, rest=None, medium=None,
            oversampling=(1, 1, 1),
            primary_beam=None, psf=None, lsf=lsf, smooth_weights=False,
            dtype=np.float32,
            fft_dtype=np.float64)
        plan = dcube.plan(driver, has_weights=False)
        # The same line in every case, in the pixels of the output cube
        # (the scratch cube may be padded for the convolution)
        z, y, x = np.indices(plan.scratch_grid().size[::-1])
        z = z - plan.scratch_edge()[2]
        line = np.exp(-0.5 * ((z - 20 - 0.1 * x) / 3) ** 2)
        cube = (line * (1 + np.sin(0.3 * x) * np.cos(0.4 * y)))
        driver.mem_copy_h2d(cube.astype(np.float32), plan.scratch_dcube())
        plan.evaluate(None, cube_extra, cube_extra)
        images.append(np.asarray(driver.mem_copy_d2h(plan.dcube())).sum(0))
    np.testing.assert_allclose(images[1], images[0], rtol=1e-4)
