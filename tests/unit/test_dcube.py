"""
Tests for the weight handling of DCube: when the cube is padded for the
PSF convolution, the high-res weights must always be cropped/downscaled
to the low-res weight cube, and only smoothed if smooth_weights is set.
"""

import numpy as np

from gbkfit.model.dmodels._dcube import DCube
from gbkfit.psflsf.psfs import PSFGauss


def _evaluate_weights(driver, smooth_weights):
    """
    Evaluate a DCube with a PSF, after writing a known pattern into its
    (padded) high-res weight cube. Return the part of the pattern that
    corresponds to the low-res cube, and the resulting low-res weights.
    """
    dcube = DCube(
        size=(24, 20, 1), step=(1, 1, 1), rpix=(11.5, 9.5, 0),
        rval=(0, 0, 0), rota=0, scale=(1, 1, 1),
        psf=PSFGauss(1.5), lsf=None, smooth_weights=smooth_weights,
        mask_cutoff=None, mask_apply=False, dtype=np.float32)
    dcube.prepare(driver, has_weights=True)

    z, y, x = np.indices(dcube.scratch_size()[::-1])
    pattern = (1 + 0.5 * np.sin(0.4 * x) * np.cos(0.3 * y))
    pattern = pattern.astype(np.float32)
    driver.mem_copy_h2d(pattern, dcube.scratch_wcube())
    driver.mem_fill(dcube.scratch_dcube(), 0)
    dcube.evaluate(None)

    edge_z, edge_y, edge_x = dcube.scratch_edge()[::-1]
    size_z, size_y, size_x = dcube.size()[::-1]
    expected = pattern[
        edge_z:edge_z + size_z,
        edge_y:edge_y + size_y,
        edge_x:edge_x + size_x]
    weights = np.asarray(driver.mem_copy_d2h(dcube.wcube()))
    return expected, weights


def test_weights_without_smoothing(driver):
    expected, weights = _evaluate_weights(driver, smooth_weights=False)
    np.testing.assert_array_equal(weights, expected)


def test_weights_with_smoothing(driver):
    unsmoothed, weights = _evaluate_weights(driver, smooth_weights=True)
    # Smoothing reduces the variations but keeps the average
    assert weights.std() < 0.8 * unsmoothed.std()
    assert abs(weights.mean() - unsmoothed.mean()) < 0.02


def test_mask(driver):
    # A pattern with values below and above the cutoff in every slice
    dcube = DCube(
        size=(12, 10, 8), step=(1, 1, 1), rpix=(5.5, 4.5, 3.5),
        rval=(0, 0, 0), rota=0, scale=(1, 1, 1),
        psf=None, lsf=None, smooth_weights=False,
        mask_cutoff=0.5, mask_apply=True, dtype=np.float32)
    dcube.prepare(driver, has_weights=False)
    z, y, x = np.indices(dcube.size()[::-1])
    pattern = (np.sin(0.7 * x + 0.3 * y + 0.5 * z) + 1).astype(np.float32)
    driver.mem_copy_h2d(pattern, dcube.scratch_dcube())
    dcube.evaluate(None)
    data = np.asarray(driver.mem_copy_d2h(dcube.dcube()))
    mask = np.asarray(driver.mem_copy_d2h(dcube.mcube()))
    keep = pattern > 0.5
    np.testing.assert_array_equal(mask, keep.astype(np.float32))
    np.testing.assert_array_equal(data, np.where(keep, pattern, np.nan))


def test_lsf_only_keeps_the_image_on_a_non_square_cube(driver):
    # An LSF smooths each spectrum only, so the image (the cube summed
    # over its spectral axis) must stay the same. Without a PSF, DCube
    # uses a point PSF, which must have the shape of a non-square image.
    from gbkfit.psflsf.lsfs import LSFGauss
    images = []
    for lsf in (None, LSFGauss(2.0)):
        dcube = DCube(
            size=(24, 16, 40), step=(1, 1, 1), rpix=(11.5, 7.5, 19.5),
            rval=(0, 0, 0), rota=0, scale=(1, 1, 1),
            psf=None, lsf=lsf, smooth_weights=False,
            mask_cutoff=None, mask_apply=False, dtype=np.float32)
        dcube.prepare(driver, has_weights=False)
        # The same line in every case, in the pixels of the output cube
        # (the scratch cube may be padded for the convolution)
        z, y, x = np.indices(dcube.scratch_size()[::-1])
        z = z - dcube.scratch_edge()[2]
        line = np.exp(-0.5 * ((z - 20 - 0.1 * x) / 3) ** 2)
        cube = (line * (1 + np.sin(0.3 * x) * np.cos(0.4 * y)))
        driver.mem_copy_h2d(cube.astype(np.float32), dcube.scratch_dcube())
        dcube.evaluate(None)
        images.append(np.asarray(driver.mem_copy_d2h(dcube.dcube())).sum(0))
    np.testing.assert_allclose(images[1], images[0], rtol=1e-4)
