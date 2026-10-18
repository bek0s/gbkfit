"""
Tests of the conventions of visibilities against CASA: the visibilities
of a point that CASA predicts (tests/data/visibilities/point.ms; see
make_point_ms.py there), read by DatasetVisibilities.from_ms, are those
of the transforms of a cube of the point.
"""

import pathlib

import numpy as np
import pytest

from gbkfit.dataset import DatasetVisibilities
from gbkfit.observation.observables._uv_transforms import uv_transforms
from gbkfit.utils import gridutils


pytest.importorskip('casacore')

MS = pathlib.Path(__file__).parents[1] / 'data' / 'visibilities' / 'point.ms'

# The rest frequency of CO(2-1), which the measurement set does not give
REST = '230.538 GHz'


@pytest.mark.parametrize('name', ['exact', 'sin', 'flat'])
def test_transforms_are_the_visibilities_that_casa_predicts(driver, name):
    # The point of 1 Jy, 20 arcsec east and 15 arcsec north of the phase
    # centre on its tangent plane (x = -20, y = 15): the reference pixel of
    # a grid there, which every transform places at its RA and Dec (then
    # moving the phase centre to that of the data); CASA's values are
    # float32
    dataset = DatasetVisibilities.from_ms(str(MS), rest=REST)
    coverage = dataset.coverage()
    expected = dataset['re'].data() + 1j * dataset['im'].data()
    point = np.degrees(gridutils._to_sky(-20, 15, coverage.phase_centre()))
    grid = gridutils.make_grid((3, 3), 1, rval=point)
    cube = np.zeros((coverage.nchannels(), 3, 3))
    cube[:, 1, 1] = 1
    transform = uv_transforms[name](
        driver, grid, coverage.uvw(), coverage.frequencies(),
        coverage.phase_centre(), 1e-12)
    out = driver.mem_alloc_d(expected.shape, np.complex128)
    transform(driver.mem_copy_h2d(cube), out)
    np.testing.assert_allclose(
        driver.mem_copy_d2h(out), expected, rtol=0, atol=1e-6)


def test_the_errors_of_casa_weights_are_those_of_stokes_i():
    # (CASA writes weights 1 / sigma^2 of each part: 1 for XX and YY,
    # so sigma = sqrt(1 + 1) / 2 for I)
    dataset = DatasetVisibilities.from_ms(str(MS), rest=REST)
    np.testing.assert_allclose(dataset['re'].error(), np.sqrt(2) / 2)
    assert dataset.coverage().frame() == 'LSRK'


def test_the_dirty_image_of_casa_is_the_point(driver):
    # Of natural weights, it peaks at the pixel of the point (x = -20, y =
    # 15 of a grid of 1 arcsec at the phase centre) with its flux, 1 Jy;
    # the dirty beam peaks at the phase centre with 1
    from gbkfit.observation import Visibilities
    dataset = DatasetVisibilities.from_ms(str(MS), rest=REST)
    observable = Visibilities.from_data(
        dataset, size=(61, 41), step=1, transform='sin')
    weights = {key: 1 / dataset[key].error() ** 2 for key in ('re', 'im')}
    dirty = observable.images(
        driver, {key: dataset[key].data() for key in ('re', 'im')},
        weights, {key: dataset[key].unit() for key in ('re', 'im')})['dirty']
    assert dirty.unit == 'Jy beam-1'
    dirty = dirty.data
    beam = observable.psf_images(driver, weights)['dirty_beam'].data
    for image, peak in ((dirty, (20 + 15, 30 - 20)), (beam, (20, 30))):
        assert np.unravel_index(np.argmax(image[0]), image[0].shape) == peak
        np.testing.assert_allclose(image[:, peak[0], peak[1]], 1, atol=1e-5)
