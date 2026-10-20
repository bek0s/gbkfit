"""
Tests of the median filter of observables (option median_size): the
model is smoothed on the sky as the data were.
"""

import numpy as np
import pytest
import scipy.ndimage

from gbkfit.model import galaxy_parser
from gbkfit.observation import (
    Flat, Observation, ObservationGroup, PixelBrightness, PixelMoments,
    PixelSpectra, observable_parser)
from gbkfit.params import ParamSpace
from gbkfit.utils.parseutils import ConfigError


def disk(emission):
    """A thin exponential disk of the given emission, rotating."""
    kinematics = dict(
        vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform'))
    return galaxy_parser.load(dict(components=[dict(
        type='smdisk', loose=False, tilted=False, rnodes=list(range(0, 12)),
        density=dict(ptraits=dict(type='exponential')),
        emission=dict(type=emission))
        | (dict(kinematics=kinematics) if emission == 'lines' else {})]))


PARAMS = dict(
    xpos=0.3, ypos=-0.6, posa=30, incl=60, pt_a=1, pt_s=3, vsys=0,
    vpt_rt=2, vpt_vt=150, dpt_a=20)


def model(driver, observable, galaxy, item, out_extra=None):
    """The model data of an item of an observation of a galaxy."""
    group = ObservationGroup(galaxy, Observation(observable, Flat(), driver))
    params = ParamSpace(group.pdescs(), {
        key: value for key, value in PARAMS.items()
        if key in group.pdescs()}).evaluate()
    return group.model_h(params, out_extra)[0][item]['d'].copy()


def zero_padded_median(data, size):
    """The median filter on the sky, channel by channel, with zero beyond."""
    return scipy.ndimage.median_filter(
        data, size=(1,) * (data.ndim - 2) + (size, size), mode='constant',
        cval=0.0)


def test_images_and_cubes_are_filtered_on_the_sky(driver):
    image = model(
        driver, PixelBrightness(size=(24, 20), median_size=3), disk('band'),
        'brightness')
    plain = model(
        driver, PixelBrightness(size=(24, 20)), disk('band'), 'brightness')
    np.testing.assert_array_equal(image, zero_padded_median(plain, 3))
    cube = model(
        driver, PixelSpectra(size=(24, 20, 41), step=(1, 1, 10),
                             median_size=5), disk('lines'), 'spectra')
    plain = model(
        driver, PixelSpectra(size=(24, 20, 41), step=(1, 1, 10)),
        disk('lines'), 'spectra')
    np.testing.assert_array_equal(cube, zero_padded_median(plain, 5))


def test_moment_maps_are_of_the_filtered_spectra(driver):
    # (the cube of the spectra, an extra output, is that of the filter)
    filtered, plain = {}, {}
    model(driver, PixelMoments(size=(24, 20), orders=(0, 1), median_size=3),
          disk('lines'), 'moment1', filtered)
    model(driver, PixelMoments(size=(24, 20), orders=(0, 1)), disk('lines'),
          'moment1', plain)
    key = 'observation0_dcube_lo'
    np.testing.assert_array_equal(
        filtered[key].data, zero_padded_median(plain[key].data, 3))


def test_the_median_size_is_an_odd_number_of_at_least_3():
    observable = PixelSpectra(size=(8, 8, 5), median_size=3)
    assert observable_parser.load(observable.dump()).dump() \
        == observable.dump()
    for size in (1, 4, 2.0):
        with pytest.raises(ConfigError, match="odd number of pixels"):
            PixelBrightness(size=(8, 8), median_size=size)
