"""
Tests of the physical units of the light of models, seen by observations
of data in their units.
"""

import astropy.units as u
import numpy as np
import pytest

from gbkfit.dataset import Beam, Data, DatasetPixelSpectra
from gbkfit.model import galaxy_parser
from gbkfit.observation import (
    Observation, ObservationGroup, PixelSpectra, line_of_sight_parser)
from gbkfit.params import ParamSpace
from gbkfit.utils.parseutils import ConfigError


HI = 1420.405751768 * u.MHz
BEAM = Beam(66, 53.7)
SIZE = (24, 24, 41)
STEP = (10, 10, 10)
PARAMS = dict(
    xpos=0, ypos=0, posa=30, incl=60, vsys=470, pt_a=1, pt_s=40, vpt_rt=40,
    vpt_vt=150, dpt_a=15)


def disk(unit=None, lines=('hi21',), thick=False):
    """A disk of HI (with the given unit of its light, if any)."""
    emission = dict(type='lines', lines=list(lines))
    if unit is not None:
        emission.update(unit=unit)
    density = dict(ptraits=dict(type='exponential'))
    if thick:
        density.update(htraits=dict(type='sech2', zmax=30))
    return dict(
        type='smdisk', rnodes=list(range(0, 121, 20)), density=density,
        kinematics=dict(
            vptraits=dict(type='tan_arctan'), dptraits=dict(type='uniform')),
        emission=emission)


def model(
        driver, components, data_unit=None, beam=BEAM, line_of_sight='flat',
        params=PARAMS):
    """
    The model of the spectra of a galaxy of the given components, seen in
    data of the given unit (none: without data) along a line of sight of
    the given type.
    """
    observable = PixelSpectra(
        size=SIZE, step=STEP, rval=(0, 0, 470), rest=HI)
    data = None
    if data_unit is not None:
        data = DatasetPixelSpectra(
            Data(np.zeros(SIZE[::-1]), unit=data_unit, beam=beam),
            step=STEP, rval=(0, 0, 470), rest=HI)
    group = ObservationGroup(
        [galaxy_parser.load(dict(components=components))], [Observation(
            observable, line_of_sight_parser.load(dict(type=line_of_sight)),
            driver, data=data)])
    space = ParamSpace(group.pdescs(), params)
    return group.model_h(space.evaluate())[0]['spectra']['d'].copy()


def per_kelvin(vsys):
    """Return Jy/beam per K at the frequency of HI at a systemic velocity."""
    frequency = HI * (1 - vsys / 299792.458)
    return ((1 * u.K).to(u.Jy / u.sr, u.brightness_temperature(frequency))
            * BEAM.area() * u.arcsec ** 2).to_value(u.Jy)


def test_one_hi_disk_in_any_unit_of_data(driver):
    # A disk of HI of 1 Msun/pc^2 at its centre, without a unit (in the
    # unit of the data), and in Msun/pc^2 seen in K and in Jy/beam: N_HI =
    # 1.823e18 cm^-2 per K km/s, and K to Jy/beam by Rayleigh-Jeans at the
    # frequency of the line at the systemic velocity, in the beam
    plain = model(driver, disk(), 'K')
    kelvin = model(driver, disk('Msun/pc2'), 'K')
    jansky = model(driver, disk('Msun/pc2'), 'Jy/beam')
    column = (1 * u.Msun / u.pc ** 2 / (1.00794 * u.u)).to_value(u.cm ** -2)
    assert_same(kelvin, plain * column / 1.823e18)
    assert_same(jansky, kelvin * per_kelvin(470))


def test_line_opacity_saturates_at_a_temperature(driver):
    # With a unit, S is a brightness temperature, whatever the unit of the
    # data: the light of each channel is S (1 - exp(-I / S)) in K, I that
    # without line opacity, and that times Jy/beam per K in Jy/beam
    params = PARAMS | dict(ht_s=5)
    gas = disk('K km/s', thick=True)
    thin = model(driver, [gas], 'K', line_of_sight='transfer', params=params)
    source = thin.max() / 3
    gas |= dict(opacity=dict(
        type='line', coefficients=dict(hi21=dict(type='uniform'))))
    params |= dict(op_hi21_a=1 / source)
    kelvin = model(driver, [gas], 'K', None, 'transfer', params)
    jansky = model(driver, [gas], 'Jy/beam', BEAM, 'transfer', params)
    assert_same(kelvin, source * -np.expm1(-thin / source))
    assert_same(jansky, kelvin * per_kelvin(470))
    assert kelvin.sum() < 0.8 * thin.sum()


def assert_same(actual, desired):
    """Assert that two models are one, to the precision of float32."""
    np.testing.assert_allclose(
        actual, desired, rtol=1e-5, atol=1e-6 * np.abs(desired).max())


@pytest.mark.parametrize('data_unit, beam, message', [
    ('Jy/beam', None, "have no beam"),
    (None, None, "without data, the model of an observation is in the unit"),
])
def test_the_units_of_model_and_data_must_go_together(
        driver, data_unit, beam, message):
    components = [
        disk('Msun/pc2') | dict(name='hi'),
        disk('K km/s', ['co10']) | dict(name='co')]
    if data_unit is not None:
        components = components[:1]
    with pytest.raises(ConfigError, match=message):
        model(driver, components, data_unit, beam)
