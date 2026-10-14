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


def disk(unit=None, lines=('hi21',)):
    """A disk of HI (with the given unit of its light, if any)."""
    emission = dict(type='lines', lines=list(lines))
    if unit is not None:
        emission.update(unit=unit)
    return dict(
        type='smdisk', rnodes=list(range(0, 121, 20)),
        density=dict(ptraits=dict(type='exponential')),
        kinematics=dict(
            vptraits=dict(type='tan_arctan'), dptraits=dict(type='uniform')),
        emission=emission)


def model(driver, components, data_unit=None, beam=BEAM):
    """
    The model of the spectra of a galaxy of the given components, seen in
    data of the given unit (none: without data).
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
            observable, line_of_sight_parser.load(dict(type='flat')), driver,
            data=data)])
    params = ParamSpace(group.pdescs(), PARAMS)
    return group.model_h(params.evaluate())[0]['spectra']['d'].copy()


def test_one_hi_disk_in_any_unit_of_data(driver):
    # A disk of HI of 1 Msun/pc^2 at its centre, without a unit (in the
    # unit of the data), and in Msun/pc^2 seen in K and in Jy/beam: N_HI =
    # 1.823e18 cm^-2 per K km/s, and K to Jy/beam by Rayleigh-Jeans at the
    # frequency of the line at the systemic velocity, in the beam
    plain = model(driver, disk(), 'K')
    kelvin = model(driver, disk('Msun/pc2'), 'K')
    jansky = model(driver, disk('Msun/pc2'), 'Jy/beam')
    column = (1 * u.Msun / u.pc ** 2 / (1.00794 * u.u)).to_value(u.cm ** -2)
    frequency = HI * (1 - 470 / 299792.458)
    per_kelvin = (1 * u.K).to(
        u.Jy / u.sr, u.brightness_temperature(frequency)) \
        * BEAM.area() * u.arcsec ** 2
    peak = plain.max()
    np.testing.assert_allclose(
        kelvin, plain * column / 1.823e18, rtol=1e-5,
        atol=1e-6 * peak * column / 1.823e18)
    np.testing.assert_allclose(
        jansky, kelvin * per_kelvin.to_value(u.Jy), rtol=1e-5,
        atol=1e-6 * kelvin.max() * per_kelvin.to_value(u.Jy))


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
