"""
Tests of the velocity dispersion along the directions of a disk (see
traits): the velocity ellipsoid of sigma_R, sigma_phi and sigma_z seen on
the line of sight, and the configurations it refuses.
"""

import copy

import numpy as np
import pytest

from gbkfit.model import galaxy_parser
from gbkfit.utils.parseutils import ConfigError


# A disk of a uniform rotation (tangential), thin or thick
def disk(dptraits, thick=False):
    density = dict(ptraits=dict(type='exponential'))
    if thick:
        density['htraits'] = dict(type='sech2', zmax=8)
    return dict(
        type='disk', loose=False, tilted=False,
        rnodes=list(range(0, 21, 2)), density=density,
        kinematics=dict(
            vptraits=dict(type='uniform', direction='tan'),
            dptraits=dptraits),
        emission=dict(type='lines'))


# (off the centre of a pixel, where the azimuth is undefined)
PROPERTIES = dict(
    xpos=0.25, ypos=-0.25, posa=30, vsys=0, pt_a=1, pt_s=5, vpt_vt=150)


def properties(thick, **values):
    """The parameter properties of a thin or thick disk."""
    return PROPERTIES | (dict(ht_s=1) if thick else {}) | values

# The traits along every direction (sigma_R, sigma_phi, sigma_z)
DIRECTIONS = [
    dict(type='uniform', direction='rad'),
    dict(type='uniform', direction='tan'),
    dict(type='uniform', direction='ver')]


def case(driver, galaxy):
    """A galaxy, seen in a cube."""
    return dict(
        driver=dict(type=driver.type()), galaxy=dict(components=[galaxy]),
        line_of_sight=dict(type='grid'),
        observable=dict(
            type='pixel_spectra', size=[48, 48, 50], step=[1, 1, 10]))


@pytest.mark.parametrize('thick', [False, True])
@pytest.mark.parametrize('incl', [30, 60, 80])
def test_dispersion_is_the_velocity_ellipsoid_on_the_line_of_sight(
        driver, evaluate_cases, thick, incl):
    # sigma_R = 10 + 20, sigma_phi = 10 + 5 and sigma_z = 10 (of the
    # isotropic trait alone); the azimuth of each voxel from its velocity,
    # vt cos(theta) sin(incl)
    dptraits = [
        dict(type='uniform'), dict(type='uniform', direction='rad'),
        dict(type='uniform', direction='tan')]
    _, extra = evaluate_cases(
        [case(driver, disk(dptraits, thick))],
        properties(thick, incl=incl, dpt_a=10, dpt1_a=20, dpt2_a=5))
    on = extra.get('bdata', component='component0').data > 0
    velocity = extra.get('vdata', component='component0').data[on]
    sigma = extra.get('ddata', component='component0').data[on]
    i = np.radians(incl)
    cos2 = (velocity / (150 * np.sin(i))) ** 2
    expected = np.sin(i) ** 2 * (15 ** 2 * cos2 + 30 ** 2 * (1 - cos2)) \
        + 10 ** 2 * np.cos(i) ** 2
    np.testing.assert_allclose(sigma ** 2, expected, rtol=1e-4)


@pytest.mark.parametrize('thick', [False, True])
def test_the_same_dispersion_along_every_direction_is_isotropic(
        driver, evaluate_cases, thick):
    values = properties(thick, incl=60, dpt_a=20)
    (isotropic,), _ = evaluate_cases(
        [case(driver, disk(dict(type='uniform'), thick))], values)
    (directions,), _ = evaluate_cases(
        [case(driver, disk(DIRECTIONS, thick))],
        values | dict(dpt1_a=20, dpt2_a=20))
    expected = isotropic['spectra']['d']
    np.testing.assert_allclose(
        directions['spectra']['d'], expected, rtol=1e-5,
        atol=1e-6 * np.max(expected))


def kinematics(**options):
    """A galaxy of a thick disk of the given kinematics."""
    thick = disk(None, thick=True)
    thick['kinematics'] |= options
    return dict(components=[thick])


@pytest.mark.parametrize('options, message', [
    (dict(dptraits=DIRECTIONS[:2]),
     "no dispersion trait for the direction 'ver'; give one, or an 'iso'"),
    (dict(dptraits=dict(type='uniform', direction='out')),
     "direction must be one of"),
    (dict(dptraits=dict(type='uniform'), equilibrium_direction='ver'),
     "it needs equilibrium"),
    (dict(dptraits=dict(type='uniform'), equilibrium='central',
          equilibrium_direction='rad'),
     r"equilibrium_direction must be one of \['iso', 'ver'\]"),
])
def test_dispersion_directions_are_checked(options, message):
    with pytest.raises(ConfigError, match=message):
        galaxy_parser.load(kinematics(**options))


def test_a_cone_has_isotropic_dispersion_only():
    cone = dict(
        type='cone', rnodes=[4, 10],
        density=dict(
            ptraits=dict(type='uniform'),
            htraits=dict(type='gauss', zmax=1.5)),
        kinematics=dict(
            vptraits=dict(type='uniform', direction='out'),
            dptraits=DIRECTIONS),
        emission=dict(type='lines'))
    with pytest.raises(
            ConfigError, match=r"takes dispersion traits along \['iso'\]"):
        galaxy_parser.load(dict(components=[cone]))


def test_dispersion_directions_are_dumped():
    # (but the isotropic ones, by default)
    info = kinematics(
        dptraits=[dict(type='uniform')] + DIRECTIONS[:1],
        equilibrium='central', equilibrium_direction='ver')
    dumped = galaxy_parser.dump(galaxy_parser.load(copy.deepcopy(info)))
    dumped = dumped['components'][0]['kinematics']
    assert dumped['dptraits'] == [
        dict(type='uniform'), dict(type='uniform', direction='rad')]
    assert dumped['equilibrium_direction'] == 'ver'
