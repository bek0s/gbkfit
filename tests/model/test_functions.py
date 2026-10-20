"""
Tests of the functions of galaxies (see GalaxyFunctions), which the
derived values of their parameters use: the circular velocity of their
mass, and the rotation and dispersion of their disks, at any radius.
"""

import copy

import numpy as np
import pytest

from gbkfit.model import galaxy_parser


# The gravitational constant (kpc (km/s)^2 / Msun)
G = 4.30091727e-6

# The length of an arcsec (kpc) at 10 Mpc
KPC_PER_ARCSEC = 1e4 * np.pi / 648000

# A thick gas disk in the mass of a black hole and a halo: its rotation
# in that gravity, with jeans pressure support, and its dispersion with
# that of its central equilibrium
GAS = dict(
    type='smdisk', name='gas', loose=False, tilted=False,
    rnodes=list(range(0, 21, 2)), rstep=0.5,
    density=dict(
        ptraits=dict(type='exponential'),
        htraits=dict(type='gauss', zmax=10)),
    kinematics=dict(
        vptraits=dict(type='mass', pressure_support='jeans'),
        dptraits=dict(type='uniform'), equilibrium='central'),
    emission=dict(type='lines'))
MASS = [
    dict(type='point', name='bh'),
    dict(type='spheroid', name='halo', profile=dict(type='nfw'))]
PARAMS = dict(
    gas_xpos=0, gas_ypos=0, gas_posa=30, gas_incl=60, gas_vsys=0,
    gas_pt_a=1, gas_pt_s=4, gas_ht_s=0.5, gas_dpt_a=10, distance=10,
    mass_bh_log_m=10, mass_halo_log_m=12, mass_halo_c=8)

# A disk without mass: its rotation of arctan traits
DISK = dict(
    type='smdisk', loose=False, tilted=False, rnodes=list(range(0, 21, 2)),
    density=dict(ptraits=dict(type='exponential')),
    kinematics=dict(
        vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform')),
    emission=dict(type='lines'))
DISK_PARAMS = dict(
    xpos=0, ypos=0, posa=30, incl=60, vsys=0, pt_a=1, pt_s=4, vpt_rt=3,
    vpt_vt=200, dpt_a=20)


def gas_galaxy(**info):
    """The galaxy of the gas disk in its mass."""
    return galaxy_parser.load(dict(
        components=[copy.deepcopy(GAS)], mass=copy.deepcopy(MASS),
        distance=dict(type='direct', cosmology='Planck18')) | info)


def test_circular_velocity_of_the_mass():
    galaxy = gas_galaxy()
    functions = galaxy.plan_functions()
    assert functions.names() == (
        'vcirc', 'vnewton', 'vcirc_bh', 'vcirc_halo', 'mass', 'mass_bh',
        'mass_halo', 'kpc_per_arcsec', 'rotation_gas', 'dispersion_gas')
    bound = functions.bind(PARAMS)
    R = np.array([0.5, 2.0, 8.0])
    np.testing.assert_allclose(
        bound['vcirc_bh'](R), np.sqrt(G * 1e10 / R), rtol=1e-9)
    np.testing.assert_allclose(
        bound['vcirc'](R) ** 2,
        bound['vcirc_bh'](R) ** 2 + bound['vcirc_halo'](R) ** 2, rtol=1e-12)
    assert bound['kpc_per_arcsec'] == pytest.approx(KPC_PER_ARCSEC, rel=1e-9)
    # The enclosed masses, in total and of each component
    np.testing.assert_allclose(bound['mass_bh'](R), 1e10)
    np.testing.assert_allclose(
        bound['mass'](R), bound['mass_bh'](R) + bound['mass_halo'](R),
        rtol=1e-12)
    # (a number gives an array of no dimensions)
    assert bound['vcirc'](2.0).shape == ()
    # The same as the rotation table of the gravity
    extra = {}
    galaxy.plan_gravity().evaluate(PARAMS, extra)
    table = extra['gas_vpt_vt_rotation']
    np.testing.assert_array_equal(bound['vnewton'](R), bound['vcirc'](R))
    for name in ('vcirc', 'vnewton', 'vcirc_bh', 'vcirc_halo'):
        np.testing.assert_allclose(
            bound[name](table['radius_kpc'].value), table[name].value,
            rtol=1e-12)


def test_rotation_and_dispersion_as_the_gravity_gives_them():
    # At the subnodes, the rotation of the disk is that of its rotation
    # table (with pressure support), and its dispersion is its dispersion
    # traits with that of its equilibrium in quadrature; between them,
    # both are interpolated linearly, as the disk is evaluated (in a halo
    # alone, at 100 Mpc: next to a black hole, the pressure exceeds the
    # gravity at a subnode, whose rotation is NaN)
    galaxy = gas_galaxy(mass=[copy.deepcopy(MASS[1])])
    params = PARAMS | dict(distance=100)
    bound = galaxy.plan_functions().bind(params)
    extra = {}
    galaxy.plan_gravity().evaluate(params, extra)
    rotation = extra['gas_vpt_vt_rotation']
    radii = rotation['radius'].value
    assert np.all(np.isfinite(rotation['vrot']))
    assert np.any(rotation['pressure'].value != 0)
    np.testing.assert_allclose(
        bound['rotation_gas'](radii), rotation['vrot'].value, rtol=1e-12)
    middle = (radii[1:] + radii[:-1]) / 2
    np.testing.assert_allclose(
        bound['rotation_gas'](middle),
        (rotation['vrot'].value[1:] + rotation['vrot'].value[:-1]) / 2,
        rtol=1e-12)
    sigma = extra['gas_equilibrium']['sigma'].value
    np.testing.assert_allclose(
        bound['dispersion_gas'](radii), np.hypot(10, sigma), rtol=1e-12)
    np.testing.assert_allclose(
        bound['dispersion_gas'](middle),
        np.hypot(10, (sigma[1:] + sigma[:-1]) / 2), rtol=1e-12)


def test_functions_of_a_disk_without_mass():
    # Its functions are named after no component; its rotation is that of
    # its traits at any radius
    galaxy = galaxy_parser.load(dict(components=[copy.deepcopy(DISK)]))
    functions = galaxy.plan_functions()
    assert functions.names() == ('rotation', 'dispersion')
    bound = functions.bind(DISK_PARAMS)
    R = np.array([[0.0, 1.3], [7.1, 20.0]])
    np.testing.assert_allclose(
        bound['rotation'](R), 200 * 2 / np.pi * np.arctan(R / 3),
        rtol=1e-12)
    np.testing.assert_allclose(bound['dispersion'](R), 20, rtol=1e-12)


def test_radii_outside_refused():
    bound = gas_galaxy().plan_functions().bind(PARAMS)
    with pytest.raises(ValueError, match='rotation_gas: the radii must be'):
        bound['rotation_gas'](21)
    with pytest.raises(ValueError, match='vcirc_halo: the radii must be'):
        bound['vcirc_halo'](-1)


def test_functions_of_a_group_prefixed():
    from gbkfit.driver.drivers.host import DriverHost
    from gbkfit.observation import (
        Observation, ObservationGroup, PixelSpectra, line_of_sight_parser)
    galaxy = gas_galaxy(name='north')
    group = ObservationGroup([galaxy], [Observation(
        PixelSpectra(size=(8, 8, 5), step=(1, 1, 10)),
        line_of_sight_parser.load(dict(type='grid')), DriverHost())])
    functions = group.plan_functions()
    assert functions.names() == ('G',) + tuple(
        f'north_{name}' for name in galaxy.plan_functions().names())
    bound = functions.bind({f'north_{k}': v for k, v in PARAMS.items()})
    assert bound['G'] == pytest.approx(G, rel=1e-6)
    own = galaxy.plan_functions().bind(PARAMS)
    R = np.array([1.0, 5.0])
    np.testing.assert_array_equal(bound['north_vcirc'](R), own['vcirc'](R))
    np.testing.assert_array_equal(
        bound['north_rotation_gas'](R), own['rotation_gas'](R))


def test_derived_values_of_a_group():
    # The derived values of a configuration read the functions of its
    # galaxies (see ParamSpace)
    from gbkfit.driver.drivers.host import DriverHost
    from gbkfit.observation import (
        Observation, ObservationGroup, PixelSpectra, line_of_sight_parser)
    from gbkfit.params import param_space_parser
    group = ObservationGroup([gas_galaxy()], [Observation(
        PixelSpectra(size=(8, 8, 5), step=(1, 1, 10)),
        line_of_sight_parser.load(dict(type='grid')), DriverHost())])
    params = param_space_parser.load(
        dict(properties=PARAMS, derived=dict(
            v_bh='vcirc_bh(2.0)', m_bh='v_bh ** 2 * 2.0 / G',
            v_sigma='rotation_gas(10) / dispersion_gas(10)')),
        pdescs=group.pdescs(), constants=group.constants(),
        functions=group.plan_functions())
    derived = params.derived_values(params.evaluate())
    assert derived['v_bh'] == pytest.approx(np.sqrt(G * 1e10 / 2), rel=1e-9)
    assert derived['m_bh'] == pytest.approx(1e10, rel=1e-9)
    bound = gas_galaxy().plan_functions().bind(PARAMS)
    assert derived['v_sigma'] == pytest.approx(
        float(bound['rotation_gas'](10) / bound['dispersion_gas'](10)),
        rel=1e-12)
