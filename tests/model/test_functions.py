"""
Tests of the functions of galaxies (see GalaxyFunctions), which the
derived values of their parameters use: the circular velocity of their
mass, and the rotation and dispersion of their disks, at any radius.
"""

import copy

import numpy as np
import pytest

from gbkfit.model import Extras, galaxy_parser


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
        'mass_halo', 'slope', 'slope_bh', 'slope_halo', 'kpc_per_arcsec',
        'rotation_gas', 'dispersion_gas')
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
    extra = Extras()
    galaxy.plan_gravity().evaluate(PARAMS, extra)
    table = extra.get('vpt_vt_rotation', component='gas')
    np.testing.assert_array_equal(bound['vnewton'](R), bound['vcirc'](R))
    for name in ('vcirc', 'vnewton', 'vcirc_bh', 'vcirc_halo'):
        np.testing.assert_allclose(
            bound[name](table['radius_kpc'].value), table[name].value,
            rtol=1e-12)


def test_density_slopes_of_the_mass():
    # Of a halo, the slope of its density: -1 - 2 x / (1 + x) of an NFW
    # halo (x = r / r_s); of a point mass, none (no density away from it);
    # of both, that of the halo (the point mass adds mass, not density)
    mass = [MASS[0], dict(
        type='spheroid', name='halo',
        profile=dict(type='nfw', parameters='density_radius'))]
    params = {k: v for k, v in PARAMS.items()
              if not k.startswith('mass_halo')}
    params |= dict(mass_halo_log_rho=7, mass_halo_rs=12)
    bound = gas_galaxy(mass=copy.deepcopy(mass)).plan_functions().bind(params)
    r = np.array([0.5, 3.0, 12.0, 40.0])
    x = r / 12
    np.testing.assert_allclose(
        bound['slope_halo'](r), -1 - 2 * x / (1 + x), rtol=1e-5)
    assert np.isnan(bound['slope_bh'](r)).all()
    np.testing.assert_allclose(
        bound['slope'](r), bound['slope_halo'](r), rtol=1e-4)
    with pytest.raises(ValueError, match="finite and positive"):
        bound['slope'](0.0)


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
    extra = Extras()
    galaxy.plan_gravity().evaluate(params, extra)
    rotation = extra.get('vpt_vt_rotation', component='gas')
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
    sigma = extra.get('equilibrium', component='gas')['sigma'].value
    np.testing.assert_allclose(
        bound['dispersion_gas'](radii), np.hypot(10, sigma), rtol=1e-12)
    np.testing.assert_allclose(
        bound['dispersion_gas'](middle),
        np.hypot(10, (sigma[1:] + sigma[:-1]) / 2), rtol=1e-12)


def test_pressure_support_of_the_velocity_ellipsoid():
    # The radial Jeans equation of the velocity ellipsoid: the pressure of
    # sigma_R, and sigma_R^2 - sigma_phi^2 (each with the dispersion of the
    # central equilibrium in quadrature), at the subnodes; the same
    # dispersion along every direction is isotropic
    def with_dispersion(dptraits):
        gas = copy.deepcopy(GAS)
        gas['kinematics']['dptraits'] = dptraits
        return gas_galaxy(components=[gas], mass=[copy.deepcopy(MASS[1])])

    def rotation_table(galaxy, params):
        extra = Extras()
        galaxy.plan_gravity().evaluate(params, extra)
        return extra.get('vpt_vt_rotation', component='gas'), extra

    directions = [
        dict(type='uniform', direction=direction)
        for direction in ('rad', 'tan', 'ver')]
    params = PARAMS | dict(distance=100, gas_dpt_a=30)
    isotropic, extra = rotation_table(
        with_dispersion(dict(type='uniform')), params)
    expected = isotropic['pressure'].value
    same, _ = rotation_table(
        with_dispersion(directions),
        params | dict(gas_dpt1_a=30, gas_dpt2_a=30))
    np.testing.assert_array_equal(same['pressure'].value, expected)
    values = params | dict(gas_dpt1_a=20, gas_dpt2_a=10)
    ellipsoid, _ = rotation_table(with_dispersion(directions), values)
    np.testing.assert_allclose(
        ellipsoid['pressure'].value, expected + 30 ** 2 - 20 ** 2,
        rtol=1e-12)
    # Its dispersion along each direction
    radii = isotropic['radius'].value
    sigma = extra.get('equilibrium', component='gas')['sigma'].value
    bound = with_dispersion(directions).plan_functions().bind(values)
    for direction, value in dict(rad=30, tan=20, ver=10).items():
        np.testing.assert_allclose(
            bound['dispersion_gas'](radii, direction),
            np.hypot(value, sigma), rtol=1e-12)
    with pytest.raises(ValueError, match='give its direction'):
        bound['dispersion_gas'](radii)


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


def test_functions_of_a_galaxy_of_mass_alone():
    # (e.g. for its rotation curve) Its parameters are those of its mass
    # and its distance; observations on the sky see nothing of it
    from gbkfit.observation import Flat, ObservationGroup, ObservationSky
    from gbkfit.observation import PixelBrightness
    from gbkfit.utils.parseutils import ConfigError
    info = dict(mass=copy.deepcopy(MASS), distance=dict(type='direct'))
    galaxy = galaxy_parser.load(copy.deepcopy(info))
    assert galaxy.components() == ()
    assert set(galaxy.pdescs()) == {
        'distance', 'mass_bh_log_m', 'mass_halo_log_m', 'mass_halo_c'}
    dumped = galaxy_parser.dump(galaxy)
    assert 'components' not in dumped
    assert galaxy_parser.load(dumped).pdescs().keys() == galaxy.pdescs().keys()
    bound = galaxy.plan_functions().bind(PARAMS)
    np.testing.assert_allclose(
        bound['vcirc_bh'](2.0), np.sqrt(G * 1e10 / 2.0), rtol=1e-9)
    with pytest.raises(ConfigError, match="it sees none"):
        ObservationGroup(
            galaxy, ObservationSky(PixelBrightness(size=(4, 4)), Flat()))
    with pytest.raises(ConfigError, match="at least one component, or mass"):
        galaxy_parser.load(dict())


def test_radii_outside_refused():
    bound = gas_galaxy().plan_functions().bind(PARAMS)
    with pytest.raises(ValueError, match='rotation_gas: the radii must be'):
        bound['rotation_gas'](21)
    with pytest.raises(ValueError, match='vcirc_halo: the radii must be'):
        bound['vcirc_halo'](-1)


def test_functions_of_a_group_prefixed():
    from gbkfit.driver.drivers.host import DriverHost
    from gbkfit.observation import (
        ObservationGroup, ObservationSky, PixelSpectra, line_of_sight_parser)
    galaxy = gas_galaxy(name='north')
    group = ObservationGroup([galaxy], [ObservationSky(
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
        ObservationGroup, ObservationSky, PixelSpectra, line_of_sight_parser)
    from gbkfit.params import param_space_parser
    group = ObservationGroup([gas_galaxy()], [ObservationSky(
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
