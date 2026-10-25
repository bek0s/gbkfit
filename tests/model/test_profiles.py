"""
Tests of profile observations (see ObservationProfile): functions of a
galaxy at radii, e.g. its rotation curve, compared with tables of them.
"""

import copy

import astropy.units as u
import numpy as np
import pytest

from gbkfit.dataset import Data, DatasetProfile
from gbkfit.model import galaxy_parser
from gbkfit.objective import Objective
from gbkfit.observation import (
    Flat, NoiseGauss, ObservationGroup, ObservationProfile, ObservationSky,
    PixelBrightness, PixelSpectra, observation_parser)
from gbkfit.utils.parseutils import ConfigError


# A galaxy of mass alone: a black hole and a halo, at 10 Mpc
MASS = [
    dict(type='point', name='bh'),
    dict(type='spheroid', name='halo', profile=dict(type='nfw'))]
PARAMS = dict(
    distance=10, mass_bh_log_m=10, mass_halo_log_m=12, mass_halo_c=8)

# The length of an arcsec (kpc) at 10 Mpc
KPC_PER_ARCSEC = 1e4 * np.pi / 648000

# A gas disk that the mass moves
GAS = dict(
    type='smdisk', name='gas', loose=False, tilted=False,
    rnodes=list(range(0, 21, 2)),
    density=dict(ptraits=dict(type='exponential')),
    kinematics=dict(vptraits=dict(type='mass'), dptraits=dict(type='uniform')),
    emission=dict(type='lines'))
GAS_PARAMS = dict(
    gas_xpos=0, gas_ypos=0, gas_posa=30, gas_incl=60, gas_vsys=0,
    gas_pt_a=1, gas_pt_s=4, gas_dpt_a=10)


def mass_galaxy(**info):
    return galaxy_parser.load(
        dict(mass=copy.deepcopy(MASS), distance=dict(type='direct')) | info)


def model(group, params):
    return {key: item['d'].copy()
            for key, item in group.plan().model_h(params)[0].items()}


def test_rotation_curve_of_a_mass_model():
    # The model is the circular velocity of the mass at the radii in kpc:
    # angles at the distance of the galaxy, lengths measured at another
    # distance as their angles, and physical lengths as they are
    galaxy = mass_galaxy()
    vcirc = galaxy.plan_functions().bind(PARAMS)['vcirc']
    radii = np.array([0.5, 2.0, 8.0])
    for profile_radii, distance, expected in (
            (radii * u.arcsec, None, radii * KPC_PER_ARCSEC),
            (radii * u.kpc, 20 * u.Mpc, radii / 2),
            (radii * u.kpc, None, radii)):
        observation = ObservationProfile(['vcirc'], profile_radii, distance)
        group = ObservationGroup(galaxy, observation)
        np.testing.assert_allclose(
            model(group, PARAMS)['vcirc'], vcirc(expected), rtol=1e-12)
    # Data of the model have no residuals
    data = observation.dataset({'vcirc': Data(vcirc(radii), error=np.ones(3))})
    objective = Objective(
        ObservationGroup(galaxy, observation.with_data(data)).plan())
    assert objective.chi2(PARAMS) == pytest.approx(0, abs=1e-20)


def test_inclination_scales_rotation_velocities():
    galaxy = mass_galaxy()
    radii = np.array([1.0, 4.0]) * u.kpc
    observation = ObservationProfile(
        ['vcirc', 'mass'], radii, inclination=60, name='rc')
    assert set(observation.pdescs()) == {'rc_incl'}
    group = ObservationGroup(galaxy, observation)
    at_60 = model(group, PARAMS | dict(rc_incl=60))
    at_30 = model(group, PARAMS | dict(rc_incl=30))
    np.testing.assert_allclose(
        at_30['vcirc'], at_60['vcirc'] * 0.5 / np.sin(np.radians(60)))
    np.testing.assert_array_equal(at_30['mass'], at_60['mass'])
    with pytest.raises(ConfigError, match="needs a name"):
        ObservationProfile(['vcirc'], radii, inclination=60)
    with pytest.raises(ConfigError, match="has none"):
        ObservationProfile(['mass'], radii, inclination=60, name='rc')


def test_profile_of_a_table_file(tmp_path):
    # A text table without a header (as those of SPARC), in kpc at its
    # distance; the dump writes an ECSV table, which loads back
    file = tmp_path / 'rotmod.dat'
    file.write_text(
        "# Distance = 20 Mpc\n"
        "1.0 100.0 5.0\n"
        "2.0 150.0 4.0\n"
        "4.0 170.0 nan\n")
    info = dict(
        type='profile', name='rc',
        data=dict(
            file=str(file), format='ascii.no_header',
            radius=dict(column='col1', unit='kpc'), distance=20,
            vcirc=dict(column='col2', error='col3')))
    observation = observation_parser.load(copy.deepcopy(info))
    data = observation.data()
    assert observation.keys() == ('vcirc',)
    np.testing.assert_array_equal(observation.radii(), [1, 2, 4] * u.kpc)
    assert observation.distance() == 20 * u.Mpc
    # (a value without an error is left out)
    np.testing.assert_array_equal(data['vcirc'].mask(), [1, 1, 0])
    dumped = observation_parser.dump(
        observation, prefix=str(tmp_path / 'dumped_'))
    again = observation_parser.load(dumped)
    np.testing.assert_array_equal(again.radii(), observation.radii())
    np.testing.assert_array_equal(
        again.data()['vcirc'].data(), data['vcirc'].data())
    assert again.distance() == observation.distance()


def test_profile_beside_a_sky_observation(monkeypatch):
    # The rotation of a disk in the mass, from a profile first and a cube
    # second: the frames are those of the cube, and the mass of the galaxy
    # is made once in an evaluation, for both
    from gbkfit.model.components._stage import GalaxyStage
    galaxy = mass_galaxy(components=[copy.deepcopy(GAS)])
    radii = np.array([2.0, 6.0, 10.0])
    profile = ObservationProfile(['rotation_gas'], radii * u.arcsec)
    cube = ObservationSky(
        PixelSpectra(size=(8, 8, 5), step=(1, 1, 10)), Flat())
    group = ObservationGroup(galaxy, [profile, cube])
    params = PARAMS | GAS_PARAMS
    masses = []
    original = GalaxyStage.mass
    monkeypatch.setattr(
        GalaxyStage, 'mass',
        lambda self, *args: masses.append(1) or original(self, *args))
    rotation = model(group, params)['rotation_gas']
    assert len(masses) == 1
    expected = galaxy.plan_functions().bind(params)['rotation_gas'](radii)
    np.testing.assert_allclose(rotation, expected, rtol=1e-12)


def test_mass_that_follows_light_of_a_rotation_curve(caplog):
    # A galaxy whose mass follows the light of a component it does not
    # move (e.g. an image and a rotation curve that fit a mass-to-light
    # ratio): the profile takes its mass; without it, nothing but derived
    # values can, which a warning says
    light = dict(
        type='smdisk', name='stars', loose=False, tilted=False,
        rnodes=list(range(0, 21, 2)),
        density=dict(ptraits=dict(type='exponential')),
        emission=dict(type='band'))
    galaxy = galaxy_parser.load(dict(
        components=[light], distance=dict(type='direct'), mass=[dict(
            type='disk', name='stellar', vertical=dict(type='thin'),
            surface=dict(type='component', component='stars'))]))
    image = ObservationSky(PixelBrightness(size=(8, 8), step=(1, 1)), Flat())
    profile = ObservationProfile(['vcirc'], np.array([2.0, 6.0]) * u.arcsec)
    group = ObservationGroup(galaxy, [image, profile])
    params = dict(
        stars_xpos=0, stars_ypos=0, stars_posa=30, stars_incl=60,
        stars_pt_a=1, stars_pt_s=4, distance=10, mass_stellar_upsilon=1e8)
    assert np.all(group.plan().model_h(params)[1]['vcirc']['d'] > 0)
    ObservationGroup(galaxy, image)
    warnings = [record for record in caplog.records
                if 'only derived values can use it' in record.message]
    assert len(warnings) == 1


def test_mock_profiles():
    galaxy = mass_galaxy()
    observation = ObservationProfile(['vcirc'], [1.0, 3.0] * u.kpc)
    group = ObservationGroup(galaxy, observation)
    (mock,), = group.plan().mock(PARAMS, NoiseGauss(rms=5.0), seed=1)
    assert isinstance(mock, DatasetProfile)
    np.testing.assert_array_equal(mock['vcirc'].error(), [5, 5])
    assert np.all(np.abs(mock['vcirc'].data() - model(group, PARAMS)['vcirc'])
                  < 5 * 5)


def test_profiles_refuse_what_they_cannot_measure():
    galaxy = mass_galaxy(components=[copy.deepcopy(GAS)])
    radii = [1.0, 3.0] * u.kpc
    with pytest.raises(ConfigError, match="not a function of radii"):
        ObservationProfile(['vrot'], radii)
    with pytest.raises(ConfigError, match="has no functions"):
        ObservationGroup(
            galaxy, ObservationProfile(['vcirc_disk'], radii)).plan()
    with pytest.raises(ConfigError, match="physical lengths"):
        ObservationGroup(
            galaxy, ObservationProfile(['rotation_gas'], radii)).plan()
    with pytest.raises(ConfigError, match="must be on its disk"):
        ObservationGroup(galaxy, ObservationProfile(
            ['rotation_gas'], [1.0, 30.0] * u.arcsec)).plan()
    with pytest.raises(ConfigError, match="give them in km / s"):
        ObservationProfile(data=DatasetProfile(
            {'vcirc': Data(np.ones(2), unit='m/s')}, radii))


def test_outputs_of_profiles_are_tables():
    from gbkfit.tasks import _detail
    observation = ObservationProfile(['vcirc'], [1.0, 3.0] * u.kpc, name='rc')
    data = observation.dataset({'vcirc': Data(
        [100.0, 150.0], error=[5.0, 5.0], unit='km/s')})
    plan = ObservationGroup(mass_galaxy(), observation.with_data(data)).plan()
    # (with the extra outputs of the gravity of a galaxy of mass alone)
    outputs = _detail.evaluation_outputs(
        plan, Objective(plan), PARAMS, True, True)
    assert {
        f'observations/rc/{kind}/vcirc.ecsv'
        for kind in ('model', 'residual', 'wresidual')} <= set(outputs)
    table = outputs['observations/rc/model/vcirc.ecsv'].value
    np.testing.assert_array_equal(table['radius'], [1, 3] * u.kpc)
    assert table['value'].unit == u.km / u.s


# A rotmod file of SPARC, made up: radius (kpc), Vobs, errV, Vgas,
# Vdisk, Vbul (km/s), at 10 Mpc
ROTMOD = (
    "# Distance = 10 Mpc\n"
    "# Rad Vobs errV Vgas Vdisk Vbul\n"
    "0.5 40.0 5.0 -3.0 50.0 0.0\n"
    "1.0 70.0 4.0 5.0 80.0 0.0\n"
    "2.0 100.0 4.0 12.0 95.0 0.0\n"
    "4.0 120.0 5.0 20.0 90.0 0.0\n")


def rotmod_info(file, column):
    """The configuration of a curve of a rotmod file of SPARC."""
    return dict(
        file=str(file), format='ascii.no_header',
        radius=dict(column='col1', unit='kpc'), distance=10,
        velocity=dict(column=column))


def test_table_mass_of_its_curve(tmp_path):
    # v^2 = upsilon V |V| at its radii, signed; at twice its distance its
    # radii and V^2 are twice theirs
    from gbkfit.model import mass_component_parser
    file = tmp_path / 'rotmod.dat'
    file.write_text(ROTMOD)
    gas = mass_component_parser.load(
        dict(type='table', name='gas') | rotmod_info(file, 'col4'))
    galaxy = galaxy_parser.load(dict(
        mass=[mass_component_parser.dump(gas)], distance=dict(type='direct')))
    assert 'mass' not in galaxy.plan_functions().names()
    radii = np.array([0.5, 1.0, 2.0, 4.0])
    gas_v = np.array([-3.0, 5.0, 12.0, 20.0])
    for distance, scale in ((10, 1), (20, 2)):
        bound = galaxy.plan_functions().bind(
            dict(distance=distance, mass_gas_upsilon=1.5))
        np.testing.assert_allclose(
            bound['vcirc'](radii * scale) ** 2 * np.sign(gas_v),
            1.5 * scale * gas_v * np.abs(gas_v), rtol=1e-12)
    with pytest.raises(ValueError, match="has a table to the radius 4"):
        bound['vcirc'](9.0)


def test_rotation_curve_of_sparc(tmp_path):
    # A fit of SPARC: the gas and the disk of the rotmod file, and a halo;
    # the observed curve of the file, of its distance
    file = tmp_path / 'rotmod.dat'
    file.write_text(ROTMOD)
    galaxy = galaxy_parser.load(dict(
        mass=[dict(type='table', name='gas') | rotmod_info(file, 'col4'),
              dict(type='table', name='disk') | rotmod_info(file, 'col5'),
              dict(type='spheroid', name='halo', profile=dict(type='nfw'))],
        distance=dict(type='direct')))
    observation = observation_parser.load(dict(
        type='profile', name='rc', inclination=60,
        data=dict(
            file=str(file), format='ascii.no_header',
            radius=dict(column='col1', unit='kpc'), distance=10,
            vcirc=dict(column='col2', error='col3'))))
    params = dict(
        distance=10, mass_gas_upsilon=1, mass_disk_upsilon=0.5,
        mass_halo_log_m=11.5, mass_halo_c=10, rc_incl=60)
    objective = Objective(ObservationGroup(galaxy, observation).plan())
    radii = np.array([0.5, 1.0, 2.0, 4.0])
    halo = galaxy.plan_functions().bind(params)['vcirc_halo'](radii)
    gas = np.array([-3.0, 5.0, 12.0, 20.0])
    disk = np.array([50.0, 80.0, 95.0, 90.0])
    vcirc = np.sqrt(gas * np.abs(gas) + 0.5 * disk ** 2 + halo ** 2)
    vobs, errv = np.array([40.0, 70, 100, 120]), np.array([5.0, 4, 4, 5])
    assert objective.chi2(params) == pytest.approx(
        np.sum(((vcirc - vobs) / errv) ** 2), rel=1e-10)
