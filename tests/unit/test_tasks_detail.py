"""Tests for the helpers of the tasks."""

import numpy as np
import pytest

import gbkfit.project
from gbkfit.tasks import _detail
from gbkfit.utils.parseutils import ConfigError


GALAXY = dict(components=[dict(
    type='smdisk', loose=False, tilted=False, rnmin=0, rnmax=4, rnsep=1,
    density=dict(ptraits=dict(type='exponential')),
    emission=dict(type='band'))])

HOST = dict(type='host')


def test_make_output_dir(tmp_path):
    path = tmp_path / 'out'
    assert _detail.make_output_dir(str(path), 'terminate') == str(path)
    assert path.is_dir()
    with pytest.raises(RuntimeError, match="already exists"):
        _detail.make_output_dir(str(path), 'terminate')
    assert _detail.make_output_dir(str(path), 'overwrite') == str(path)
    # Each unique directory is new
    first = _detail.make_output_dir(str(path), 'unique')
    second = _detail.make_output_dir(str(path), 'unique')
    assert first == str(tmp_path / 'out_1')
    assert second == str(tmp_path / 'out_2')
    (tmp_path / 'file').write_text('')
    with pytest.raises(RuntimeError, match="exists as a file"):
        _detail.make_output_dir(str(tmp_path / 'file'), 'overwrite')


def test_configs_resolve_relative_paths_against_their_root(tmp_path):
    # (their directory, or their section 'root', relative to it, or the
    # folder of their project)
    config = tmp_path / 'configs' / 'fit.yaml'
    config.parent.mkdir()
    config.write_text("galaxies: {}\nroot: ../data\n")
    file = _detail.read_config(str(config), ('galaxies',))
    assert file.config == dict(galaxies=[{}])
    assert file.root == str(tmp_path / 'data')
    config.write_text("galaxies: {}\n")
    assert _detail.read_config(str(config), ('galaxies',)).root == \
        str(config.parent)
    with pytest.raises(ConfigError, match="needs the sections"):
        _detail.read_config(str(config), ('galaxies', 'fitter'))
    gbkfit.project.create(str(tmp_path))
    assert _detail.read_config(str(config), ('galaxies',)).root == \
        str(tmp_path)


def observation(**options):
    """The configuration of an observation of GALAXY, with options."""
    return dict(
        type='sky',
        line_of_sight=dict(type='flat'),
        observable=dict(type='pixel_brightness', size=[8, 8])) | options


def load_group(**sections):
    """The ObservationGroup of GALAXY and the sections of a configuration."""
    return _detail.load_observation_group(dict(galaxies=[GALAXY]) | sections)


def test_observations_use_the_drivers_of_the_configuration():
    # One driver: every observation has it
    group = load_group(
        driver=HOST,
        observations=[observation(name='a'), observation(name='b')])
    a, b = [obs.driver() for obs in group.observations()]
    assert a is b
    # Several: each observation names its own
    group = load_group(
        drivers=dict(cpu1=HOST, cpu2=HOST),
        observations=[
            observation(name='a', driver='cpu2'),
            observation(name='b', driver='cpu1'),
            observation(name='c', driver='cpu2')])
    a, b, c = [obs.driver() for obs in group.observations()]
    assert a is c and a is not b
    # None: the host
    group = load_group(observations=[observation()])
    assert group.observations()[0].driver().type() == 'host'


# A galaxy with mass, seen as a spectral cube
MASSIVE = dict(
    components=[dict(
        type='smdisk', loose=False, tilted=False, rnmin=0, rnmax=4,
        rnsep=1, density=dict(ptraits=dict(type='exponential')),
        kinematics=dict(
            vptraits=dict(type='mass'), dptraits=dict(type='uniform')),
        emission=dict(type='lines'))],
    mass=[dict(type='spheroid', name='halo', profile=dict(
        type='pseudo_isothermal', parameters='density_radius'))],
    distance=dict(type='direct'))

CUBE = dict(
    type='sky',
    line_of_sight=dict(type='flat'),
    observable=dict(type='pixel_spectra', size=[8, 8, 5]))


def test_galaxies_with_mass_use_the_drivers_of_the_configuration():
    # (for their gravity, as observations do for their models)
    def drivers(galaxy, **sections):
        group = _detail.load_observation_group(
            dict(galaxies=[galaxy]) | sections)
        return group.galaxies()[0].driver(), group.observations()[0].driver()
    # One driver: the galaxy has it; several: it names its own
    galaxy, observation = drivers(MASSIVE, driver=HOST, observations=[CUBE])
    assert galaxy is observation
    several = dict(drivers=dict(cpu1=HOST, cpu2=HOST),
                   observations=[CUBE | dict(driver='cpu1')])
    galaxy, observation = drivers(MASSIVE | dict(driver='cpu1'), **several)
    assert galaxy is observation
    galaxy, observation = drivers(MASSIVE | dict(driver='cpu2'), **several)
    assert galaxy is not observation
    with pytest.raises(ConfigError, match="the galaxy must name its driver"):
        drivers(MASSIVE, **several)
    # None: the host
    galaxy, _ = drivers(MASSIVE, observations=[CUBE])
    assert galaxy.type() == 'host'


@pytest.mark.parametrize('sections, message', [
    (dict(drivers=dict(cpu1=HOST, cpu2=HOST), observations=[observation()]),
     "must name its driver"),
    (dict(drivers=dict(cpu=HOST), observations=[observation(driver='gpu')]),
     "unknown driver 'gpu'"),
    (dict(driver=HOST, observations=[observation(driver='cpu')]),
     "unknown driver 'cpu'"),
    (dict(driver=HOST, drivers=dict(cpu=HOST), observations=[observation()]),
     "not both"),
    # (the form of configurations before drivers were declared at the top)
    (dict(observations=[observation(driver=HOST)]),
     "declares at its top level"),
])
def test_driver_errors(sections, message):
    with pytest.raises(ConfigError, match=message):
        load_group(**sections)


def test_parallel_evaluations_are_on_drivers_of_the_configuration(tmp_path):
    from astropy.io import fits
    from gbkfit.objective import Objective
    fits.writeto(tmp_path / 'image.fits', np.zeros((8, 8)))
    cfg = dict(
        galaxies=[GALAXY], drivers=dict(cpu1=HOST, cpu2=HOST),
        observations=[dict(
            type='sky',
            line_of_sight=dict(type='flat'),
            observable=dict(type='pixel_brightness'),
            data=dict(data=str(tmp_path / 'image.fits')), driver='cpu1')])
    drivers = _detail.load_drivers(cfg)
    objective = Objective(_detail.load_observation_group(cfg, drivers))
    with _detail.load_pool(
            cfg | dict(parallel=dict(drivers=['cpu1', 'cpu2'])),
            objective, drivers) as pool:
        first, second = pool.copies()
        assert first is objective
        assert second.group().observations()[0].driver() is drivers['cpu2']
    assert _detail.load_pool(cfg, objective, drivers) is None
    with pytest.raises(ConfigError, match="unknown driver 'gpu'"):
        _detail.load_pool(
            cfg | dict(parallel=dict(drivers=['gpu'])), objective, drivers)


def test_depth_order_has_galaxies_and_groups_of_them():
    from gbkfit.observation import DepthGroup
    order = _detail.load_depth_order(['fg', {'pair': ['a', 'b']}])
    assert order == ['fg', DepthGroup('pair', ('a', 'b'))]
    with pytest.raises(ConfigError, match="a group of depth_order is"):
        _detail.load_depth_order([{'pair': ['a', 'b'], 'other': ['c']}])
