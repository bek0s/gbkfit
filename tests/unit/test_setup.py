"""Tests for the front door: configurations set up (gbkfit.setup)."""

import numpy as np
import pytest
from astropy.io import fits

import gbkfit
import gbkfit.project
from gbkfit import setup
from gbkfit.utils.parseutils import ConfigError


GALAXY = dict(components=[dict(
    type='disk', loose=False, tilted=False, rnmin=0, rnmax=4, rnsep=1,
    density=dict(ptraits=dict(type='exponential')),
    emission=dict(type='band'))])

PARAMS = dict(properties=dict(
    xpos=0, ypos=0, posa=30, incl=50, pt_a=1,
    pt_s=dict(value=2, min=1, max=4)))

HOST = dict(type='host')

VERSION = gbkfit.project.FORMAT_VERSION


def observation(**options):
    """The configuration of an observation of GALAXY, with options."""
    return dict(
        type='sky',
        line_of_sight=dict(type='flat'),
        observable=dict(type='pixel_brightness', size=[8, 8])) | options


def image(path):
    """Write an image of 8 x 8 zeros, and return its observation."""
    fits.writeto(path, np.zeros((8, 8)))
    return dict(
        type='sky', line_of_sight=dict(type='flat'),
        observable=dict(type='pixel_brightness'),
        data=dict(data=path.name))


def test_the_front_door_is_gbkfit_load():
    assert gbkfit.load is setup.load
    assert gbkfit.Setup is setup.Setup


def test_a_configuration_with_data_sets_up_its_objective(tmp_path):
    # (a dictionary: its paths are relative to its section 'root')
    config = dict(
        version=VERSION, root=str(tmp_path), galaxies=[GALAXY],
        observations=[image(tmp_path / 'image.fits')], params=PARAMS)
    with gbkfit.load(config) as loaded:
        evaluation = loaded.evaluate()
        assert evaluation.objective.merits.chi2 > 0
        assert loaded.fitter is None and loaded.pool is None
        outputs = loaded.outputs(evaluation, data=True)
        assert {'model', 'residual', 'wresidual', 'data'} <= {
            output.kind for output in outputs.values()}


def test_a_setup_in_python_is_that_of_its_configuration(tmp_path):
    # (of the objects of the sections of the configuration: the group of
    # its galaxies and observations, and its parameters)
    from gbkfit.model import galaxy_parser
    from gbkfit.observation import ObservationGroup, observation_parser
    from gbkfit.params import ParamSpace
    obs = image(tmp_path / 'image.fits') | dict(
        data=dict(data=str(tmp_path / 'image.fits')))
    with gbkfit.load(dict(
            version=VERSION, galaxies=[GALAXY], observations=[obs],
            params=PARAMS)) as loaded:
        expected = loaded.evaluate().objective.merits
    group = ObservationGroup(
        galaxy_parser.load(GALAXY), observation_parser.load(obs))
    params = ParamSpace(group.pdescs(), PARAMS['properties'])
    # (to rounding: the host kernels sum in no fixed order)
    with gbkfit.Setup(group, params) as built:
        assert tuple(built.evaluate().objective.merits) == pytest.approx(
            tuple(expected), rel=1e-12)


def test_a_setup_takes_the_parameters_of_its_group():
    from gbkfit.model import galaxy_parser
    from gbkfit.observation import ObservationGroup, observation_parser
    from gbkfit.params import ParamScalarDesc, ParamSpace
    group = ObservationGroup(
        galaxy_parser.load(GALAXY), observation_parser.load(observation()))
    with pytest.raises(ConfigError, match="they lack"):
        gbkfit.Setup(group, ParamSpace(dict(a=ParamScalarDesc()), dict(a=1)))
    params = ParamSpace(
        group.pdescs(), PARAMS['properties'], function_names=['vcirc'])
    with pytest.raises(ConfigError, match=r"no functions \['vcirc'\]"):
        gbkfit.Setup(group, params)


def test_the_outputs_of_an_evaluation_are_its_own():
    # (the plan reuses its arrays of the models: the outputs of a second
    # evaluation must not overwrite those of the first)
    loaded = gbkfit.load(dict(
        version=VERSION, galaxies=[GALAXY], observations=[observation()],
        params=PARAMS))
    first = loaded.outputs(loaded.evaluate())
    model = next(o for o in first.values() if o.kind == 'model')
    before = np.array(model.value.data)
    loaded.outputs(loaded.evaluate(dict(pt_s=3.0)))
    np.testing.assert_array_equal(model.value.data, before)


def test_a_configuration_without_data_cannot_fit():
    loaded = gbkfit.load(dict(
        version=VERSION, galaxies=[GALAXY], observations=[observation()],
        params=PARAMS))
    assert loaded.objective is None
    model = loaded.group_plan.model_h(loaded.evaluate().values)[0]
    assert model['brightness']['d'].sum() > 0
    with pytest.raises(ConfigError, match="a fit needs a fitter"):
        loaded.fit()
    # (a configuration of a fit needs data)
    with pytest.raises(ConfigError, match="needs data; these have none"):
        gbkfit.load(dict(
            version=VERSION, galaxies=[GALAXY], observations=[observation()],
            params=PARAMS, fitter=dict(type='scipy.least_squares')))


def test_the_drivers_given_replace_those_of_the_configuration():
    from gbkfit.driver.drivers.host import DriverHost
    driver = DriverHost()
    loaded = gbkfit.load(
        dict(version=VERSION, galaxies=[GALAXY],
             observations=[observation()], params=PARAMS,
             driver=dict(type='cuda')),
        drivers={None: driver})
    assert loaded.group_plan.observation_drivers() == (driver,)


def test_configs_resolve_relative_paths_against_their_root(tmp_path):
    # (their directory, or their section 'root', relative to it, or the
    # folder of their project)
    config = tmp_path / 'configs' / 'fit.yaml'
    config.parent.mkdir()
    config.write_text("version: 3\ngalaxies: {}\nroot: ../data\n")
    file = setup.read_config(str(config), ('galaxies',))
    assert file.config == dict(galaxies=[{}])
    assert file.root == str(tmp_path / 'data')
    config.write_text("version: 3\ngalaxies: {}\n")
    assert setup.read_config(str(config), ('galaxies',)).root == \
        str(config.parent)
    with pytest.raises(ConfigError, match="needs the sections"):
        setup.read_config(str(config), ('galaxies', 'fitter'))
    gbkfit.project.create(str(tmp_path))
    assert setup.read_config(str(config), ('galaxies',)).root == \
        str(tmp_path)


def load_plan(galaxy=GALAXY, **sections):
    """
    The plan of the ObservationGroup of a galaxy (GALAXY by default) and
    the sections of a configuration, on its drivers.
    """
    cfg = dict(galaxies=[galaxy]) | sections
    group = setup._load_observation_group(cfg)
    return group.plan(*setup._load_placement(
        cfg, group, setup._load_drivers(cfg)))


def test_observations_use_the_drivers_of_the_configuration():
    # One driver: every observation has it
    plan = load_plan(
        driver=HOST,
        observations=[observation(name='a'), observation(name='b')])
    a, b = plan.observation_drivers()
    assert a is b
    # Several: each observation names its own
    plan = load_plan(
        drivers=dict(cpu1=HOST, cpu2=HOST),
        observations=[
            observation(name='a', driver='cpu2'),
            observation(name='b', driver='cpu1'),
            observation(name='c', driver='cpu2')])
    a, b, c = plan.observation_drivers()
    assert a is c and a is not b
    # None: the host
    plan = load_plan(observations=[observation()])
    assert plan.observation_drivers()[0].type() == 'host'


# A galaxy with mass, seen as a spectral cube
MASSIVE = dict(
    components=[dict(
        type='disk', loose=False, tilted=False, rnmin=0, rnmax=4,
        rnsep=1, density=dict(ptraits=dict(type='exponential')),
        kinematics=dict(
            type='traits',
            vptraits=dict(type='mass'), dptraits=dict(type='uniform')),
        emission=dict(type='lines'))],
    mass=[dict(type='spheroid', name='halo', profile=dict(
        type='pseudo_isothermal', parameters='density_radius'))],
    distance=dict(type='direct'))

CUBE = dict(
    type='sky',
    line_of_sight=dict(type='flat'),
    observable=dict(
        type='pixel_spectra', size=[8, 8],
        spectral=dict(size=5, convention='relativistic')))


def test_galaxies_with_mass_use_the_drivers_of_the_configuration():
    # (for their stage, as observations do for their models)
    def drivers(galaxy, **sections):
        plan = load_plan(galaxy, **sections)
        return plan.galaxy_drivers()[0], plan.observation_drivers()[0]
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


def test_galaxies_without_mass_follow_their_observations():
    # Without the option, a galaxy has the driver of the observations that
    # see it, if they have one, else the host (its stage is for them)
    several = dict(drivers=dict(cpu1=HOST, cpu2=HOST))
    plan = load_plan(**several, observations=[observation(driver='cpu1')])
    assert plan.galaxy_drivers() == plan.observation_drivers()
    plan = load_plan(**several, observations=[
        observation(name='a', driver='cpu1'),
        observation(name='b', driver='cpu2')])
    galaxy, = plan.galaxy_drivers()
    assert galaxy.type() == 'host'
    assert galaxy not in plan.observation_drivers()


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
        load_plan(**sections)


def test_parallel_evaluations_are_on_drivers_of_the_configuration(tmp_path):
    config = dict(
        version=VERSION, root=str(tmp_path), galaxies=[GALAXY],
        drivers=dict(cpu1=HOST, cpu2=HOST),
        observations=[image(tmp_path / 'image.fits') | dict(driver='cpu1')],
        params=PARAMS)
    with gbkfit.load(
            config | dict(parallel=dict(drivers=['cpu1', 'cpu2']))) as loaded:
        first, second = loaded.pool.copies()
        assert first is loaded.objective
        assert second.group_plan().observation_drivers() != \
            first.group_plan().observation_drivers()
    assert gbkfit.load(config).pool is None
    with pytest.raises(ConfigError, match="unknown driver 'gpu'"):
        gbkfit.load(config | dict(parallel=dict(drivers=['gpu'])))


def test_depth_order_has_galaxies_and_groups_of_them():
    from gbkfit.observation import DepthGroup
    order = setup._load_depth_order(['fg', {'pair': ['a', 'b']}])
    assert order == ['fg', DepthGroup('pair', ('a', 'b'))]
    with pytest.raises(ConfigError, match="a group of depth_order is"):
        setup._load_depth_order([{'pair': ['a', 'b'], 'other': ['c']}])
