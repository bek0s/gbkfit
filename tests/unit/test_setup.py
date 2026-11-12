"""Tests for the front door: configurations set up (gbkfit.setup)."""

import numpy as np
import pytest
from astropy.io import fits

import gbkfit
from gbkfit.config import FORMAT_VERSION, ConfigError
import gbkfit.project
from gbkfit import setup
from gbkfit.setup import _config
from gbkfit.objective import Objective


GALAXY = dict(components=[dict(
    type='thin_disk', loose=False, tilted=False, rnodes=dict(max=4, sep=1),
    rptraits=dict(type='exponential'),
    emission=dict(type='band'))])

PARAMS = dict(properties=dict(
    xpos=0, ypos=0, posa=30, incl=50, rpt_a=1,
    rpt_s=dict(value=2, min=1, max=4)))

HOST = dict(type='host')

VERSION = FORMAT_VERSION


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
        data=dict(type='pixel_brightness', data=path.name))


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
        assert loaded.fitter() is None and loaded.pool() is None
        outputs = loaded.outputs(evaluation, data=True)
        assert {'model', 'residual', 'wresidual', 'data'} <= {
            output.kind for output in outputs.values()}


def test_a_setup_in_python_is_that_of_its_configuration(tmp_path):
    # (of the objects of the sections of the configuration: the group of
    # its galaxies and observations, and its parameters)
    from gbkfit.model import galaxy_parser
    from gbkfit.observation import ObservationGroup, observation_parser
    from gbkfit.params import ParamSpace
    obs = image(tmp_path / 'image.fits') | dict(data=dict(
        type='pixel_brightness', data=str(tmp_path / 'image.fits')))
    with gbkfit.load(dict(
            version=VERSION, galaxies=[GALAXY], observations=[obs],
            params=PARAMS)) as loaded:
        expected = loaded.evaluate().objective.merits
    group = ObservationGroup(
        galaxy_parser.load(GALAXY), observation_parser.load(obs))
    params = ParamSpace.from_properties(group.pdescs(), PARAMS['properties'])
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
    with pytest.raises(ValueError, match="they lack"):
        gbkfit.Setup(group, ParamSpace.from_properties(
            dict(a=ParamScalarDesc()), dict(a=1)))
    params = ParamSpace.from_properties(
        group.pdescs(), PARAMS['properties'], function_names=['vcirc'])
    with pytest.raises(ValueError, match=r"no functions \['vcirc'\]"):
        gbkfit.Setup(group, params)


def noisy_image(path, seed=1):
    """
    Write an image of 8 x 8 values of noise of errors 0.1, and return its
    observation.
    """
    rng = np.random.default_rng(seed)
    fits.writeto(path, rng.normal(0, 0.1, (8, 8)))
    return dict(
        type='sky', line_of_sight=dict(type='flat'),
        observable=dict(type='pixel_brightness'),
        data=dict(type='pixel_brightness', data=path.name, error=0.1))


def test_the_likelihoods_are_one_for_all_or_by_observation_name(tmp_path):
    # (of the section likelihoods: Gaussian by default; one for all; or by
    # observation name, the Gaussian for the others)
    observations = [
        noisy_image(tmp_path / 'a.fits', 1) | dict(name='a'),
        noisy_image(tmp_path / 'b.fits', 2) | dict(name='b')]
    config = dict(
        version=VERSION, root=str(tmp_path), galaxies=[GALAXY],
        observations=observations, params=PARAMS)

    def chi2(**sections):
        with gbkfit.load(config | sections) as loaded:
            return loaded.evaluate().objective.data_chi2
    default = chi2()
    assert chi2(likelihoods=dict(type='gauss', weights=2)) == pytest.approx(
        4 * default)
    assert chi2(likelihoods=dict(b=dict(type='gauss', weights=2))) \
        == pytest.approx(default + 3 * chi2(observations=[
            observations[1]]), rel=1e-6)
    # (a fitted noise of one is a parameter, named after it, of the data's
    # unit)
    with gbkfit.load(config | dict(
            likelihoods=dict(a=dict(type='gauss', fitted_noise=dict(
                brightness=dict(scale=True, scale_min=0.1)))),
            params=dict(properties=PARAMS['properties'] | dict(
                a_noise_scale_brightness=2)))) as loaded:
        assert set(loaded.likelihoods().pdescs()) == {
            'a_noise_scale_brightness'}
        assert loaded.evaluate().values['a_noise_scale_brightness'] == 2


@pytest.mark.parametrize('likelihoods, path, message', [
    (dict(c=dict(type='gauss')), ['likelihoods'],
     r"names unknown observations \['c'\]"),
    (dict(b=dict(type='gauss', weights=dict(moment1=2))),
     ['likelihoods', 'b'], r"does not have: \['moment1'\]"),
    (dict(b=dict(type='gaus')), ['likelihoods', 'b'], "unknown type"),
    (dict(type='gauss', fitted_noise=dict(moment0=dict(jitter=True))),
     ['likelihoods'], r"observation 'a': the options of the likelihood"),
    ('gauss', [], "section 'likelihoods' must be a dictionary")])
def test_the_likelihoods_have_the_paths_of_their_errors(
        tmp_path, likelihoods, path, message):
    observations = [
        noisy_image(tmp_path / 'a.fits', 1) | dict(name='a'),
        noisy_image(tmp_path / 'b.fits', 2) | dict(name='b')]
    with pytest.raises((ConfigError, ValueError), match=message) as error:
        gbkfit.load(dict(
            version=VERSION, root=str(tmp_path), galaxies=[GALAXY],
            observations=observations, params=PARAMS,
            likelihoods=likelihoods))
    assert getattr(error.value, 'config_path', path)[:len(path)] == path


def test_a_fit_fits_the_observations_with_data(tmp_path):
    # The others are evaluated alongside (e.g. a prediction of a band
    # without data), at the values of the fit
    observations = [
        observation(name='prediction'),
        noisy_image(tmp_path / 'image.fits') | dict(name='image')]
    config = dict(
        version=VERSION, root=str(tmp_path), galaxies=[GALAXY],
        observations=observations, params=PARAMS,
        fitter=dict(type='scipy.least_squares'),
        bootstrap=dict(nsamples=3, scheme=dict(image=dict(type='gauss'))))
    with gbkfit.load(config) as loaded, gbkfit.load(
            config | dict(observations=observations[1:])) as alone:
        assert loaded.objective().observations() == (1,)
        assert loaded.evaluate().objective.merits.chi2 == pytest.approx(
            alone.evaluate().objective.merits.chi2, rel=1e-6)
        best = loaded.fit().best()
        assert best.free == pytest.approx(alone.fit().best().free, rel=1e-4)
        outputs = loaded.outputs(
            loaded.evaluate(best.free), data=True)
        kinds = {}
        for output in outputs.values():
            kinds.setdefault(output.ids.get('observation'), set()).add(
                output.kind)
        assert kinds['prediction'] == {'model'}
        assert {'model', 'data', 'residual', 'wresidual'} \
            <= kinds['image']
        # (a bootstrap refits the data alone, as the fit)
        bootstrap = loaded.bootstrap(best.elements)
        assert len(bootstrap.samples) == 3 and not bootstrap.failures


def test_a_setup_without_data_states_the_likelihoods_of_its_fit():
    # (e.g. a setup of mock data: checked against the observations, they
    # compare nothing until the observations have data)
    loaded = gbkfit.load(dict(
        version=VERSION, galaxies=[GALAXY],
        observations=[observation(name='image')], params=PARAMS,
        likelihoods=dict(image=dict(type='gauss', weights=2))))
    assert loaded.objective() is None
    assert loaded.likelihoods().observations() == ()
    with pytest.raises(ValueError, match=r"does not have: \['spectra'\]"):
        gbkfit.load(dict(
            version=VERSION, galaxies=[GALAXY],
            observations=[observation(name='image')], params=PARAMS,
            likelihoods=dict(type='gauss', weights=dict(spectra=2))))


def test_an_observation_of_mocks_takes_data_in_one_line(tmp_path):
    # (its grid, as without data, is checked against the data: one
    # configuration of an observation draws mocks and fits data)
    fits.writeto(tmp_path / 'image.fits', np.zeros((8, 8)))
    data = dict(type='pixel_brightness', data='image.fits')
    config = dict(
        version=VERSION, root=str(tmp_path), galaxies=[GALAXY],
        params=PARAMS)
    with gbkfit.load(config | dict(
            observations=[observation(data=data)])) as loaded:
        assert loaded.objective().observations() == (0,)
    with pytest.raises(ConfigError, match="are not those of the data"):
        gbkfit.load(config | dict(observations=[observation(
            data=data, observable=dict(
                type='pixel_brightness', size=[8, 6]))]))


def test_a_setup_takes_likelihoods_of_its_group():
    from gbkfit.model import galaxy_parser
    from gbkfit.observation import (
        Likelihoods, ObservationGroup, observation_parser)
    from gbkfit.params import ParamSpace

    def group():
        return ObservationGroup(
            galaxy_parser.load(GALAXY),
            observation_parser.load(observation()))
    params = ParamSpace.from_properties(
        group().pdescs(), PARAMS['properties'])
    with pytest.raises(ValueError, match="of another group"):
        gbkfit.Setup(group(), params, Likelihoods(group()))


def test_the_outputs_of_an_evaluation_are_its_own():
    # (the plan reuses its arrays of the models: the outputs of a second
    # evaluation must not overwrite those of the first)
    loaded = gbkfit.load(dict(
        version=VERSION, galaxies=[GALAXY], observations=[observation()],
        params=PARAMS))
    first = loaded.outputs(loaded.evaluate())
    model = next(o for o in first.values() if o.kind == 'model')
    before = np.array(model.value.data)
    loaded.outputs(loaded.evaluate(dict(rpt_s=3.0)))
    np.testing.assert_array_equal(model.value.data, before)


def test_a_configuration_without_data_cannot_fit():
    loaded = gbkfit.load(dict(
        version=VERSION, galaxies=[GALAXY], observations=[observation()],
        params=PARAMS))
    assert loaded.objective() is None
    model = loaded.group_plan().model_h(loaded.evaluate().values)[0]
    assert model['brightness'].data.sum() > 0
    with pytest.raises(ValueError, match="a fit needs a fitter"):
        loaded.fit()
    # (a configuration of a fit, with workers even of MPI outside a job,
    # loads without data, e.g. to draw mocks, but does not fit: audit
    # 2026-10-31 #27, #28)
    loaded = gbkfit.load(dict(
        version=VERSION, galaxies=[GALAXY], observations=[observation()],
        params=PARAMS, fitter=dict(type='scipy.least_squares'),
        workers=dict(type='mpi')))
    assert loaded.objective() is None and loaded.pool() is None
    loaded.evaluate()
    with pytest.raises(ValueError, match="a fit needs data"):
        loaded.fit()


def test_the_drivers_given_replace_those_of_the_configuration():
    from gbkfit.driver.drivers.host import DriverHost
    driver = DriverHost()
    loaded = gbkfit.load(
        dict(version=VERSION, galaxies=[GALAXY],
             observations=[observation()], params=PARAMS,
             driver=dict(type='cuda')),
        drivers={None: driver})
    assert loaded.group_plan().observation_drivers() == (driver,)


def test_configs_resolve_relative_paths_against_their_root(tmp_path):
    # (their directory, or their section 'root', relative to it, or the
    # folder of their project)
    config = tmp_path / 'configs' / 'fit.yaml'
    config.parent.mkdir()
    config.write_text(f"version: {VERSION}\ngalaxies: {{}}\nroot: ../data\n")
    file = setup.read_config(str(config), ('galaxies',))
    assert file.config == dict(galaxies=[{}])
    assert file.root == str(tmp_path / 'data')
    config.write_text(f"version: {VERSION}\ngalaxies: {{}}\n")
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
    group = _config._load_observation_group(cfg)
    from gbkfit.driver.drivers.host import DriverHost
    return group.plan(*_config._load_placement(
        cfg, group, _config._load_drivers(cfg), DriverHost()))


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
        type='thin_disk', loose=False, tilted=False, rnodes=dict(max=4, sep=1),
        rptraits=dict(type='exponential'),
        vptraits=dict(type='mass'), dptraits=dict(type='uniform'),
        emission=dict(type='lines'))],
    mass=[dict(type='halo', name='halo', profile=dict(
        type='pseudo_isothermal', parameterisation='density_radius'))],
    distance=dict(type='parameter'))

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


def test_a_setup_in_python_places_its_galaxies_as_its_configuration():
    # (the rule of the galaxies without mass was the loader's alone: in
    # Python, one went on the host while its observation ran on another
    # driver, a copy at every evaluation; audit 2026-11-11 g#1)
    from gbkfit.driver.drivers.host import DriverHost
    from gbkfit.params import ParamSpace
    cpu1, cpu2 = DriverHost(), DriverHost()
    mass = dict(mass=MASSIVE['mass'], distance=MASSIVE['distance'])
    config = dict(
        version=VERSION, drivers=dict(cpu1=HOST, cpu2=HOST),
        galaxies=[
            # (with mass: its own driver; without: that of the
            # observations that see it, if they share one, else the host)
            GALAXY | mass | dict(name='m', driver='cpu2'),
            GALAXY | dict(name='a'),
            GALAXY | dict(name='b'),
            GALAXY | dict(name='c', driver='cpu2')],
        observations=[
            observation(name='o1', galaxies=['m', 'a', 'b'], driver='cpu1'),
            observation(name='o2', galaxies=['b', 'c'], driver='cpu2')])
    group = _config._load_observation_group(config)
    properties = dict.fromkeys(group.pdescs(), 1.0)
    names = {id(cpu1): 'cpu1', id(cpu2): 'cpu2'}

    def placed(setup):
        plan = setup.group_plan()
        return [
            [names.get(id(driver), driver.type()) for driver in drivers]
            for drivers in (plan.galaxy_drivers(), plan.observation_drivers())]
    with gbkfit.load(
            config | dict(params=dict(properties=properties)),
            drivers=dict(cpu1=cpu1, cpu2=cpu2)) as loaded:
        from_config = placed(loaded)
    with gbkfit.Setup(
            group, ParamSpace.from_properties(group.pdescs(), properties),
            placement=dict(o1=cpu1, o2=cpu2, m=cpu2, c=cpu2)) as built:
        in_python = placed(built)
    assert from_config == in_python == [
        ['cpu2', 'cpu1', 'host', 'cpu2'], ['cpu1', 'cpu2']]


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


def test_workers_are_on_drivers_of_the_configuration(tmp_path):
    config = dict(
        version=VERSION, root=str(tmp_path), galaxies=[GALAXY],
        drivers=dict(cpu1=HOST, cpu2=HOST),
        observations=[image(tmp_path / 'image.fits') | dict(driver='cpu1')],
        params=PARAMS)
    workers = dict(type='threads', drivers=['cpu1', 'cpu2'])
    with gbkfit.load(config | dict(workers=workers)) as loaded:
        values = loaded.params().evaluate()
        assert loaded.pool().size() == 2
        assert loaded.pool().map(Objective.chi2, [values] * 3) == \
            pytest.approx([loaded.objective().chi2(values)] * 3, rel=1e-12)
    assert gbkfit.load(config).pool() is None
    workers = dict(type='threads', drivers=['gpu'])
    with pytest.raises(ConfigError, match="unknown driver 'gpu'"):
        gbkfit.load(config | dict(workers=workers))
    assert gbkfit.describe(config | dict(workers=workers)).problems


def test_depth_order_has_galaxies_and_groups_of_them():
    from gbkfit.observation import DepthGroup
    order = _config._load_depth_order(['fg', {'pair': ['a', 'b']}])
    assert order == ['fg', DepthGroup('pair', ('a', 'b'))]
    with pytest.raises(ConfigError, match="a group of depth_order is"):
        _config._load_depth_order([{'pair': ['a', 'b'], 'other': ['c']}])


def test_a_configuration_is_described_without_devices(tmp_path):
    # Its problems with their paths, its parameters; a driver of cuda is
    # checked by type only (no CuPy), and the rules of what observations
    # see are checked without plans
    config = dict(
        version=VERSION, driver=dict(type='cuda'), galaxies=[GALAXY],
        observations=[observation()], params=PARAMS)
    description = gbkfit.describe(config)
    assert description.problems == ()
    assert set(description.params.pdescs()) >= {'xpos', 'rpt_s'}
    stars = dict(
        type='sky', lines=['stars'], line_of_sight=dict(type='flat'),
        observable=dict(type='pixel_spectra', size=[8, 8], spectral=dict(
            size=11, convention='relativistic')))
    described = gbkfit.describe(config | dict(observations=[stars]))
    (problem,) = described.problems
    assert problem.severity == 'error' and problem.path == ['observations', 0]
    assert described.params is None


def test_the_front_door_lists_its_names():
    # (ConfigError was only at gbkfit.utils.parseutils, and dir(gbkfit)
    # did not list load: audit 2026-10-31 #60)
    assert gbkfit.ConfigError is ConfigError
    assert {'load', 'describe', 'ConfigError'} <= set(dir(gbkfit))


def test_describe_checks_the_rules_of_a_fit():
    # (an optimizer without a start passed describe, and failed at load:
    # audit 2026-10-31 #26)
    params = dict(properties=PARAMS['properties'] | dict(
        rpt_s=dict(min=1, max=4)))
    description = gbkfit.describe(dict(
        version=VERSION, galaxies=[GALAXY], observations=[observation()],
        params=params, fitter=dict(type='scipy.least_squares')))
    errors = [p for p in description.problems if p.severity == 'error']
    assert len(errors) == 1 and "needs option 'value'" in errors[0].message


@pytest.mark.parametrize('properties, path, message', [
    # (of the fitter: at the root in describe, and with no path at load,
    # which checked it in Setup; audit 2026-11-11 g#5)
    (dict(rpt_a=dict(solve=dict(
        type='gauss', mean=1, sigma_aicp=[0.1, 10], likelihood='profiled'))),
     ['fitter'], "need a fit that minimises the aicp"),
    # (of the parameters)
    (dict(rpt_s=dict(min=1, max=4)),
     ['params', 'properties', 'rpt_s'], "needs option 'value'"),
])
def test_the_rules_of_a_fit_are_at_the_options_they_are_about(
        properties, path, message):
    # The same in describe and at load: of options that are invalid
    # together, ValueErrors (see ConfigError)
    from gbkfit.config import error_location
    config = dict(
        version=VERSION, galaxies=[GALAXY], observations=[observation()],
        params=dict(properties=PARAMS['properties'] | properties),
        fitter=dict(type='scipy.least_squares'))
    (problem,) = gbkfit.describe(config).problems
    assert problem.path == path and message in problem.message
    with pytest.raises(ValueError, match=message) as error:
        gbkfit.load(config)
    assert error_location(error.value)[0] == path


def test_a_setup_that_fails_closes_what_it_made(tmp_path, monkeypatch):
    # (its plan and its objective, which hold the memory of devices: close
    # never runs for a setup that was not made)
    from gbkfit.observation import GroupPlan
    from gbkfit.parallel import WorkersThreads
    closed = []
    for cls in (GroupPlan, Objective):
        def close(self, close=cls.close):
            closed.append(type(self))
            close(self)
        monkeypatch.setattr(cls, 'close', close)

    def fail(*args):
        raise RuntimeError("no device")
    config = dict(
        version=VERSION, root=str(tmp_path), galaxies=[GALAXY],
        observations=[image(tmp_path / 'image.fits')], params=PARAMS)
    monkeypatch.setattr(WorkersThreads, 'start', fail)
    with pytest.raises(RuntimeError, match="no device"):
        gbkfit.load(config | dict(workers=dict(
            type='threads', drivers=[HOST])))
    assert set(closed) == {Objective, GroupPlan}
    closed.clear()
    monkeypatch.setattr(gbkfit.setup._setup, 'Objective', fail)
    with pytest.raises(RuntimeError, match="no device"):
        gbkfit.load(config)
    assert set(closed) == {GroupPlan}


def test_describe_checks_the_names_of_the_drivers():
    # (load refused a name that describe let through)
    (problem,) = gbkfit.describe(dict(
        version=VERSION, galaxies=[GALAXY],
        observations=[observation(driver='gpu 0')], params=PARAMS,
        drivers={'gpu 0': HOST})).problems
    assert problem.path == ['drivers', 'gpu 0']
    assert "invalid name 'gpu 0'" in problem.message


def test_configurations_in_python_take_quantities_and_numpy_values():
    # (a round trip through JSON refused them, though the options take
    # them)
    import astropy.units as u
    size = np.array([8, 8])
    config = dict(
        version=VERSION, galaxies=[GALAXY], params=PARAMS,
        lines=dict(line=6563 * u.AA), observations=[observation(
            observable=dict(type='pixel_brightness', size=size),
            oversampling=[np.int64(2), np.int64(2)])])
    plain = _config._config_file(config).config
    assert plain['lines']['line'] == 6563 * u.AA
    assert plain['observations'][0]['observable']['size'] == [8, 8]
    with gbkfit.load(config) as loaded:
        model = loaded.group_plan().model_h(loaded.evaluate().values)[0]
        assert model['brightness'].data.shape == (8, 8)


def test_the_units_of_a_configuration_are_read_as_fits_reads_them(tmp_path):
    # (several '/', which the FITS standard allows; names are not
    # corrected)
    import astropy.units as u
    data = image(tmp_path / 'image.fits')
    config = dict(
        version=VERSION, root=str(tmp_path), galaxies=[GALAXY],
        observations=[data], params=PARAMS)
    data['data']['unit'] = 'erg/s/cm2/Angstrom'
    with gbkfit.load(config) as loaded:
        dataset = loaded.group().observations()[0].data()
        assert next(iter(dataset.values())).unit() == (
            u.erg / u.s / u.cm ** 2 / u.Angstrom)
    data['data']['unit'] = 'ERG/S'
    with pytest.raises(ValueError, match="not a valid unit"):
        gbkfit.load(config)
