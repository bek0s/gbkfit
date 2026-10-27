"""
Helpers that build observation groups from the cases of the tests.
"""


def split_case(case):
    """
    A galaxy, an observation of it and the configuration of its driver,
    from a case of the tests: a dict with a driver, a galaxy, a line of
    sight, an observable (with the primary beam, psf and lsf of the
    instrument, and the oversampling and dtype of the observation), an optional
    name, which names the galaxy, and the optional lines or band that the
    observation sees. The components of the galaxy have their own
    geometries (geometry 'own', unless they give one), whose parameters
    the cases name after them.
    """
    case = dict(case)
    driver = case.pop('driver')
    observable = dict(case.pop('observable'))
    observation = dict(
        type='sky',
        observable=observable,
        line_of_sight=case.pop('line_of_sight'),
        instrument={k: observable.pop(k)
                    for k in ('primary_beam', 'psf', 'lsf')
                    if k in observable})
    for key in ('oversampling', 'dtype'):
        if key in observable:
            observation[key] = observable.pop(key)
    for key in ('lines', 'band'):
        if key in case:
            observation[key] = case.pop(key)
    galaxy = dict(case.pop('galaxy'))
    components = galaxy.get('components', [])
    galaxy['components'] = [
        dict(geometry='own') | component for component in (
            components if isinstance(components, list) else [components])]
    name = case.pop('name', None)
    if name is not None:
        galaxy['name'] = name
        observation['galaxies'] = [name]
    if case:
        raise ValueError(f"unknown case keys: {list(case)}")
    return galaxy, observation, driver


def observation_group(cases):
    """An ObservationGroup of cases of the tests (see split_case)."""
    from gbkfit.model import galaxy_parser
    from gbkfit.observation import ObservationGroup, observation_parser
    galaxies, observations, _ = zip(*[split_case(case) for case in cases])
    return ObservationGroup(
        galaxy_parser.load(list(galaxies)),
        observation_parser.load(list(observations)))


def group_plan(cases):
    """
    A plan of the ObservationGroup of cases of the tests (see split_case
    and observation_group): each observation on the driver of its case,
    the galaxies on the host.
    """
    from gbkfit.driver import driver_parser
    from gbkfit.driver.drivers.host import DriverHost
    from gbkfit.observation import GroupPlan
    drivers = [split_case(case)[2] for case in cases]
    return GroupPlan(
        observation_group(cases), [DriverHost()] * len(cases),
        [driver_parser.load(driver) for driver in drivers])


def config_group(config):
    """
    An ObservationGroup of the galaxies and observations of a config,
    loaded as gbkfit.load loads them.
    """
    from gbkfit import setup
    return setup._load_observation_group(config)


def config_group_plan(config):
    """
    A plan of the ObservationGroup of a config (see config_group) on its
    drivers, as gbkfit.load loads it.
    """
    from gbkfit import setup
    group = config_group(config)
    from gbkfit.driver.drivers.host import DriverHost
    return group.plan(*setup._load_placement(
        config, group, setup._load_drivers(config), DriverHost()))
