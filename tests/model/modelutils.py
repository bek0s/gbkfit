"""
Helpers that build observation groups from the cases of the tests.
"""


def split_case(case):
    """
    A galaxy, an observation of it and the configuration of its driver,
    from a case of the tests: a dict with a driver, a galaxy, a line of
    sight, an observable (with the primary beam, psf and lsf of the
    instrument, and the scale and dtype of the observation), and an
    optional name, which names the galaxy.
    """
    case = dict(case)
    driver = case.pop('driver')
    observable = dict(case.pop('observable'))
    observation = dict(
        observable=observable,
        line_of_sight=case.pop('line_of_sight'),
        instrument={k: observable.pop(k)
                    for k in ('primary_beam', 'psf', 'lsf')
                    if k in observable})
    for key in ('scale', 'dtype'):
        if key in observable:
            observation[key] = observable.pop(key)
    galaxy = dict(case.pop('galaxy'))
    name = case.pop('name', None)
    if name is not None:
        galaxy['name'] = name
        observation['galaxies'] = [name]
    if case:
        raise ValueError(f"unknown case keys: {list(case)}")
    return galaxy, observation, driver


def observation_group(cases):
    """
    An ObservationGroup of cases of the tests (see split_case); each
    observation has the driver of its case.
    """
    from gbkfit.driver import driver_parser
    from gbkfit.model import galaxy_parser
    from gbkfit.observation import ObservationGroup, observation_parser
    galaxies, observations, drivers = zip(
        *[split_case(case) for case in cases])
    return ObservationGroup(
        galaxy_parser.load(list(galaxies)),
        observation_parser.load(list(observations), drivers=[
            {None: driver_parser.load(driver)} for driver in drivers]))


def config_group(config):
    """
    An ObservationGroup of the drivers, galaxies and observations of a
    config, loaded as the tasks load them.
    """
    from gbkfit.tasks import _detail
    return _detail.load_observation_group(config)
