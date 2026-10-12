"""
Helpers that build observation groups from the cases of the tests.
"""


def split_case(case):
    """
    A model, an observation of it and the configuration of its driver,
    from a case of the tests: a dict with a driver, a model, an
    observable (with the primary beam, psf and lsf of the instrument, and
    the scale and dtype of the observation), and an optional name, which
    names the model.
    """
    case = dict(case)
    driver = case.pop('driver')
    observable = dict(case.pop('observable'))
    observation = dict(
        observable=observable,
        instrument={k: observable.pop(k)
                    for k in ('primary_beam', 'psf', 'lsf')
                    if k in observable})
    for key in ('scale', 'dtype'):
        if key in observable:
            observation[key] = observable.pop(key)
    model = dict(case.pop('model'))
    name = case.pop('name', None)
    if name is not None:
        model['name'] = name
        observation['model'] = name
    if case:
        raise ValueError(f"unknown case keys: {list(case)}")
    return model, observation, driver


def observation_group(cases):
    """
    An ObservationGroup of cases of the tests (see split_case); each
    observation has the driver of its case.
    """
    from gbkfit.driver import driver_parser
    from gbkfit.model import model_parser
    from gbkfit.observation import ObservationGroup, observation_parser
    models, observations, drivers = zip(*[split_case(case) for case in cases])
    return ObservationGroup(
        model_parser.load(list(models)),
        observation_parser.load(list(observations), drivers=[
            {None: driver_parser.load(driver)} for driver in drivers]))


def config_group(config):
    """
    An ObservationGroup of the drivers, models and observations of a
    config, loaded as the tasks load them.
    """
    from gbkfit.tasks import _detail
    return _detail.load_observation_group(config)
