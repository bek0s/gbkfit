"""Helpers that build observation groups from the models of the tests."""


def split_model(model):
    """
    A gmodel and an observation of it, from a model of the tests: a dict
    with a driver, a gmodel, a 'dmodel' (the observable, with the psf and
    lsf of the instrument, and the scale and dtype of the observation),
    and an optional name, which names the gmodel.
    """
    model = dict(model)
    observable = dict(model.pop('dmodel'))
    observation = dict(
        driver=model.pop('driver'),
        observable=observable,
        instrument={k: observable.pop(k) for k in ('psf', 'lsf')
                    if k in observable})
    for key in ('scale', 'dtype'):
        if key in observable:
            observation[key] = observable.pop(key)
    gmodel = dict(model.pop('gmodel'))
    name = model.pop('name', None)
    if name is not None:
        gmodel['name'] = name
        observation['gmodel'] = name
    if model:
        raise ValueError(f"unknown model keys: {list(model)}")
    return gmodel, observation


def observation_group(models, dataset=None):
    """
    An ObservationGroup of models of the tests (see split_model), with the
    grids of the given datasets (if any).
    """
    from gbkfit.model import gmodel_parser
    from gbkfit.observation import ObservationGroup, observation_parser
    gmodels, observations = zip(*[split_model(model) for model in models])
    return ObservationGroup(
        gmodel_parser.load(list(gmodels)),
        observation_parser.load(list(observations), dataset=dataset))


def config_group(config):
    """An ObservationGroup of the gmodels and observations of a config."""
    from gbkfit.model import gmodel_parser
    from gbkfit.observation import ObservationGroup, observation_parser
    return ObservationGroup(
        gmodel_parser.load(config['gmodels']),
        observation_parser.load(config['observations']))
