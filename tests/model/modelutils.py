"""
Helpers that build observation groups from the models of the tests, and
components for the tests.
"""

import numpy as np

from gbkfit.model.gmodels.core import Component, ComponentPlan


def split_model(model):
    """
    A gmodel and an observation of it, from a model of the tests: a dict
    with a driver, a gmodel, a 'dmodel' (the observable, with the primary
    beam, psf and lsf of the instrument, and the scale and dtype of the observation),
    and an optional name, which names the gmodel.
    """
    model = dict(model)
    observable = dict(model.pop('dmodel'))
    observation = dict(
        driver=model.pop('driver'),
        observable=observable,
        instrument={k: observable.pop(k)
                    for k in ('primary_beam', 'psf', 'lsf')
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


def observation_group(models):
    """An ObservationGroup of models of the tests (see split_model)."""
    from gbkfit.model import gmodel_parser
    from gbkfit.observation import ObservationGroup, observation_parser
    gmodels, observations = zip(*[split_model(model) for model in models])
    return ObservationGroup(
        gmodel_parser.load(list(gmodels)),
        observation_parser.load(list(observations)))


def config_group(config):
    """An ObservationGroup of the gmodels and observations of a config."""
    from gbkfit.model import gmodel_parser
    from gbkfit.observation import ObservationGroup, observation_parser
    return ObservationGroup(
        gmodel_parser.load(config['gmodels']),
        observation_parser.load(config['observations']))


class WeightComponent(Component):
    """
    A component that sets the spatial weights to 2, except those of the
    first row of pixels, which it sets to 0.
    """

    @staticmethod
    def type():
        return 'weights'

    @classmethod
    def load(cls, info):
        return cls()

    def dump(self):
        return {}

    def __init__(self):
        super().__init__(name=None)

    def pdescs(self):
        return {}

    def has_weights(self):
        return True

    def plan(self, driver, spectral, dtype, lines):
        return WeightComponentPlan(driver, dtype)


class WeightComponentPlan(ComponentPlan):

    def __init__(self, driver, dtype):
        self._driver = driver
        self._dtype = dtype

    def evaluate(self, params, grid, outputs, out_extra):
        wdata = np.full(outputs['wdata'].shape, 2, self._dtype)
        wdata[..., 0, :] = 0
        self._driver.mem_copy_h2d(wdata, outputs['wdata'])
