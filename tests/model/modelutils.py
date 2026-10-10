"""
Helpers that build observation groups from the cases of the tests, and
components for the tests.
"""

import numpy as np

from gbkfit.model.components.base import Component, ComponentPlan


def split_case(case):
    """
    A model and an observation of it, from a case of the tests: a dict
    with a driver, a model, an observable (with the primary beam, psf and
    lsf of the instrument, and the scale and dtype of the observation),
    and an optional name, which names the model.
    """
    case = dict(case)
    observable = dict(case.pop('observable'))
    observation = dict(
        driver=case.pop('driver'),
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
    return model, observation


def observation_group(cases):
    """An ObservationGroup of cases of the tests (see split_case)."""
    from gbkfit.model import model_parser
    from gbkfit.observation import ObservationGroup, observation_parser
    models, observations = zip(*[split_case(case) for case in cases])
    return ObservationGroup(
        model_parser.load(list(models)),
        observation_parser.load(list(observations)))


def config_group(config):
    """An ObservationGroup of the models and observations of a config."""
    from gbkfit.model import model_parser
    from gbkfit.observation import ObservationGroup, observation_parser
    return ObservationGroup(
        model_parser.load(config['models']),
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
