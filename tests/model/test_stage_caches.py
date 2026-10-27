"""
The caches of the stages of evaluations (see GroupPlan.stage_caches):
in every golden case (see tests/data/golden), a step in any parameter
evaluates again exactly the stages that read it, and those that take
them, and gives the models of a plan evaluated anew.
"""

import copy
import pathlib

import numpy as np
import pytest

import gbkfit
from gbkfit.observation import GroupPlan
from gbkfit.params import ParamValues
from gbkfit.setup import Setup, read_config


GOLDEN_DIR = pathlib.Path(__file__).parents[1] / 'data' / 'golden'

CASES = sorted(
    path.name for path in GOLDEN_DIR.iterdir()
    if (path / 'config.yaml').exists())


def _setup(case: str) -> Setup:
    """Set up a golden case on the host."""
    file = read_config(str(GOLDEN_DIR / case / 'config.yaml'))
    return gbkfit.load(file._replace(
        config=file.config | dict(driver=dict(type='host'))))


def _stepped(value: float | np.ndarray) -> float | np.ndarray:
    """Return a value of a parameter moved a little (each element)."""
    return value + 1e-3 * np.maximum(np.abs(value), 1)


def _declared(plan: GroupPlan, name: str) -> set[str]:
    """
    Return the labels of the stages of a plan (see GroupPlan.stage_caches)
    that a step in the parameter of the given name evaluates again: those
    that read it, and those that take one that does.
    """
    group = plan.group()
    labels = [group.galaxy_label(g) for g in range(len(group.galaxies()))]
    caches = plan.stage_caches()
    # (the names of the parameters that each reads, in the group: those
    # of the parts of a galaxy are its own)
    reads = {}
    for label, cache in caches.items():
        kind, _, rest = label.partition('.')
        mapping = {} if kind != 'galaxy' \
            else group.mappings()[labels.index(rest.partition('.')[0])]
        reads[id(cache)] = {mapping.get(read, read) for read in cache.reads()}

    def again(cache: object) -> bool:
        return name in reads[id(cache)] or any(map(again, cache.takes()))
    return {label for label, cache in caches.items() if again(cache)}


def _same_models(actual: list, expected: list) -> None:
    """
    Check that the model data of two evaluations are the same, to 1000
    times the precision of their dtype (relative to the largest value of
    each array): some kernels add in an order that changes from run to
    run (e.g. those of clouds), even on the host.
    """
    for model, other in zip(actual, expected, strict=True):
        for key, item in model.items():
            for part, array in item.items():
                if array is None:
                    continue
                scale = np.nanmax(np.abs(other[key][part]), initial=0)
                np.testing.assert_allclose(
                    array, other[key][part], rtol=0,
                    atol=1e3 * np.finfo(array.dtype).eps * scale,
                    err_msg=f"{key} {part}")


def _evaluate(plan: GroupPlan, values: ParamValues) -> list:
    """Return copies of the model data of a plan on the host."""
    return copy.deepcopy(plan.model_h(values))


@pytest.mark.parametrize('case', CASES)
def test_a_step_evaluates_again_the_stages_that_read_it(case):
    setup = _setup(case)
    plan = setup.group_plan
    group = plan.group()
    values = setup.params.evaluate()
    caches = plan.stage_caches()
    with setup:
        plan.model_h(values)
        for name in group.pdescs():
            stepped = values | {name: _stepped(values[name])}
            versions = {
                label: cache.version() for label, cache in caches.items()}
            models = _evaluate(plan, stepped)
            again = {
                label for label, cache in caches.items()
                if cache.version() != versions[label]}
            assert again == _declared(plan, name), name
            fresh = GroupPlan(
                group, plan.galaxy_drivers(), plan.observation_drivers())
            _same_models(models, _evaluate(fresh, stepped))
            plan.model_h(values)
