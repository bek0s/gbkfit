"""
Tests of copies of setups in other processes (pickled, e.g. for the
workers of an executor): the objectives and parameters of every golden
configuration evaluate there as they do here.
"""

import concurrent.futures
import multiprocessing
import pathlib

import pytest

from gbkfit.objective import Merits, Objective
from gbkfit.params import ParamSpace
from gbkfit.params._errors import ParamEvaluationError
from gbkfit.setup import load


GOLDEN_DIR = pathlib.Path(__file__).parents[1] / 'data' / 'golden'


@pytest.fixture(scope='module')
def executor():
    # (spawn: a fresh process, which has only what the copies bring)
    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(
            1, mp_context=context) as executor:
        yield executor


def merits(
        objective: Objective,
        params: ParamSpace
) -> tuple[Merits, dict[str, float]]:
    """Return the merits and the derived values at the start values."""
    values = params.evaluate()
    with objective:
        merits_ = objective.evaluate(values).merits
        try:
            derived = params.derived_elements(
                values, objective.group_plan().functions())
        except ParamEvaluationError:
            derived = {}
    return merits_, derived


@pytest.mark.parametrize(
    'path', sorted(GOLDEN_DIR.glob('*/config.yaml')),
    ids=lambda path: path.parent.name)
def test_copies_in_other_processes_evaluate_as_the_originals(executor, path):
    with load(str(path)) as setup:
        if setup.objective is None:
            pytest.skip("no data")
        here = merits(setup.objective, setup.params)
        there = executor.submit(
            merits, setup.objective, setup.params).result()
    # (to the rounding of sums in parallel, whose order varies)
    assert there[0] == pytest.approx(here[0], rel=1e-12)
    assert there[1] == pytest.approx(here[1], rel=1e-12)
