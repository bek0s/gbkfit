"""
The typical values of the parameters (see ParamDesc): every parameter of
the golden configurations has one (but those that users declare for
their own functions), and their models evaluate at them, before any
value is chosen (e.g. a new combination of galaxies and observations in
a UI).
"""

import pathlib

import numpy as np
import pytest

from gbkfit.params import default_values
from gbkfit.setup import describe, load


GOLDEN_DIR = pathlib.Path(__file__).parents[1] / 'data' / 'golden'

# The parameters of the functions of users (their likelihoods), by case
USER_PARAMS = dict(user_likelihoods=['cube_s', 'maps_nu', 'jittered_jitter'])


@pytest.mark.parametrize(
    'path', sorted(GOLDEN_DIR.glob('*/config.yaml')),
    ids=lambda path: path.parent.name)
def test_models_evaluate_at_the_defaults(path):
    pdescs = describe(str(path)).group.pdescs()
    assert [name for name, pdesc in pdescs.items()
            if pdesc.default() is None] \
        == USER_PARAMS.get(path.parent.name, [])
    with load(str(path)) as setup:
        models = setup.group_plan.model_h(default_values(pdescs))
        arrays = [
            array for observation in models for item in observation.values()
            for array in item.values() if array is not None]
        assert arrays and all(np.isfinite(array).any() for array in arrays)
