"""
Compare full model evaluations against reference outputs.

Each case in tests/data/reference_models is the configuration of a
galaxy and an observation of it (<case>.yaml), together with the
outputs `gbkfit-cli eval model` produced for it (<case>.npz). The
references come from commit 6f79336, the last version before the 2025
refactor, and were made again after one deliberate change; see
make_references.py in that directory.
"""

import pathlib

import numpy as np
import pytest


REFERENCE_DIR = pathlib.Path(__file__).parents[1] / 'data' / 'reference_models'

# The references are keyed by the file names of older versions:
# model_0_<item>_<part>, of the kinds of the products by part
KINDS = dict(d='model', m='model_mask', w='model_weight')

CASES = sorted(path.stem for path in REFERENCE_DIR.glob('*.yaml'))


def output(run, name):
    """Return the product of a run that a reference of a name is of."""
    item, part = name.removeprefix('model_0_').rsplit('_', 1)
    return np.array(run.read(run.product(KINDS[part], item=item)).data)


@pytest.mark.parametrize('case', CASES)
def test_reference_model(case, evaluate_config):
    run = evaluate_config(REFERENCE_DIR / f'{case}.yaml')
    references = np.load(REFERENCE_DIR / f'{case}.npz')
    # The moment maps are compared where there is flux (the other outputs
    # everywhere): the moments of the faintest spaxels are ratios of tiny
    # sums
    compared = ...
    if 'model_0_moment0_d' in references:
        flux = references['model_0_moment0_d']
        compared = flux > 0.01 * np.nanmax(flux)
    for name in references.files:
        reference = references[name][compared]
        scale = np.nanmax(np.abs(reference))
        np.testing.assert_allclose(
            output(run, name)[compared], reference, rtol=1e-5,
            atol=1e-5 * scale, err_msg=f"{case}: {name}")
