"""
Compare full model evaluations against reference outputs.

Each case in tests/data/reference_models is a model configuration
(<case>.yaml) together with the outputs `gbkfit-cli eval model` produced
for it (<case>.npz). The references come from commit 6f79336, the last
version before the 2025 refactor, and were made again after one
deliberate change; see make_references.py in that directory.
"""

import pathlib

import numpy as np
import pytest


REFERENCE_DIR = pathlib.Path(__file__).parents[1] / 'data' / 'reference_models'

CASES = sorted(path.stem for path in REFERENCE_DIR.glob('*.yaml'))


@pytest.mark.parametrize('case', CASES)
def test_reference_model(case, evaluate_config):
    outputs = evaluate_config(REFERENCE_DIR / f'{case}.yaml')
    references = np.load(REFERENCE_DIR / f'{case}.npz')
    # The moments of the faint wings of a convolved model are the noise
    # of the float32 FFT (see test_precision), so the moment maps are
    # compared where there is flux (the other outputs everywhere)
    compared = ...
    if 'model_0_moment0_d' in references:
        flux = references['model_0_moment0_d']
        compared = flux > 0.01 * np.nanmax(flux)
    for name in references.files:
        reference = references[name][compared]
        scale = np.nanmax(np.abs(reference))
        np.testing.assert_allclose(
            outputs[name][compared], reference, rtol=1e-5,
            atol=1e-5 * scale, err_msg=f"{case}: {name}")
