"""
Compare full model evaluations against reference outputs.

Each case in tests/data/reference_models is a model configuration
(<case>.yaml) together with the outputs `gbkfit-cli eval model` produced
for it (<case>.npz). The references come from the last version of gbkfit
that evaluated models end to end (commit 6f79336); see make_references.py
in that directory for how they were made.

The configurations use the config format of 6f79336. HEAD has started
moving to a new format (a single `models` section instead of `drivers`,
`dmodels` and `gmodels`), so they need converting once that refactor is
finished. The reference outputs stay valid.
"""

import pathlib

import numpy as np
import pytest


REFERENCE_DIR = pathlib.Path(__file__).parents[1] / 'data' / 'reference_models'

CASES = sorted(path.stem for path in REFERENCE_DIR.glob('*.yaml'))


@pytest.mark.xfail(
    reason="model evaluation is broken at HEAD (unfinished refactor)")
@pytest.mark.parametrize('case', CASES)
def test_reference_model(case, evaluate_model):
    outputs = evaluate_model(REFERENCE_DIR / f'{case}.yaml')
    references = np.load(REFERENCE_DIR / f'{case}.npz')
    for name in references.files:
        reference = references[name]
        scale = np.nanmax(np.abs(reference))
        np.testing.assert_allclose(
            outputs[name], reference, rtol=1e-5, atol=1e-5 * scale,
            err_msg=f"{case}: {name}")
