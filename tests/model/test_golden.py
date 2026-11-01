"""
Compare the code with the golden values of the cases in tests/data/golden
(see make_golden.py there): end to end, from a configuration with data to
its model, chi-square and log-likelihood, on each driver, with the
reference of the case (those of the host, but of the cases that keep
their own). A change that must not change the results must pass them as
they are; one that changes them on purpose makes them again
(make_golden.py values CASE ...).
"""

import importlib.util
import json
import pathlib

import pytest


GOLDEN_DIR = pathlib.Path(__file__).parents[1] / 'data' / 'golden'

CASES = sorted(
    path.name for path in GOLDEN_DIR.iterdir()
    if (path / 'config.yaml').exists())


def _load_make_golden():
    """Import make_golden.py, which is not in a package."""
    spec = importlib.util.spec_from_file_location(
        'make_golden', GOLDEN_DIR / 'make_golden.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_golden = _load_make_golden()

# The floating types in which the drivers are compared (see
# test_drivers_and_dtypes_agree)
DTYPES = ('float32', 'float64')

# The relative tolerance of float32 against float64: the rounding of the
# float32 sums of many terms (e.g. along the lines of sight of a thick
# disk)
DTYPE_TOLERANCE = 1e-4


@pytest.mark.parametrize('case', CASES)
def test_golden(case, driver):
    folder = GOLDEN_DIR / case
    name = driver.type()
    with open(make_golden.reference_file(folder, name)) as f:
        golden = json.load(f)
    actual = make_golden.golden_values(folder, name)
    problems = make_golden.compare(
        actual, golden, make_golden.tolerance(folder, name))
    assert not problems, f"{case} on {name}:\n" + '\n'.join(problems[:20])


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.parametrize('case', CASES)
def test_drivers_and_dtypes_agree(case):
    # Each case on the host and on CUDA, in float32 and in float64: the
    # drivers agree at the tolerance of CUDA, and float32 with float64
    # (but the cases that differ by design, see Case.per_driver and
    # Case.per_dtype of make_golden.py)
    cupy = pytest.importorskip('cupy')
    if not cupy.cuda.is_available():
        pytest.skip("no CUDA GPU")
    recipe = make_golden.recipes()[case]
    folder = GOLDEN_DIR / case
    values = {
        (driver, dtype): make_golden.golden_values(folder, driver, dtype)
        for driver in make_golden.DRIVERS for dtype in DTYPES}
    problems = []
    if not recipe.per_driver:
        for dtype in DTYPES:
            problems += make_golden.compare(
                values['cuda', dtype], values['host', dtype],
                make_golden.TOLERANCES[dtype, 'cuda'], f'cuda.{dtype}')
    if not recipe.per_dtype:
        for driver in make_golden.DRIVERS:
            problems += make_golden.compare(
                values[driver, 'float32'], values[driver, 'float64'],
                DTYPE_TOLERANCE, f'{driver}.float32')
    assert not problems, f"{case}:\n" + '\n'.join(problems[:20])
