"""
Compare the code with the golden values of the cases in tests/data/golden
(see make_golden.py there): end to end, from a configuration with data to
its model, chi-square and log-likelihood, on each driver. A change that
must not change the results must pass them as they are; one that changes
them on purpose makes them again (make_golden.py values CASE ...).
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


@pytest.mark.parametrize('case', CASES)
def test_golden(case, driver):
    folder = GOLDEN_DIR / case
    name = driver.type()
    with open(folder / f'golden_{name}.json') as f:
        golden = json.load(f)
    actual = make_golden.golden_values(folder, name)
    problems = make_golden.compare(
        actual, golden, make_golden.tolerance(folder, name))
    assert not problems, f"{case} on {name}:\n" + '\n'.join(problems[:20])
