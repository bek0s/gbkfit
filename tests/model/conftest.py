
import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits

from modelutils import observation_group


@pytest.fixture
def evaluate_config(tmp_path):
    """
    Return a function that evaluates a configuration file with
    `gbkfit-cli eval model` and returns its FITS outputs, keyed by file
    name. Each configuration is evaluated in its own directory.
    """
    def evaluate(config):
        workdir = tmp_path / config.stem
        workdir.mkdir()
        result = subprocess.run(
            [sys.executable, '-m', 'gbkfit.apps.cli',
             'eval', 'model', str(config)],
            cwd=workdir, capture_output=True, text=True)
        if result.returncode != 0:
            pytest.fail(f"gbkfit-cli failed:\n{result.stderr[-3000:]}")
        return {
            path.stem: np.array(fits.getdata(path))
            for path in sorted((workdir / 'output').glob('*.fits'))}
    return evaluate


@pytest.fixture
def evaluate_cases():
    """
    Return a function that evaluates cases of the tests (see
    split_case) through the Python API. It returns the model data of
    every observation (on the host) and the extra outputs (e.g. the
    velocity field of each component), keyed as in
    `ObservationGroup.model_h`.
    """
    import gbkfit.params

    def evaluate(cases, properties, modes=None):
        group = observation_group(cases)
        params = gbkfit.params.EvaluationParams(
            group.pdescs(), properties, constants=group.constants(),
            modes=modes)
        extra = {}
        data = group.model_h(params.evaluate(), extra)
        return data, extra
    return evaluate
