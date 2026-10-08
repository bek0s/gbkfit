
import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits


@pytest.fixture
def evaluate_model(tmp_path):
    """
    Return a function that evaluates a model configuration file with
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
def evaluate_models():
    """
    Return a function that evaluates in-memory model configurations
    through the Python API. It returns the model data of every model
    (on the host) and the extra outputs (e.g. the velocity field of
    each component), keyed as in `ModelGroup.model_h`.
    """
    import gbkfit.model
    import gbkfit.params

    def evaluate(models, properties):
        model_group = gbkfit.model.ModelGroup(
            gbkfit.model.model_parser.load(models))
        params = gbkfit.params.EvaluationParams(
            model_group.pdescs(), properties)
        extra = {}
        data = model_group.model_h(params.evaluate(), extra)
        return data, extra
    return evaluate
