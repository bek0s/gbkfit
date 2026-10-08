"""
Tests for the eval task (gbkfit-cli eval).
"""

import pathlib
import subprocess
import sys

import numpy as np
import pytest
import ruamel.yaml
from astropy.io import fits


REFERENCE_DIR = pathlib.Path(__file__).parents[1] / 'data' / 'reference_models'

yaml = ruamel.yaml.YAML(typ='safe')


def run_eval(mode, config, workdir):
    """Run `gbkfit-cli eval` and return its FITS outputs by file name."""
    workdir.mkdir()
    yaml.dump(config, workdir / 'config.yaml')
    result = subprocess.run(
        [sys.executable, '-m', 'gbkfit.apps.cli', 'eval', mode, 'config.yaml'],
        cwd=workdir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"gbkfit-cli failed:\n{result.stderr[-3000:]}")
    return {
        path.stem: np.array(fits.getdata(path))
        for path in (workdir / 'output').glob('*.fits')}


def test_objective_residuals(tmp_path):
    # Evaluate a model, and use it plus 1 as the data, with errors of 2.
    # Then the residual (model - data) / error is -0.5 everywhere, and
    # the weighted residual (weight 0.5) is -0.25. The data file has no
    # WCS, so its reference pixel is its centre, as in the model.
    config = yaml.load(REFERENCE_DIR / 'thin_disk_scube.yaml')
    model = run_eval('model', config, tmp_path / 'model')['model_0_scube_d']
    fits.writeto(tmp_path / 'data.fits', model + 1)
    config['datasets'] = [dict(
        type='scube',
        scube=dict(data=str(tmp_path / 'data.fits'), error=2.0),
        step=config['models'][0]['dmodel']['step'])]
    config['objective'] = dict(wu=0.5)
    outputs = run_eval('objective', config, tmp_path / 'objective')
    np.testing.assert_allclose(outputs['residual_scube_d'], -0.5, rtol=1e-5)
    np.testing.assert_allclose(outputs['wresidual_scube_d'], -0.25, rtol=1e-5)
