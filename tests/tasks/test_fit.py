"""
Tests for the fit task (gbkfit-cli fit).
"""

import json
import pathlib
import subprocess
import sys

import pytest
import ruamel.yaml
from astropy.io import fits


REFERENCE_DIR = pathlib.Path(__file__).parents[1] / 'data' / 'reference_models'

yaml = ruamel.yaml.YAML(typ='safe')


def run_cli(workdir, *args):
    """Run gbkfit-cli in a directory; fail the test if it fails."""
    result = subprocess.run(
        [sys.executable, '-m', 'gbkfit.apps.cli', *args],
        cwd=workdir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"gbkfit-cli failed:\n{result.stderr[-3000:]}")


def test_fit_recovers_the_parameters_of_noiseless_data(tmp_path):
    pytest.importorskip('lmfit')
    # The data are the model of the reference config
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    yaml.dump(config, tmp_path / 'model.yaml')
    run_cli(tmp_path, 'eval', 'model', 'model.yaml', '--output-dir', 'model')
    model = fits.getdata(tmp_path / 'model' / 'model_0_spectra_d.fits')
    fits.writeto(tmp_path / 'data.fits', model)
    # Fit three of its parameters, from elsewhere
    observation = config['observations'][0]
    observation['data'] = dict(
        data='data.fits', error=0.01 * float(model.max()),
        step=observation['observable'].pop('step'))
    observation['observable'].pop('size')
    truth = dict(config['params']['properties'])
    config['params']['properties'] |= dict(
        xpos=dict(value=0, min=-3, max=3),
        posa=dict(value=40, min=0, max=90),
        vpt_vt=dict(value=150, min=50, max=300))
    config['fitter'] = dict(type='lmfit.least_squares')
    yaml.dump(config, tmp_path / 'fit.yaml')
    run_cli(tmp_path, 'fit', 'fit.yaml', '--output-dir', 'fit')
    result = json.loads((tmp_path / 'fit' / 'result.json').read_text())
    best = result['solutions'][0]['free']
    for name in ('xpos', 'posa', 'vpt_vt'):
        assert best[name] == pytest.approx(truth[name], abs=1e-3)
    # The arrays of the result, and the model and residuals at the best fit
    for name in ('result.npz', 'model_0_spectra_d.fits',
                 'residual_spectra_d.fits'):
        assert (tmp_path / 'fit' / name).exists()
