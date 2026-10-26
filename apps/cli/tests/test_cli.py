"""
Tests for gbkfit-cli: it runs the tasks of the library with its options,
rejects invalid options, and reports errors in configurations without a
traceback.
"""

import json
import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits

from gbkfit import outputs


def run_cli(workdir, *args):
    """Run gbkfit-cli with the given arguments in a directory."""
    return subprocess.run(
        [sys.executable, '-m', 'gbkfit_cli', *args],
        cwd=workdir, capture_output=True, text=True)


def write_image(filename):
    """Write an image on the sky."""
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CRVAL1=150.0, CRPIX1=12.5, CDELT1=-1 / 3600,
        CTYPE2='DEC--TAN', CRVAL2=2.0, CRPIX2=10.5, CDELT2=1 / 3600))
    fits.writeto(filename, np.ones((20, 24), np.float32), header)


def test_eval_writes_a_run(tmp_path):
    config = dict(
        driver=dict(type='host'),
        galaxies=[dict(components=[dict(
            type='disk', loose=False, tilted=False, rnodes=list(range(8)),
            density=dict(ptraits=dict(type='exponential')),
            emission=dict(type='band'))])],
        observations=[dict(
            type='sky',
            observable=dict(type='pixel_brightness', size=[16, 12]),
            line_of_sight=dict(type='flat'))],
        params=dict(properties=dict(
            xpos=0, ypos=0, posa=30, incl=45, pt_a=1, pt_s=3)))
    (tmp_path / 'config.json').write_text(json.dumps(config))
    result = run_cli(tmp_path, 'eval', 'model', 'config.json')
    assert result.returncode == 0, result.stderr
    run = outputs.open(tmp_path / 'output')
    assert (run.task, run.status) == ('eval', 'complete')
    assert run.product('model').shape == (12, 16)


def test_prep_writes_to_the_output_directory(tmp_path):
    write_image(tmp_path / 'image.fits')
    result = run_cli(
        tmp_path, 'prep', 'pixel_brightness', '--data-d', 'image.fits',
        '--output-dir', 'prepared')
    assert result.returncode == 0
    assert fits.getdata(tmp_path / 'prepared/prep_image.fits').shape == (
        20, 24)
    assert "output will be stored" in result.stderr


def test_prep_takes_the_moments_of_a_table_of_kinematics(tmp_path):
    # (a series without h3 and h4 is a Gaussian: its V and sigma)
    from astropy.table import Table
    Table(dict(V=[10.0], S=[120.0], H3=[0.0], H4=[0.0])).write(
        tmp_path / 'kinematics.ecsv')
    result = run_cli(
        tmp_path, 'prep', 'gauss_hermite_moments', 'kinematics.ecsv',
        '--columns', 'V', 'S', 'H3', 'H4', '--output-dir', 'prepared')
    assert result.returncode == 0
    table = Table.read(tmp_path / 'prepared/prep_kinematics.ecsv')
    assert (table['mean'][0], table['dispersion'][0]) == (10, 120)


@pytest.mark.parametrize('args, message', [
    # One value for each moment order, also when the orders come last
    (['pixel_moments', '--data-d', 'image.fits', '--nanpad', '1', '0', '1'],
     "invalid length"),
    (['pixel_brightness', '--data-d', 'image.fits', '--ccl-lratio', 'nan'],
     "must be a number"),
    (['region_moments', '--bins', 'image.fits', '--data-d', 'image.fits',
      '--minify', '0'], "unrecognized arguments: --minify")])
def test_invalid_arguments_are_rejected(tmp_path, args, message):
    write_image(tmp_path / 'image.fits')
    result = run_cli(tmp_path, 'prep', *args)
    assert result.returncode != 0 and message in result.stderr


def test_configuration_errors_are_reported_without_a_traceback(tmp_path):
    result = run_cli(
        tmp_path, 'prep', 'pixel_spectra', '--data-d', 'cube.fits',
        '--data-e', 'errors.fits', '--noise-error')
    assert result.returncode == 1
    assert "error: give a file of the errors" in result.stderr
    assert 'Traceback' not in result.stderr
