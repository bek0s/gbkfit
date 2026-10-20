"""
Tests for the runs of GBKFIT: their manifest, writer and reader
(gbkfit.outputs).
"""

import json

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

import gbkfit
from gbkfit import outputs
from gbkfit.utils import fitsutils, gridutils


COORDS = gridutils.Coords((2.0, 3.0), (11.5, 9.5), (150.0, 2.0), 30.0)


def test_products_are_found_by_kind_and_ids_and_read_back():
    with outputs.create('run', 'eval', options=dict(mode='model')) as run:
        run.write(
            'observations/ngc1313/model/vlos.fits', 'model',
            gridutils.GridData(np.ones((20, 24)), COORDS, None, 'km/s'),
            observation='ngc1313', item='vlos')
        run.write(
            'observations/ngc1313/model/mmap0.fits', 'model',
            np.zeros((20, 24)), unit='Jy/beam',
            observation='ngc1313', item='mmap0')
        run.write(
            'galaxies/main/components/disk/rotation.ecsv', 'rotation_curve',
            Table(dict(r=[1.0, 2.0] * u.arcsec, v=[50.0, 80.0] * u.km / u.s)),
            galaxy='main', component='disk')
        run.write('params.json', 'params', dict(vsys=1.0, bad=np.nan))
    run = outputs.open('run')
    assert (run.task, run.options, run.status) == \
        ('eval', dict(mode='model'), 'complete')
    assert len(run.products('model', observation='ngc1313')) == 2

    vlos = run.read(run.product('model', item='vlos'))
    assert vlos.header['BUNIT'] == 'km/s'
    assert vlos.header['GBKFKIND'] == 'model'
    assert vlos.header['GBKFVER'] == gbkfit.__version__
    _, coords = fitsutils.read_data(
        run.filename(run.product('model', item='vlos')))
    np.testing.assert_allclose(coords.rval, COORDS.rval)
    assert run.read(run.product('model', item='mmap0')).header['BUNIT'] == \
        'Jy/beam'

    rotation = run.read(run.product('rotation_curve', component='disk'))
    assert rotation['v'].unit == u.km / u.s
    # (strict JSON: NaN is null)
    assert run.read(run.product('params')) == dict(vsys=1.0, bad=None)


def test_a_product_of_the_same_path_is_replaced():
    run = outputs.create('run', 'fit')
    run.write('fit/best.json', 'best', dict(chi2=10.0))
    assert outputs.open('run').status == 'running'
    run.write('fit/best.json', 'best', dict(chi2=5.0))
    reader = outputs.open('run')
    assert reader.read(reader.product('best')) == dict(chi2=5.0)


def test_a_run_left_by_an_error_has_failed():
    with pytest.raises(RuntimeError):
        with outputs.create('run', 'fit'):
            raise RuntimeError
    assert outputs.open('run').status == 'failed'


def test_files_of_other_code_are_recorded():
    run = outputs.create('run', 'fit')
    with open(run.filename('logs/gbkfit.log'), 'w') as f:
        f.write('fitting\n')
    run.record('logs/gbkfit.log', 'log', 'log')
    reader = outputs.open('run')
    product = reader.product('log')
    assert open(reader.filename(product)).read() == 'fitting\n'
    with pytest.raises(ValueError, match='no reader'):
        reader.read(product)


@pytest.mark.parametrize('path', [
    '../out.json', '/tmp/out.json', 'gbkfit.json', 'out.txt'])
def test_invalid_paths_are_refused(path):
    run = outputs.create('run', 'eval')
    with pytest.raises(ValueError):
        run.write(path, 'params', {})


def test_values_must_suit_the_format():
    run = outputs.create('run', 'eval')
    with pytest.raises(TypeError, match='cannot be written'):
        run.write('out.fits', 'params', dict(a=1))
    with pytest.raises(TypeError, match='cannot be written'):
        run.write('out.json', 'model', np.zeros(3))
    with pytest.raises(TypeError, match='carries its own'):
        run.write(
            'out.fits', 'model',
            gridutils.GridData(np.ones((20, 24)), COORDS, None), unit='K')


def test_lookups_of_one_product_need_exactly_one():
    run = outputs.create('run', 'eval')
    run.write('a.json', 'params', {}, observation='a')
    run.write('b.json', 'params', {}, observation='b')
    reader = outputs.open('run')
    with pytest.raises(LookupError, match='has 2 products'):
        reader.product('params')
    with pytest.raises(LookupError, match='has 0 products'):
        reader.product('model')


def test_existing_directories():
    outputs.create('run', 'eval')
    with pytest.raises(FileExistsError):
        outputs.create('run', 'eval')
    assert outputs.create('run', 'eval', 'unique').path.endswith('run_1')
    assert outputs.create('run', 'eval', 'overwrite').path.endswith('run')


def test_only_runs_of_known_versions_are_opened(tmp_path):
    with pytest.raises(ValueError, match='not a run'):
        outputs.open(str(tmp_path))
    outputs.create('run', 'eval')
    manifest = json.load(open('run/gbkfit.json'))
    manifest['version'] = outputs.FORMAT_VERSION + 1
    json.dump(manifest, open('run/gbkfit.json', 'w'))
    with pytest.raises(ValueError, match='upgrade GBKFIT'):
        outputs.open('run')
