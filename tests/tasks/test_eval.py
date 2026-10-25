"""
Tests for the eval task (gbkfit.tasks.eval).
"""

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import ruamel.yaml
from astropy.io import fits

import gbkfit
from gbkfit import outputs, project


REFERENCE_DIR = pathlib.Path(__file__).parents[1] / 'data' / 'reference_models'

yaml = ruamel.yaml.YAML(typ='safe')


def run_task(workdir, task, options, *flags):
    """
    Run a task of gbkfit (python -m gbkfit.tasks, with the given flags) in
    a directory, and return the finished process.
    """
    return subprocess.run(
        [sys.executable, '-m', 'gbkfit.tasks', *flags, task,
         json.dumps(options)],
        cwd=workdir, capture_output=True, text=True)


def run_eval(mode, config, workdir, profile_iters=0):
    """
    Run the eval task on a configuration (or on a configuration file) in a
    new directory, and return its run (see gbkfit.outputs).
    """
    workdir.mkdir()
    if isinstance(config, dict):
        yaml.dump(config, workdir / 'config.yaml')
        config = 'config.yaml'
    result = run_task(
        workdir, 'eval', eval_options(mode, config, profile_iters))
    if result.returncode != 0:
        pytest.fail(f"eval failed:\n{result.stderr[-3000:]}")
    return outputs.open(workdir / 'output')


def eval_options(mode, config, profile_iters=0):
    """Return the options of the eval task, into output/."""
    return dict(
        mode=mode, config=str(config), profile_iters=profile_iters,
        output_dir='output', output_dir_mode='unique')


def array(run, kind, **ids):
    """Return the data of the one FITS product of a kind and ids."""
    return np.array(run.read(run.product(kind, **ids)).data)


def mock_data(run, k):
    """
    Return the configuration file of mock k of a run, and the values and
    errors of the data of its observation, by the paths it has.
    """
    filename = pathlib.Path(run.filename(run.product('mock_config', mock=k)))
    config = yaml.load(filename)
    root = filename.parent / config['root']
    data = config['observations'][0]['data']
    return filename, *(
        np.array(fits.getdata(root / data[part])) for part in ('data', 'error'))


def test_objective_residuals(tmp_path):
    # Evaluate a model, and use it plus 1 as the data, with errors of 2.
    # Then the residual (model - data) / error is -0.5 everywhere, and
    # the weighted residual (weight 0.5) is -0.25. The data file has no
    # WCS, so its reference pixel is its centre, as in the model.
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    model = array(run_eval('model', config, tmp_path / 'model'), 'model')
    fits.writeto(tmp_path / 'data.fits', model + 1)
    observation = config['observations'][0]
    observation['data'] = dict(
        data=str(tmp_path / 'data.fits'), error=2.0,
        step=observation['observable'].pop('step'))
    observation['observable'].pop('size')
    observation['likelihood'] = dict(type='gaussian', weights=0.5)
    run = run_eval('objective', config, tmp_path / 'objective')
    np.testing.assert_allclose(array(run, 'residual'), -0.5, rtol=1e-5)
    np.testing.assert_allclose(array(run, 'wresidual'), -0.25, rtol=1e-5)
    np.testing.assert_allclose(
        array(run, 'data', observation='observation0', item='spectra'),
        model + 1)
    # The extra outputs of the model (e.g. the high-resolution cube)
    dcube_hi = run.product(
        'extra', observation='observation0', name='dcube_hi')
    assert dcube_hi.path == 'observations/observation0/extra/dcube_hi.fits'
    assert dcube_hi.title and dcube_hi.description
    bdata = run.product('extra', component='component0', name='bdata')
    assert bdata.path == (
        'observations/observation0/galaxies/galaxy0/components/component0/'
        'bdata.fits')


@pytest.mark.parametrize('mode', ['model', 'objective'])
def test_profiling_writes_the_timings(tmp_path, mode):
    # The timings of the profiled evaluations only: model_eval once for
    # each of the 3, in both modes (objective mode times the chi-square)
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    if mode == 'objective':
        model = array(run_eval('model', config, tmp_path / 'model'), 'model')
        fits.writeto(tmp_path / 'data.fits', model)
        observation = config['observations'][0]
        observation['data'] = dict(
            data=str(tmp_path / 'data.fits'), error=1.0,
            step=observation['observable'].pop('step'))
        observation['observable'].pop('size')
    run = run_eval(mode, config, tmp_path / mode, profile_iters=3)
    timings = run.read(run.product('timings'))
    assert timings['model_eval']['count'] == 3
    # (and the log of the run has the messages of the task)
    log = pathlib.Path(run.filename(run.product('log'))).read_text()
    assert 'running performance test...' in log


def test_mock_data_and_their_configuration(tmp_path):
    # Two mocks of noise of rms 0.01: the model plus the noise, with the
    # rms as their errors, and a configuration of each that fits it: at
    # the parameters of the model, its residuals are of one standard
    # deviation
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    config['mock'] = dict(
        noise=dict(type='gauss', rms=0.01), nmocks=2, seed=3)
    run = run_eval('mock', config, tmp_path / 'mock')
    (filename, data0, error0), (_, data1, _) = (
        mock_data(run, k) for k in ('0', '1'))
    noise = data0 - array(run, 'model')
    assert noise.std() == pytest.approx(0.01, rel=0.01)
    np.testing.assert_array_equal(error0, np.float32(0.01))
    assert not np.array_equal(data0, data1)
    assert len(run.products('mock_data', mock='0')) == 3
    fit = run_eval('objective', filename, tmp_path / 'fit')
    assert array(fit, 'residual').std() == pytest.approx(1, rel=0.01)


def test_rotation_curve_from_radial_nodes(tmp_path):
    # A rotation curve given by an expression of the radial nodes of the
    # disk and of user-defined parameters is the same as the one given by
    # its values
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    component = config['galaxies'][0]['components'][0]
    for option in ('rnmin', 'rnmax', 'rnsep'):
        component.pop(option, None)
    rnodes = list(range(0, 21, 2))
    component['rnodes'] = rnodes
    component['kinematics']['vptraits'] = dict(type='nw_uniform', direction='tan')
    properties = config['params']['properties']
    del properties['vpt_rt']
    config['pdescs'] = dict(vmax=dict(type='scalar'), rt=dict(type='scalar'))
    properties.update(vmax=180, rt=3, vpt_vt='vmax * np.arctan(rnodes / rt)')
    expression = array(
        run_eval('model', config, tmp_path / 'expression'), 'model')
    del config['pdescs'], properties['vmax'], properties['rt']
    properties['vpt_vt'] = (180 * np.arctan(np.array(rnodes) / 3)).tolist()
    values = array(run_eval('model', config, tmp_path / 'values'), 'model')
    np.testing.assert_array_equal(expression, values)


def test_a_configuration_loads_to_evaluate_it_in_python(tmp_path):
    # (at the values of its parameters here, as the task does)
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    setup = gbkfit.load(str(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml'))
    assert setup.objective is None
    values = setup.params.evaluate()
    model = setup.group_plan.model_h(values)[0]['spectra']['d']
    expected = array(run_eval('model', config, tmp_path / 'model'), 'model')
    np.testing.assert_array_equal(model, expected)
    # With data (the model plus 1, with errors of 2), its outputs have the
    # data and the residuals, -0.5, as those the task writes
    fits.writeto(tmp_path / 'data.fits', expected + 1)
    observation = config['observations'][0]
    observation['data'] = dict(
        data=str(tmp_path / 'data.fits'), error=2.0,
        step=observation['observable'].pop('step'))
    observation['observable'].pop('size')
    yaml.dump(config, tmp_path / 'data.yaml')
    setup = gbkfit.load(str(tmp_path / 'data.yaml'))
    outputs = setup.outputs(setup.params.evaluate(), data=True)
    kinds = {output.kind for output in outputs.values()}
    assert {'model', 'data', 'error', 'residual', 'wresidual'} <= kinds
    assert 'extra' not in kinds
    residual = outputs['observations/observation0/residual/spectra.fits']
    np.testing.assert_allclose(residual.value.data, -0.5, rtol=1e-5)


def test_a_setup_of_a_project_runs_in_its_runs(tmp_path):
    # A setup that names the pieces of the reference model, whose data are
    # its model plus 1 in the folder data of the project (relative to it),
    # and changes nothing of its parameters
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    model = array(run_eval('model', config, tmp_path / 'model'), 'model')
    study = tmp_path / 'study'
    project.create(str(study))
    fits.writeto(study / 'data/cube.fits', model + 1)
    observation = config['observations'][0]
    observation['data'] = dict(
        data='data/cube.fits', error=2.0,
        step=observation['observable'].pop('step'))
    observation['observable'].pop('size')
    for path, piece in (
            ('galaxies/disk.yaml', config['galaxies'][0]),
            ('observations/cube.yaml', observation),
            ('params/model.yaml', config['params']),
            ('setups/disk/cube.yaml', dict(
                driver=config['driver'], galaxies=['disk'],
                observations='cube',
                params=['model', dict(properties=dict(incl=60))]))):
        (study / path).parent.mkdir(parents=True, exist_ok=True)
        yaml.dump(piece, study / path)
    options = eval_options('objective', 'study/setups/disk/cube.yaml')
    result = run_task(tmp_path, 'eval', options | dict(output_dir=None))
    assert result.returncode == 0, result.stderr
    run = outputs.open(study / 'runs/disk/cube')
    np.testing.assert_allclose(array(run, 'residual'), -0.5, rtol=1e-5)
    setup = run.read(run.product('setup'))
    assert setup['project'] == '../../..'
    assert setup['file'] == 'setups/disk/cube.yaml'
    assert setup['parameters']['incl'] == 'setups/disk/cube.yaml'
    assert setup['parameters']['vsys'] == 'params/model.yaml'


def test_unknown_options(tmp_path):
    # A misspelt option is an error that suggests the right one, or a
    # warning with --lenient
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    component = config['galaxies'][0]['components'][0]
    component['rnmx'] = component['rnmax']
    yaml.dump(config, tmp_path / 'config.yaml')

    options = eval_options('model', 'config.yaml')
    result = run_task(tmp_path, 'eval', options)
    assert result.returncode != 0
    assert "galaxies[0].components[0]" in result.stderr
    assert "'rnmx' (did you mean 'rnmax'?)" in result.stderr
    result = run_task(tmp_path, 'eval', options, '--lenient')
    assert result.returncode == 0
    assert "'rnmx' (did you mean 'rnmax'?); they are ignored" in \
        result.stderr


def test_outputs_have_the_world_coordinates_of_the_model(tmp_path):
    # The model is written with the coordinates of its observable, so it can
    # be read back with them
    from gbkfit.utils import fitsutils
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    config['observations'][0]['observable'].update(rval=[150, 2, 1500], rota=30)
    run = run_eval('model', config, tmp_path / 'model')
    _, coords = fitsutils.read_data(run.filename(run.product('model')))
    np.testing.assert_allclose(coords.step, [1, 1, 10], rtol=1e-12)
    np.testing.assert_allclose(coords.rpix, [23.5, 23.5, 24.5], rtol=1e-12)
    np.testing.assert_allclose(coords.rval, [150, 2, 1500], rtol=1e-12)
    np.testing.assert_allclose(coords.rota, 30, atol=1e-9)


def test_region_spectra_outputs_and_residuals(tmp_path):
    # Spectra in apertures are written with the regions along x and the
    # velocity along y. As data plus 1, with errors of 2, every residual
    # is -0.5.
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    observation = config['observations'][0]
    scube = observation['observable']
    regions = dict(type='apertures', apertures=[
        dict(type='field'), dict(type='circle', x=0, y=0, radius=3)])
    observation['observable'] = dict(
        type='region_spectra', regions=regions,
        size=scube['size'][:2], step=scube['step'][:2],
        spec_size=scube['size'][2], spec_step=scube['step'][2])
    run = run_eval('model', config, tmp_path / 'model')
    model = array(run, 'model')
    assert model.shape == (50, 2)
    header = run.read(run.product('model')).header
    assert header['CTYPE2'] == 'VRAD'
    assert header['CDELT2'] == 10
    fits.writeto(tmp_path / 'data.fits', model + 1, header)
    observable = observation['observable']
    for key in ('regions', 'spec_size', 'spec_step'):
        observable.pop(key)
    observation['data'] = dict(
        regions=regions, data=str(tmp_path / 'data.fits'), error=2.0)
    run = run_eval('objective', config, tmp_path / 'objective')
    np.testing.assert_allclose(array(run, 'residual'), -0.5, rtol=1e-5)


def test_one_galaxy_seen_by_two_observations(tmp_path):
    # The observations refer to their galaxy, so there can be more
    # observations than galaxies
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    scube = config['observations'][0]
    mmaps = dict(
        type='sky',
        name='maps', line_of_sight=dict(type='flat'),
        observable=dict(type='pixel_moments', size=[48, 48],
                        spec_convention='relativistic'))
    config['observations'] = [scube | dict(name='cube'), mmaps]
    run = run_eval('model', config, tmp_path / 'model')
    assert array(run, 'model', observation='cube').shape == (50, 48, 48)
    assert array(
        run, 'model', observation='maps', item='moment1').shape == (48, 48)


def test_configuration_errors_are_reported_without_a_traceback(tmp_path):
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    config['params']['properties']['incl'] = 'nonsense +'
    yaml.dump(config, tmp_path / 'config.yaml')
    result = run_task(
        tmp_path, 'eval', eval_options('model', 'config.yaml'))
    assert result.returncode == 1
    assert "error: params" in result.stderr
    assert 'Traceback' not in result.stderr
