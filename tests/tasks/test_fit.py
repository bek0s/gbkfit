"""
Tests for the fit task (gbkfit.tasks.fit).
"""

import json
import pathlib
import subprocess
import sys

import pytest
import ruamel.yaml
from astropy.io import fits

from gbkfit import outputs


REFERENCE_DIR = pathlib.Path(__file__).parents[1] / 'data' / 'reference_models'

yaml = ruamel.yaml.YAML(typ='safe')


def run_task(workdir, task, **options):
    """
    Run a task of gbkfit (python -m gbkfit.tasks) in a directory; fail the
    test if it fails.
    """
    result = subprocess.run(
        [sys.executable, '-m', 'gbkfit.tasks', task, json.dumps(options)],
        cwd=workdir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"task {task} failed:\n{result.stderr[-3000:]}")


def evaluate_model(workdir):
    """Evaluate the model of model.yaml in a directory, into model/."""
    run_task(
        workdir, 'eval', mode='model', config='model.yaml', profile_iters=0,
        output_dir='model', output_dir_mode='unique')


def read(directory, kind, **ids):
    """Return the one product of a kind and ids of the run in a directory."""
    run = outputs.open(directory)
    return run.read(run.product(kind, **ids))


@pytest.mark.slow
def test_fit_recovers_the_parameters_of_noiseless_data(tmp_path):
    pytest.importorskip('lmfit')
    # The data are the model of the reference config
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    yaml.dump(config, tmp_path / 'model.yaml')
    evaluate_model(tmp_path)
    model = read(tmp_path / 'model', 'model').data
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
    # (from another directory: the paths of a configuration are relative
    # to its own directory)
    (tmp_path / 'run').mkdir()
    run_task(
        tmp_path / 'run', 'fit', config='../fit.yaml', output_dir='../fit',
        output_dir_mode='unique')
    run = outputs.open(tmp_path / 'fit')
    result = run.read(run.product('fit_result'))
    best = result['solutions'][0]['free']
    for name in ('xpos', 'posa', 'vpt_vt'):
        assert best[name] == pytest.approx(truth[name], abs=1e-3)
    # Its progress, every report a line of JSON, the last final; and the
    # configuration at the best fit so far
    progress = pathlib.Path(run.filename(run.product('progress')))
    assert json.loads(progress.read_text().splitlines()[-1])['final']
    best_config = run.read(run.product('best_config'))
    for name in ('xpos', 'posa', 'vpt_vt'):
        assert best_config['params']['properties'][name]['value'] \
            == pytest.approx(truth[name], abs=1e-3)
    # The arrays of the result, and the model and residuals at the best fit
    for kind in ('covariance', 'laplace'):
        assert run.product(kind, solution='0')
    for kind in ('data', 'model', 'residual', 'wresidual'):
        assert run.product(kind, observation='observation0', item='spectra')
    # A configuration that starts another fit at the best fit
    restart_file = pathlib.Path(run.filename(run.product('restart_config')))
    restart = yaml.load(restart_file)
    for name in ('xpos', 'posa', 'vpt_vt'):
        assert restart['params']['properties'][name]['value'] == \
            pytest.approx(best[name])
    assert restart['fitter'] == config['fitter']
    # which runs from its directory (its root is that of fit.yaml), with
    # another fitter
    restart['fitter'] = dict(type='scipy.least_squares')
    yaml.dump(restart, restart_file)
    run_task(
        restart_file.parent, 'fit', config=restart_file.name,
        output_dir=str(tmp_path / 'refit'), output_dir_mode='unique')
    refit = read(tmp_path / 'refit', 'fit_result')
    for name in ('xpos', 'posa', 'vpt_vt'):
        assert refit['solutions'][0]['free'][name] == pytest.approx(
            best[name], abs=1e-3)


def test_the_best_fit_so_far_starts_at_it(tmp_path):
    # fit/best.yaml starts the free elements at the best fit so far, those
    # of a parameter given as offsets (a mode) too: the elements it starts
    # at are those of the best point of the progress (a few evaluations of
    # the golden case of the features of the parameters)
    import gbkfit
    golden = pathlib.Path(__file__).parents[1] / 'data' / 'golden'
    config = yaml.load(golden / 'params_features' / 'config.yaml')
    config |= dict(
        root=str(golden / 'params_features'),
        fitter=dict(type='scipy.least_squares', max_nfev=3))
    yaml.dump(config, tmp_path / 'fit.yaml')
    run_task(
        tmp_path, 'fit', config='fit.yaml', output_dir='fit',
        output_dir_mode='unique')
    run = outputs.open(tmp_path / 'fit')
    progress = pathlib.Path(run.filename(run.product('progress')))
    best = json.loads(progress.read_text().splitlines()[-1])['best']
    params = gbkfit.load(run.filename(run.product('best_config'))).params
    assert params.element_values(params.evaluate()) == pytest.approx(
        params.element_values(params.evaluate(best)))


@pytest.mark.slow
def test_a_fit_resumes_from_its_output_directory(tmp_path):
    # Its configuration and the log of its evaluations are there: resumed,
    # it replays them all, and ends where it ended
    pytest.importorskip('cma')
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    yaml.dump(config, tmp_path / 'model.yaml')
    evaluate_model(tmp_path)
    model = read(tmp_path / 'model', 'model').data
    fits.writeto(tmp_path / 'data.fits', model)
    observation = config['observations'][0]
    observation['data'] = dict(
        data='data.fits', error=0.01 * float(model.max()),
        step=observation['observable'].pop('step'))
    observation['observable'].pop('size')
    config['params']['properties'] |= dict(
        xpos=dict(value=0, min=-3, max=3),
        posa=dict(value=40, min=0, max=90))
    config['fitter'] = dict(type='cmaes', maxfevals=300)
    yaml.dump(config, tmp_path / 'fit.yaml')
    run_task(
        tmp_path, 'fit', config='fit.yaml', output_dir='fit',
        output_dir_mode='unique')
    first = read(tmp_path / 'fit', 'fit_result')
    run_task(tmp_path, 'resume', output_dir='fit')
    resumed = read(tmp_path / 'fit', 'fit_result')
    assert resumed['solutions'][0]['free'] == first['solutions'][0]['free']


@pytest.mark.slow
def test_a_fit_is_bootstrapped_as_it_ends_or_later(tmp_path):
    # With a section 'bootstrap', the fit refits its residuals redrawn as
    # it ends; later, the fit is bootstrapped again from its output
    # directory, by another configuration, with another fitter
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    yaml.dump(config, tmp_path / 'model.yaml')
    evaluate_model(tmp_path)
    model = read(tmp_path / 'model', 'model').data
    fits.writeto(tmp_path / 'data.fits', model)
    observation = config['observations'][0]
    observation['data'] = dict(
        data='data.fits', error=0.01 * float(model.max()),
        step=observation['observable'].pop('step'))
    observation['observable'].pop('size')
    config['params']['properties'] |= dict(
        xpos=dict(value=0, min=-3, max=3),
        posa=dict(value=40, min=0, max=90))
    config['fitter'] = dict(type='scipy.least_squares')
    scheme = dict(type='blocks', size=[3, 3, 2])
    config['bootstrap'] = dict(nsamples=3, scheme=scheme)
    yaml.dump(config, tmp_path / 'fit.yaml')
    run_task(
        tmp_path, 'fit', config='fit.yaml', output_dir='fit',
        output_dir_mode='unique')
    result = read(tmp_path / 'fit', 'fit_result')
    assert (result['bootstrap']['nsamples'],
            result['bootstrap']['nfailed']) == (3, 0)
    assert len(read(tmp_path / 'fit', 'bootstrap')) == 3
    later = dict(bootstrap=dict(
        nsamples=4, scheme=scheme,
        fitter=dict(type='scipy.least_squares', method='lm')))
    yaml.dump(later, tmp_path / 'later.yaml')
    run_task(tmp_path, 'bootstrap', output_dir='fit', config='later.yaml')
    result = read(tmp_path / 'fit', 'fit_result')
    assert result['bootstrap']['nsamples'] == 4
    assert result['bootstrap']['options']['fitter']['method'] == 'lm'
    assert len(read(tmp_path / 'fit', 'bootstrap')) == 4
    assert read(tmp_path / 'fit', 'bootstrap_config') == later
