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


def run_eval(mode, config, workdir, *options):
    """
    Run `gbkfit-cli eval` (with the given command line options) and return
    its FITS outputs by file name.
    """
    workdir.mkdir()
    yaml.dump(config, workdir / 'config.yaml')
    result = subprocess.run(
        [sys.executable, '-m', 'gbkfit.apps.cli', 'eval', mode, 'config.yaml',
         *options],
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
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    model = run_eval('model', config, tmp_path / 'model')['model_0_spectra_d']
    fits.writeto(tmp_path / 'data.fits', model + 1)
    observation = config['observations'][0]
    observation['data'] = dict(
        data=str(tmp_path / 'data.fits'), error=2.0,
        step=observation['observable'].pop('step'))
    observation['observable'].pop('size')
    observation['likelihood'] = dict(type='gaussian', weights=0.5)
    outputs = run_eval('objective', config, tmp_path / 'objective')
    np.testing.assert_allclose(outputs['residual_spectra_d'], -0.5, rtol=1e-5)
    np.testing.assert_allclose(outputs['wresidual_spectra_d'], -0.25, rtol=1e-5)
    # The extra outputs of the model (e.g. the high-resolution cube)
    assert 'residual_extra_observation0_dcube_hi' in outputs


@pytest.mark.parametrize('mode', ['model', 'objective'])
def test_profiling_writes_the_timings(tmp_path, mode):
    # The timings of the profiled evaluations only: model_eval once for
    # each of the 3 in model mode, and twice for each in objective mode
    # (the likelihood and the residual sum)
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    if mode == 'objective':
        model = run_eval('model', config, tmp_path / 'model')
        fits.writeto(tmp_path / 'data.fits', model['model_0_spectra_d'])
        observation = config['observations'][0]
        observation['data'] = dict(
            data=str(tmp_path / 'data.fits'), error=1.0,
            step=observation['observable'].pop('step'))
        observation['observable'].pop('size')
    run_eval(mode, config, tmp_path / mode, '--profile', '3')
    timings = yaml.load(tmp_path / mode / 'output' / 'gbkfit_eval_timings.yaml')
    expected = dict(model=3, objective=6)[mode]
    assert timings['model_eval']['count'] == expected


def test_rotation_curve_from_radial_nodes(tmp_path):
    # A rotation curve given by an expression of the radial nodes of the
    # disk and of user-defined parameters is the same as the one given by
    # its values
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    component = config['models'][0]['components'][0]
    for option in ('rnmin', 'rnmax', 'rnsep'):
        component.pop(option, None)
    rnodes = list(range(0, 21, 2))
    component.update(rnodes=rnodes, vptraits=dict(type='nw_tan_uniform'))
    properties = config['params']['properties']
    del properties['vpt_rt']
    config['pdescs'] = dict(vmax=dict(type='scalar'), rt=dict(type='scalar'))
    properties.update(vmax=180, rt=3, vpt_vt='vmax * np.arctan(rnodes / rt)')
    expression = run_eval('model', config, tmp_path / 'expression')
    del config['pdescs'], properties['vmax'], properties['rt']
    properties['vpt_vt'] = (180 * np.arctan(np.array(rnodes) / 3)).tolist()
    values = run_eval('model', config, tmp_path / 'values')
    np.testing.assert_array_equal(
        expression['model_0_spectra_d'], values['model_0_spectra_d'])


def test_unknown_options(tmp_path):
    # A misspelt option is a warning that suggests the right one, or an
    # error with --strict
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    component = config['models'][0]['components'][0]
    component['rnmx'] = component['rnmax']
    yaml.dump(config, tmp_path / 'config.yaml')

    def run(*options):
        return subprocess.run(
            [sys.executable, '-m', 'gbkfit.apps.cli', 'eval', 'model',
             'config.yaml', *options],
            cwd=tmp_path, capture_output=True, text=True)
    result = run()
    assert result.returncode == 0
    assert "'rnmx' (did you mean 'rnmax'?)" in result.stderr
    result = run('--strict')
    assert result.returncode != 0
    assert "models[0].components[0]" in result.stderr
    assert "'rnmx' (did you mean 'rnmax'?)" in result.stderr


def test_outputs_have_the_world_coordinates_of_the_model(tmp_path):
    # The model is written with the coordinates of its dmodel, so it can
    # be read back with them
    from gbkfit.utils import fitsutils
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    config['observations'][0]['observable'].update(rval=[150, 2, 1500], rota=30)
    run_eval('model', config, tmp_path / 'model')
    _, coords = fitsutils.read_data(
        tmp_path / 'model' / 'output' / 'model_0_spectra_d.fits')
    np.testing.assert_allclose(coords.step, [1, 1, 10], rtol=1e-12)
    np.testing.assert_allclose(coords.rpix, [23.5, 23.5, 24.5], rtol=1e-12)
    np.testing.assert_allclose(coords.rval, [150, 2, 1500], rtol=1e-12)
    np.testing.assert_allclose(coords.rota, 30, atol=1e-9)


def test_outputs_are_written_by_type(tmp_path):
    # Data on a grid gets world coordinates, other arrays none, and the
    # other values go together to gbkfit_eval_extra.json and .yaml
    import json
    from gbkfit.tasks import eval as eval_task
    from gbkfit.utils import gridutils
    coords = gridutils.Coords((1.0, 1.0), (3.5, 3.5), (150.0, 2.0), 0.0)
    eval_task._write_outputs(tmp_path, dict(
        grid=gridutils.GridData(np.ones((8, 8)), coords, None),
        array=np.ones((4, 4)),
        number=np.int64(12345),
        info=dict(name='disk', sizes=[1, 2])))
    assert fits.getheader(tmp_path / 'grid.fits')['CTYPE1'] == 'RA---TAN'
    assert 'CTYPE1' not in fits.getheader(tmp_path / 'array.fits')
    values = json.loads((tmp_path / 'gbkfit_eval_extra.json').read_text())
    assert values == dict(number=12345, info=dict(name='disk', sizes=[1, 2]))
    assert (tmp_path / 'gbkfit_eval_extra.yaml').exists()
    with pytest.raises(TypeError, match="unsupported type: object"):
        eval_task._write_outputs(tmp_path, dict(thing=object()))


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
    model = run_eval('model', config, tmp_path / 'model')['model_0_spectra_d']
    assert model.shape == (50, 2)
    header = fits.getheader(
        tmp_path / 'model' / 'output' / 'model_0_spectra_d.fits')
    assert header['CTYPE2'] == 'VRAD'
    assert header['CDELT2'] == 10
    fits.writeto(tmp_path / 'data.fits', model + 1, header)
    observable = observation['observable']
    for key in ('regions', 'spec_size', 'spec_step'):
        observable.pop(key)
    observation['data'] = dict(
        regions=regions, data=str(tmp_path / 'data.fits'), error=2.0)
    outputs = run_eval('objective', config, tmp_path / 'objective')
    np.testing.assert_allclose(outputs['residual_spectra_d'], -0.5, rtol=1e-5)


def test_one_model_seen_by_two_observations(tmp_path):
    # The observations refer to their gmodel, so there can be more
    # observations than gmodels
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    scube = config['observations'][0]
    mmaps = dict(driver=dict(type='host'), name='maps',
                 observable=dict(type='pixel_moments', size=[48, 48]))
    config['observations'] = [scube | dict(name='cube'), mmaps]
    outputs = run_eval('model', config, tmp_path / 'model')
    assert outputs['model_0_spectra_d'].shape == (50, 48, 48)
    assert outputs['model_1_moment1_d'].shape == (48, 48)
