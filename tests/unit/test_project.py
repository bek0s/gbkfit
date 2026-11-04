"""
Tests for projects (gbkfit.project): configurations name their pieces,
and the layers of their parameters merge in order.
"""

import os
from pathlib import Path

import pytest
import ruamel.yaml

from gbkfit import project
from gbkfit.config import ConfigError


yaml = ruamel.yaml.YAML(typ='safe')

GALAXY = dict(components=[dict(type='disk', name='disk')])
OBSERVATION = dict(data=dict(data='data/ngc1313/hi.fits'))
DATASET = dict(type='pixel_spectra', data='data/ngc1313/hi.fits')
PSF = dict(type='gauss', sigma=1.5)
LSF = dict(type='gauss', sigma=15)
NOISE = dict(type='gauss', rms=0.01)
FITTER = dict(type='nautilus', n_live=2000)
PRIOR = dict(type='gauss', expr='disk_incl', mean=60, std=10)


def write(path, value):
    """Write a YAML file, and its folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml.dump(value, path)


@pytest.fixture
def study(tmp_path):
    """A project with a piece of each kind, one in a folder."""
    path = tmp_path / 'study'
    study = project.create(path)
    write(path / 'galaxies/arctan.yaml', GALAXY)
    write(path / 'observations/ngc1313/hi.yaml', OBSERVATION)
    write(path / 'params/bounds.yaml', dict(properties=dict(
        disk_vsys=dict(value=0, min=-500, max=500),
        disk_incl=dict(value=45, min=0, max=90))))
    write(path / 'params/ngc1313.yaml', dict(
        properties=dict(disk_vsys=dict(value=470, min=400, max=540)),
        priors=[PRIOR]))
    write(path / 'fitters/nautilus.yaml', FITTER)
    return study


def test_a_setup_has_its_pieces_in_place(study):
    setup = dict(
        galaxies=['arctan', GALAXY], observations='ngc1313/hi',
        params=['bounds', 'ngc1313', dict(properties=dict(disk_incl=60))],
        fitter='nautilus', driver=dict(type='host'))
    filename = os.path.join(study.path, 'setups/ngc1313/arctan.yaml')
    config, sources = project.resolve(setup, filename, study)
    assert config == dict(
        galaxies=[GALAXY, GALAXY], observations=[OBSERVATION],
        params=dict(
            properties=dict(
                disk_vsys=dict(value=470, min=400, max=540), disk_incl=60),
            priors=[PRIOR]),
        fitter=FITTER, driver=dict(type='host'))
    assert sources == dict(
        file='setups/ngc1313/arctan.yaml',
        galaxies=['galaxies/arctan.yaml', 'setups/ngc1313/arctan.yaml'],
        observations=['observations/ngc1313/hi.yaml'],
        data=['observations/ngc1313/hi.yaml'],
        params=[
            'params/bounds.yaml', 'params/ngc1313.yaml',
            'setups/ngc1313/arctan.yaml'],
        parameters=dict(
            disk_vsys='params/ngc1313.yaml',
            disk_incl='setups/ngc1313/arctan.yaml'),
        fitter='fitters/nautilus.yaml')
    assert study.setup_name(filename) == os.path.join('ngc1313', 'arctan')
    assert study.setup_name(os.path.join(study.path, 'fit.yaml')) is None


def test_observations_name_their_parts_and_mock_data_their_noise(study):
    path = Path(study.path)
    write(path / 'datasets/ngc1313/hi.yaml', DATASET)
    write(path / 'psfs/beam.yaml', PSF)
    write(path / 'noise/radio.yaml', NOISE)
    write(path / 'observations/named.yaml', dict(
        data='ngc1313/hi', instrument=dict(psf='beam', lsf=LSF)))
    setup = dict(
        observations=['named', 'ngc1313/hi'],
        mock=dict(noise=['radio', None], seed=1))
    filename = os.path.join(study.path, 'setups/mock.yaml')
    config, sources = project.resolve(setup, filename, study)
    assert config == dict(
        observations=[
            dict(data=DATASET, instrument=dict(psf=PSF, lsf=LSF)),
            OBSERVATION],
        mock=dict(noise=[NOISE, None], seed=1))
    # (the parts in place are from the files of their observations)
    assert sources['data'] == [
        'datasets/ngc1313/hi.yaml', 'observations/ngc1313/hi.yaml']
    assert sources['psf'] == ['psfs/beam.yaml', None]
    assert sources['lsf'] == ['observations/named.yaml', None]
    assert sources['noise'] == ['noise/radio.yaml', None]


def test_a_layer_replaces_all_the_properties_of_a_parameter():
    params, owners = project.merge_params([
        dict(properties={'vt': dict(value=100), 'vt[0]': 0, 'incl': 45},
             priors=[PRIOR], derived=dict(vmax='max(vt)')),
        dict(properties={'vt[1:]': dict(value=150)}, priors=[PRIOR],
             derived=dict(v22='vt[2]'), transforms=dict(file='t.py'))])
    assert params == dict(
        properties={'incl': 45, 'vt[1:]': dict(value=150)},
        priors=[PRIOR, PRIOR], derived=dict(vmax='max(vt)', v22='vt[2]'),
        transforms=dict(file='t.py'))
    assert owners == dict(vt=1, incl=0)


@pytest.mark.parametrize('setup, message', [
    (dict(galaxies=['spiral']), "galaxies[0]: the project has no piece "
                                "'spiral' of galaxies (galaxies/spiral.yaml)"),
    (dict(fitter='../galaxies/arctan'), "must be a path in fitters/"),
    (dict(observations=[dict(instrument=dict(psf='beam'))]),
     "observations[0].instrument.psf: the project has no piece 'beam' of "
     "psfs (psfs/beam.yaml)"),
    (dict(params=['bounds', [3]]), "params[1]: a layer of parameters must "
                                   "be a dictionary")])
def test_invalid_names_and_layers(study, setup, message):
    filename = os.path.join(study.path, 'setups/fit.yaml')
    with pytest.raises(ConfigError) as e:
        project.resolve(setup, filename, study)
    assert message in str(e.value)


def test_pieces_need_a_project(tmp_path):
    with pytest.raises(ConfigError, match="not in one"):
        project.resolve(
            dict(galaxies=['arctan']), str(tmp_path / 'fit.yaml'), None)
    # (in place, and in layers, they do not)
    config, _ = project.resolve(
        dict(galaxies=GALAXY, params=[dict(properties=dict(a=1))]),
        str(tmp_path / 'fit.yaml'), None)
    assert config == dict(galaxies=GALAXY, params=dict(properties=dict(a=1)))


def test_a_project_is_found_above_its_files(study, tmp_path):
    found = project.find(os.path.join(study.path, 'setups/ngc1313/a.yaml'))
    assert found.path == study.path
    assert project.find(tmp_path) is None
    write(tmp_path / 'newer' / project.MARKER, dict(
        format=project.FORMAT, version=project.FORMAT_VERSION + 1))
    with pytest.raises(ValueError, match="newer"):
        project.find(tmp_path / 'newer' / 'setups')


def test_pieces_are_made_from_templates_in_their_folders(study):
    path = study.create_piece('observations', 'ngc1313/halpha')
    assert path == os.path.join(study.path, 'observations/ngc1313/halpha.yaml')
    with pytest.raises(FileExistsError):
        study.create_piece('observations', 'ngc1313/halpha')
    path = study.create_piece('psfs', 'beam', PSF)
    assert yaml.load(Path(path)) == PSF
    with pytest.raises(ConfigError, match="pieces are in the folders"):
        study.create_piece('runs', 'x')
    with pytest.raises(ConfigError, match="must be a path in setups/"):
        study.create_piece('setups', '../x')


def test_pieces_are_renamed_with_the_names_of_them(study):
    path = Path(study.path)
    (path / 'datasets').mkdir(exist_ok=True)
    write(path / 'datasets/ngc1313/hi.yaml', DATASET)
    write(path / 'psfs/beam.yaml', PSF)
    (path / 'observations/ngc1313/hi.yaml').write_text(
        "# The HI of NGC 1313\n"
        "data: ngc1313/hi\n"
        "# its instrument\n"
        "instrument: {psf: beam, lsf: {type: gauss, sigma: 15}}\n")
    (path / 'setups').mkdir(exist_ok=True)
    (path / 'setups/fit.yaml').write_text(
        "galaxies: [arctan]\n"
        "observations:\n"
        "- ngc1313/hi\n"
        "- {data: ngc1313/hi, instrument: {psf: beam}}\n"
        "params: [bounds, ngc1313]\n"
        "fitter: nautilus\n")
    assert study.references('datasets', 'ngc1313/hi') == [
        ('observations', 'ngc1313/hi'), ('setups', 'fit')]
    assert study.references('psfs', 'beam') == [
        ('observations', 'ngc1313/hi'), ('setups', 'fit')]
    assert study.references('galaxies', 'arctan') == [('setups', 'fit')]
    assert study.references('datasets', 'other') == []
    # (renamed, the names of it follow; comments and layout stay)
    target = study.rename_piece('datasets', 'ngc1313/hi', 'hi')
    assert target == str(path / 'datasets/hi.yaml')
    assert not (path / 'datasets/ngc1313').exists()
    assert (path / 'observations/ngc1313/hi.yaml').read_text() == (
        "# The HI of NGC 1313\n"
        "data: hi\n"
        "# its instrument\n"
        "instrument: {psf: beam, lsf: {type: gauss, sigma: 15}}\n")
    assert yaml.load(path / 'setups/fit.yaml')['observations'] == [
        'ngc1313/hi', dict(data='hi', instrument=dict(psf='beam'))]
    study.rename_piece('observations', 'ngc1313/hi', 'hi')
    assert yaml.load(path / 'setups/fit.yaml')['observations'][0] == 'hi'
    with pytest.raises(FileExistsError):
        study.rename_piece('galaxies', 'arctan', 'arctan')
    with pytest.raises(FileNotFoundError):
        study.rename_piece('galaxies', 'nothing', 'other')
    with pytest.raises(ConfigError, match="must be a path in galaxies/"):
        study.rename_piece('galaxies', 'arctan', '../arctan')
    # (deleted, the names of it stay)
    study.delete_piece('psfs', 'beam')
    assert not (path / 'psfs/beam.yaml').exists()
    assert study.references('psfs', 'beam') == [
        ('observations', 'hi'), ('setups', 'fit')]
    with pytest.raises(FileNotFoundError):
        study.delete_piece('psfs', 'beam')
    with pytest.raises(ConfigError, match="pieces are in the folders"):
        study.delete_piece('runs', 'x')


def test_pieces_are_updated_in_place(study):
    path = study.create_piece('psfs', 'beam')
    # (the comment of the template, which says what it is, goes as it
    # changes; others stay, as do the options that do not change)
    study.update_piece('psfs', 'beam', dict(type='gauss', sigma=1.5))
    assert Path(path).read_text().startswith('# A PSF')
    study.update_piece('psfs', 'beam', dict(type='gauss', sigma=2.0))
    assert Path(path).read_text() == 'type: gauss\nsigma: 2.0\n'
    Path(path).write_text('# my seeing\ntype: gauss  # round\nsigma: 2.0\n')
    content = dict(type='gauss', sigma=1.0)
    study.update_piece('psfs', 'beam', content)
    assert yaml.load(Path(path)) == content
    assert Path(path).read_text().startswith('# my seeing')
    assert '# round' in Path(path).read_text()
    with pytest.raises(FileNotFoundError):
        study.update_piece('psfs', 'nothing', content)
