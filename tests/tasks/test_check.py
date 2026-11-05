"""
Tests for the check of configurations (gbkfit.tasks.check): their
warnings and errors, with their paths, also of texts not saved yet.
"""

import pathlib

import ruamel.yaml

from gbkfit import project
from gbkfit.tasks.check import Problem, check


REFERENCE_DIR = pathlib.Path(__file__).parents[1] / 'data' / 'reference_models'

yaml = ruamel.yaml.YAML(typ='safe')


def test_the_problems_of_a_configuration_have_their_paths(tmp_path):
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    component = config['galaxies'][0]['components'][0]
    component['rnode'] = component.pop('rnodes')
    filename = tmp_path / 'config.yaml'
    yaml.dump(config, filename)
    path = ['galaxies', 0, 'components', 0]
    # (an unknown option is an error, and the check goes on past it)
    (unknown, error) = check(str(filename)).problems
    assert (unknown.severity, unknown.path) == ('error', path)
    assert "'rnode' (did you mean 'rnodes'?)" in unknown.message
    assert (error.severity, error.path) == ('error', path)
    # (a text in place of the file: fixed, and then without problems)
    component['rnodes'] = component.pop('rnode')
    text = filename.read_text()
    yaml.dump(config, filename)
    fixed = filename.read_text()
    filename.write_text(text)
    assert check(str(filename), {str(filename): fixed}).problems == []


def test_the_sections_of_tasks_are_checked_too(tmp_path):
    # (the mock data of eval, the bootstrap of fit: their errors showed
    # only when the task ran)
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    config['mock'] = dict(noise=dict(type='gauss', rms=1), nmock=2)
    filename = tmp_path / 'config.yaml'
    yaml.dump(config, filename)
    (problem,) = check(str(filename)).problems
    assert problem.path == ['mock'] and "'nmock'" in problem.message



def test_the_files_of_the_sections_of_tasks_are_those_of_the_root(
        tmp_path, monkeypatch):
    # (the error map of a mock, next to the configuration, checked from
    # another directory: it was looked for there, audit 2026-10-31 #33)
    import astropy.io.fits
    import numpy as np
    config = yaml.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.yaml')
    config['mock'] = dict(noise=dict(type='gauss', error='error.fits'))
    folder = tmp_path / 'project'
    folder.mkdir()
    astropy.io.fits.writeto(folder / 'error.fits', np.ones((2, 2, 2)))
    yaml.dump(config, folder / 'config.yaml')
    monkeypatch.chdir(tmp_path)
    problems = check(str(folder / 'config.yaml')).problems
    assert not [p for p in problems if 'error.fits' in p.message]

def test_a_configuration_that_cannot_be_read_has_an_error(tmp_path):
    filename = tmp_path / 'config.yaml'
    assert check(str(filename), {str(filename): 'galaxies: [\n'}) \
        .problems[0].severity == 'error'
    version = project.FORMAT_VERSION
    message = (
        f"the configuration has no version: 'version: {version}' for the "
        f"format of this gbkfit (gbkfit-cli migrate rewrites configurations "
        f"of older ones)")
    assert check(str(filename), {str(filename): 'fitter: {}\n'}).problems \
        == [Problem('error', [], message)]
    text = f'version: {version}\nfitter: {{}}\n'
    assert check(str(filename), {str(filename): text}).problems \
        == [Problem('error', [], "the configuration needs the sections "
                                 "['galaxies', 'observations', 'params']")]


def test_the_templates_of_a_project_make_a_setup_without_problems(tmp_path):
    study = project.create(str(tmp_path / 'study'))
    folders = (
        'galaxies', 'observations', 'psfs', 'lsfs', 'noise', 'params',
        'fitters')
    for folder in folders:
        study.create_piece(folder, 'first')
    # (the observation names the PSF and the LSF of the project)
    observation = pathlib.Path(study.piece_file('observations', 'first'))
    named = observation.read_text() \
        .replace('psf: {type: gauss, sigma: 1.5}', 'psf: first') \
        .replace('lsf: {type: gauss, sigma: 15.0}', 'lsf: first')
    assert 'psf: first' in named and 'lsf: first' in named
    observation.write_text(named)
    setup = pathlib.Path(study.create_piece('setups', 'first'))
    assert 'galaxies: [first]' in setup.read_text()
    setup.write_text(setup.read_text() + 'mock: {noise: first}\n')
    assert check(str(setup)).problems == []
