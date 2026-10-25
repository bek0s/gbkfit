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
    component['rnmx'] = component.pop('rnmax')
    filename = tmp_path / 'config.yaml'
    yaml.dump(config, filename)
    path = ['galaxies', 0, 'components', 0]
    # (an unknown option is an error, and the check goes on past it)
    (unknown, error) = check(str(filename)).problems
    assert (unknown.severity, unknown.path) == ('error', path)
    assert "'rnmx' (did you mean 'rnmax'?)" in unknown.message
    assert (error.severity, error.path) == ('error', path)
    # (a text in place of the file: fixed, and then without problems)
    component['rnmax'] = component.pop('rnmx')
    text = filename.read_text()
    yaml.dump(config, filename)
    fixed = filename.read_text()
    filename.write_text(text)
    assert check(str(filename), {str(filename): fixed}).problems == []


def test_a_configuration_that_cannot_be_read_has_an_error(tmp_path):
    filename = tmp_path / 'config.yaml'
    assert check(str(filename), {str(filename): 'galaxies: [\n'}) \
        .problems[0].severity == 'error'
    assert check(str(filename), {str(filename): 'fitter: {}\n'}).problems \
        == [Problem('error', [], "the configuration needs the sections "
                                 "['galaxies', 'observations', 'params']")]


def test_the_templates_of_a_project_make_a_setup_without_problems(tmp_path):
    study = project.create(str(tmp_path / 'study'))
    for folder in ('galaxies', 'observations', 'params', 'fitters'):
        study.create_piece(folder, 'first')
    setup = study.create_piece('setups', 'first')
    assert 'galaxies: [first]' in pathlib.Path(setup).read_text()
    assert check(setup).problems == []
