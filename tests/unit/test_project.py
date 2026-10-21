"""
Tests for projects (gbkfit.project): configurations name their pieces,
and the layers of their parameters merge in order.
"""

import os

import pytest
import ruamel.yaml

from gbkfit import project
from gbkfit.utils.parseutils import ConfigError


yaml = ruamel.yaml.YAML(typ='safe')

GALAXY = dict(components=[dict(type='smdisk', name='disk')])
OBSERVATION = dict(data=dict(data='data/ngc1313/hi.fits'))
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
        params=[
            'params/bounds.yaml', 'params/ngc1313.yaml',
            'setups/ngc1313/arctan.yaml'],
        parameters=dict(
            disk_vsys='params/ngc1313.yaml',
            disk_incl='setups/ngc1313/arctan.yaml'),
        fitter='fitters/nautilus.yaml')
    assert study.setup_name(filename) == os.path.join('ngc1313', 'arctan')
    assert study.setup_name(os.path.join(study.path, 'fit.yaml')) is None


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
