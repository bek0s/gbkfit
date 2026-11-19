"""
Tests of the parameters of the items of a setup: the params of an item in
its own names, which its prefix makes those of the setup (see
prefixed_params), the pieces that the items of a setup use under its
names, and the layers of params of an item (see gbkfit.project.resolve).
"""

from pathlib import Path

import pytest
import ruamel.yaml

from gbkfit import project
from gbkfit.config import ConfigError
from gbkfit.params import prefixed_params


yaml = ruamel.yaml.YAML(typ='safe')

GALAXY = dict(components=[dict(type='thin_disk', name='gas')])

# The parameters of the galaxy, in its own names
PARAMS = dict(properties=dict(
    gas_xpos=0, gas_posa=dict(value=30, min=0, max=90),
    gas_incl='gas_posa + 1'))


def write(study, name, value):
    """Write a piece or a setup of a project (e.g. 'params/ngc1')."""
    path = Path(study.path, f'{name}.yaml')
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml.dump(value, path)


def read(study, name):
    """Read a piece or a setup of a project."""
    return yaml.load(Path(study.path, f'{name}.yaml'))


@pytest.fixture
def study(tmp_path):
    """A project of a galaxy and its parameters, as pieces."""
    created = project.create(str(tmp_path / 'study'))
    write(created, 'galaxies/ngc1', GALAXY)
    write(created, 'params/ngc1', PARAMS)
    return created


def resolved(study, setup):
    return project.resolve(setup, f'{study.path}/setups/s.yaml', study)


def test_the_params_of_an_item_take_its_prefix():
    # (every name of it: keys, ties, derived values and their names, the
    # expressions of priors; not the names that expressions reserve, nor
    # attributes, e.g. the sin of np.sin)
    params = dict(
        properties={'gas_posa': 30, 'gas_incl[1:]': 'gas_posa[0] + 1',
                    'bulge_posa': 'np.sin(abs(mass_xpos))'},
        derived=dict(vmax='np.max(gas_vpt_vt)', twice=dict(expr='2 * vmax')),
        priors=[dict(type='joint', expr='gas_incl - bulge_posa')])
    assert prefixed_params(params, 'd_') == dict(
        properties={'d_gas_posa': 30, 'd_gas_incl[1:]': 'd_gas_posa[0] + 1',
                    'd_bulge_posa': 'np.sin(abs(d_mass_xpos))'},
        derived=dict(
            d_vmax='np.max(d_gas_vpt_vt)', d_twice=dict(expr='2 * d_vmax')),
        priors=[dict(type='joint', expr='d_gas_incl - d_bulge_posa')])
    # (their transforms are code, which names the parameters of the whole
    # group)
    with pytest.raises(ValueError, match='transforms'):
        prefixed_params(dict(transforms=dict(file='t.py')), 'd_')


def test_the_items_of_a_setup_use_pieces_under_its_names(study):
    # (one galaxy, the deflector of a lens in one setup and the source in
    # another, from one piece without a name)
    config = resolved(study, dict(galaxies=[
        dict(name='source', use='ngc1'), dict(name='deflector', use='ngc1')]))
    assert [g['name'] for g in config.config['galaxies']] == [
        'source', 'deflector']
    assert config.sources['galaxies'] == ['galaxies/ngc1.yaml'] * 2
    # (a piece of a name of its own is used by it only)
    write(study, 'galaxies/named', GALAXY | dict(name='ngc'))
    with pytest.raises(ConfigError, match="is named 'ngc'"):
        resolved(study, dict(galaxies=[dict(name='source', use='named')]))
    with pytest.raises(ConfigError, match='only the piece'):
        resolved(study, dict(galaxies=[dict(
            name='source', use='ngc1', redshift=0.1)]))


def test_the_params_of_an_item_are_in_its_own_names(study):
    # (from a piece, or in place; in order with the layers of the names of
    # the setup, e.g. ties across galaxies; their sources)
    config = resolved(study, dict(
        galaxies=[dict(name='source', use='ngc1'),
                  dict(name='deflector', use='ngc1')],
        params=[
            dict(of='source', use='ngc1'), dict(of='deflector', use=['ngc1']),
            dict(of='deflector', properties=dict(
                mass_xpos='gas_xpos', gas_incl=60)),
            dict(properties=dict(source_gas_xpos='deflector_gas_xpos'))]))
    assert config.config['params'] == dict(properties=dict(
        source_gas_posa=dict(value=30, min=0, max=90),
        source_gas_incl='source_gas_posa + 1',
        deflector_gas_xpos=0, deflector_gas_posa=dict(value=30, min=0, max=90),
        deflector_gas_incl=60, deflector_mass_xpos='deflector_gas_xpos',
        source_gas_xpos='deflector_gas_xpos'))
    assert config.sources['params'] == [
        'params/ngc1.yaml', ['params/ngc1.yaml'], 'setups/s.yaml',
        'setups/s.yaml']
    assert config.sources['parameters']['deflector_gas_incl'] \
        == 'setups/s.yaml'
    assert config.sources['parameters']['deflector_gas_posa'] \
        == 'params/ngc1.yaml'


def test_the_params_of_an_item_are_of_one_of_the_setup(study, tmp_path):
    with pytest.raises(ConfigError, match="'back', which names no galaxy"):
        resolved(study, dict(
            galaxies=[dict(name='front', use='ngc1')],
            params=[dict(of='back', use='ngc1')]))
    with pytest.raises(ConfigError, match='not both'):
        resolved(study, dict(
            galaxies=[dict(name='front', use='ngc1')],
            params=[dict(of='front', use='ngc1', properties=dict(a=1))]))
    # (in place, they are of a configuration outside a project too)
    config = project.resolve(dict(
        galaxies=[GALAXY | dict(name='front')],
        params=[dict(of='front', properties=dict(gas_xpos=1))]),
        str(tmp_path / 'one.yaml'), None)
    assert config.config['params'] == dict(properties=dict(front_gas_xpos=1))


def test_the_pieces_that_items_use_are_named_by_the_setup(study):
    # (as the setup names them: their links, and their renames)
    write(study, 'setups/lensed', dict(
        galaxies=[dict(name='deflector', use='ngc1')],
        params=[dict(of='deflector', use='ngc1')]))
    assert (('setups', 'lensed'), ('galaxies', 'ngc1')) in study.links()
    assert (('setups', 'lensed'), ('params', 'ngc1')) in study.links()
    study.rename_piece('params', 'ngc1', 'disk')
    assert read(study, 'setups/lensed') == dict(
        galaxies=[dict(name='deflector', use='ngc1')],
        params=[dict(of='deflector', use='disk')])

