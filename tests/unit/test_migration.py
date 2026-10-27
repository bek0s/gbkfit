"""
Tests for the migration of configurations (gbkfit.migration): the
loaders refuse configurations of other versions of the format, and the
migration rewrites those of older ones (files, with their comments, and
projects) and leaves the current ones as they are.
"""

import json
import subprocess
import sys

import pytest
import ruamel.yaml

from gbkfit import migration, project, setup
from gbkfit.utils.parseutils import ConfigError


yaml = ruamel.yaml.YAML(typ='safe')

OBSERVATION = dict(
    type='sky', name='kmos', line_of_sight=dict(type='flat'),
    observable=dict(type='pixel_spectra', size=[4, 4, 5]))


@pytest.mark.parametrize('version, message', [
    (None, "has no version: 'version: 3'"),
    (2, "is of version 2 of the format, not 3"),
    (4, "is of version 4 of the format, not 3")])
def test_configurations_of_other_versions_are_refused(version, message):
    config = dict(galaxies=[], observations=[], params={})
    if version is not None:
        config['version'] = version
    with pytest.raises(ConfigError, match=message) as error:
        setup.load(config)
    assert "gbkfit-cli migrate" in error.value.message


@pytest.mark.parametrize('config, from_version, message', [
    (dict(), None, "has no version"),
    (dict(version=2), 1, "is of version 2; not 1"),
    (dict(version=4), None, "versions of the format are 1 to 3")])
def test_the_version_of_a_migration_must_be_known(
        config, from_version, message):
    with pytest.raises(ConfigError, match=message):
        migration.migrate(config, from_version)


def test_a_file_is_migrated_in_place_with_its_comments(tmp_path):
    filename = tmp_path / 'config.yaml'
    filename.write_text(
        "observations:\n"
        "- name: kmos\n"
        "  calibration: true  # the flux of the model is off\n"
        "params:\n"
        "  vsys: 0\n"
        "  kmos_calibration: {value: 1, min: 0.5, max: 2}  # free\n")
    result = subprocess.run(
        [sys.executable, '-m', 'gbkfit.tasks', 'migrate',
         json.dumps(dict(path=str(filename), from_version=1))],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "kmos_calibration -> kmos_flux_scale" in result.stderr
    assert filename.read_text() == (
        "version: 3\n"
        "observations:\n"
        "- name: kmos\n"
        "  flux_scale: one    # the flux of the model is off\n"
        "params:\n"
        "  vsys: 0\n"
        "  kmos_flux_scale: {value: 1, min: 0.5, max: 2}   # free\n")
    # (a configuration of the current version is left as it is)
    assert migration.migrate_file(str(filename)) == []


def test_amplitudes_are_migrated_to_a_solved_flux_scale():
    config = dict(
        observations=[OBSERVATION | dict(likelihood=dict(
            type='gaussian', amplitudes='per_spectrum',
            amplitude_sigma=0.5))],
        params=dict(vsys=0))
    migration.migrate(config, 1)
    assert config['observations'][0]['flux_scale'] == 'per_spectrum'
    assert config['observations'][0]['likelihood'] == dict(type='gaussian')
    assert config['params']['kmos_flux_scale'] == dict(solve=dict(
        type='gaussian', mean=1, sigma=0.5, minimum=0))


def test_a_project_is_migrated_across_its_pieces(tmp_path):
    study = project.create(str(tmp_path / 'study'))
    yaml.dump(
        dict(format=project.FORMAT, version=1),
        tmp_path / 'study' / project.MARKER)
    yaml.dump(
        OBSERVATION | dict(calibration=True),
        tmp_path / 'study' / 'observations' / 'kmos.yaml')
    yaml.dump(
        dict(kmos_calibration=1),
        tmp_path / 'study' / 'params' / 'flux.yaml')
    changes = migration.migrate_project(study.path)
    assert changes == {
        'observations/kmos.yaml': [
            migration.Change([], "calibration -> flux_scale: one")],
        'params/flux.yaml': [
            migration.Change([], "kmos_calibration -> kmos_flux_scale")]}
    assert study.piece('observations', 'kmos')['flux_scale'] == 'one'
    assert study.piece('params', 'flux') == dict(kmos_flux_scale=1)
    assert yaml.load(tmp_path / 'study' / project.MARKER)['version'] == \
        project.FORMAT_VERSION


def test_a_project_of_the_current_version_is_left_as_it_is(tmp_path):
    # (the template galaxy is named: a migration from version 1 would
    # give it its own geometry, and its parameters other names)
    study = project.create(str(tmp_path / 'study'))
    for folder in ('galaxies', 'observations', 'params', 'setups'):
        study.create_piece(folder, 'first')
    galaxy = study.piece('galaxies', 'first')
    assert migration.migrate_project(study.path) == {}
    assert study.piece('galaxies', 'first') == galaxy


def test_the_old_disk_types_are_migrated():
    components = [
        dict(type='smdisk', name='a'),
        dict(type='mcdisk', name='b', cflux=1e-3, seed=2),
        dict(type='smcone', name='c')]
    migration.migrate(dict(galaxies=[dict(components=components)]), 1)
    assert [c['type'] for c in components] == ['disk', 'disk', 'cone']
    assert components[1] == dict(
        type='disk', name='b', clouds=dict(cflux=1e-3, seed=2),
        geometry='own')


@pytest.mark.parametrize('components, owns', [
    # (several components, a named one, a warped one; not one of the
    # galaxy alone, whose parameters keep their names)
    ([dict(type='disk', name='a'), dict(type='point', name='b')],
     [True, True]),
    ([dict(type='disk', name='a')], [True]),
    ([dict(type='disk', tilted=True)], [True]),
    ([dict(type='disk', loose=False)], [False]),
    ([dict(type='disk', name='a', geometry='g')], [False])])
def test_old_components_keep_their_own_geometries(components, owns):
    migration.migrate(dict(galaxies=[dict(components=components)]), 1)
    assert [c.get('geometry') == 'own' for c in components] == owns


def test_components_of_version_2_keep_the_geometry_of_their_galaxy():
    components = [dict(type='disk', name='a'), dict(type='point', name='b')]
    migration.migrate(dict(galaxies=[dict(components=components)]), 2)
    assert all('geometry' not in c for c in components)


def test_the_old_observation_options_are_migrated():
    observation = OBSERVATION | dict(
        scale=[2, 2, 1], weights=dict(component='disk'))
    migration.migrate(dict(observations=[observation]), 1)
    assert observation == OBSERVATION | dict(
        oversampling=[2, 2, 1], pixel_weights=dict(component='disk'))
