"""
Tests for the migration of configurations (gbkfit.migration): the
loaders refuse old forms with their new form, and the migration rewrites
files (with their comments) and projects.
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


@pytest.mark.parametrize('options, path, message', [
    (dict(calibration=True), 'calibration', "is now flux_scale: 'one'"),
    (dict(likelihood=dict(
        type='gaussian', amplitudes='per_spectrum', amplitude_sigma=0.5)),
     'likelihood',
     r"\{solve: \{type: gaussian, mean: 1, sigma: 0.5, minimum: 0\}\}"),
    (dict(scale=[2, 2, 1]), 'scale', "'scale' is now oversampling"),
    (dict(weights=dict(component='disk')), 'weights',
     "'weights' is now pixel_weights")])
def test_old_forms_are_refused_with_their_new_form(options, path, message):
    config = dict(observations=[OBSERVATION | options])
    with pytest.raises(ConfigError, match=message) as error:
        setup._prepare_config(config, (), ('observations',))
    assert error.value.path == ['observations', 0, path]
    assert "gbkfit-cli migrate rewrites it" in error.value.message


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
         json.dumps(dict(path=str(filename)))],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "kmos_calibration -> kmos_flux_scale" in result.stderr
    assert filename.read_text() == (
        "observations:\n"
        "- name: kmos\n"
        "  flux_scale: one    # the flux of the model is off\n"
        "params:\n"
        "  vsys: 0\n"
        "  kmos_flux_scale: {value: 1, min: 0.5, max: 2}   # free\n")
    # (a configuration in the current format is left as it is)
    assert migration.migrate_file(str(filename)) == []


def test_amplitudes_are_migrated_to_a_solved_flux_scale():
    config = dict(
        observations=[OBSERVATION | dict(likelihood=dict(
            type='gaussian', amplitudes='per_spectrum',
            amplitude_sigma=0.5))],
        params=dict(vsys=0))
    migration.migrate(config)
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


def test_the_old_disk_types_are_refused_and_migrated():
    components = [
        dict(type='smdisk', name='a'),
        dict(type='mcdisk', name='b', cflux=1e-3, seed=2),
        dict(type='smcone', name='c')]
    config = dict(galaxies=[dict(components=components)])
    with pytest.raises(ConfigError, match="type 'smdisk' is now type 'disk'"):
        setup._prepare_config(config, (), ('galaxies',))
    migration.migrate(config)
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
    migration.migrate(dict(galaxies=[dict(components=components)]))
    assert [c.get('geometry') == 'own' for c in components] == owns


def test_the_old_observation_options_are_migrated():
    observation = OBSERVATION | dict(
        scale=[2, 2, 1], weights=dict(component='disk'))
    migration.migrate(dict(observations=[observation]))
    assert observation == OBSERVATION | dict(
        oversampling=[2, 2, 1], pixel_weights=dict(component='disk'))
