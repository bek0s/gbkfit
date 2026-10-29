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
from gbkfit.observation import observable_parser
from gbkfit.utils.parseutils import ConfigError


yaml = ruamel.yaml.YAML(typ='safe')

OBSERVATION = dict(
    type='sky', name='kmos', line_of_sight=dict(type='flat'),
    observable=dict(
        type='pixel_spectra', size=[4, 4],
        spectral=dict(size=5, convention='relativistic')))


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
        "  properties:\n"
        "    vsys: 0\n"
        "    kmos_calibration: {value: 1, min: 0.5, max: 2}  # free\n")
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
        "  properties:\n"
        "    vsys: 0\n"
        "    kmos_flux_scale: {value: 1, min: 0.5, max: 2}   # free\n")
    # (a configuration of the current version is left as it is)
    assert migration.migrate_file(str(filename)) == []


def test_amplitudes_are_migrated_to_a_solved_flux_scale():
    config = dict(
        observations=[OBSERVATION | dict(likelihood=dict(
            type='gaussian', amplitudes='per_spectrum',
            amplitude_sigma=0.5))],
        params=dict(properties=dict(vsys=0)))
    migration.migrate(config, 1)
    assert config['observations'][0]['flux_scale'] == 'per_spectrum'
    assert config['observations'][0]['likelihood'] == dict(type='gaussian')
    assert config['params']['properties']['kmos_flux_scale'] == dict(
        solve=dict(type='gaussian', mean=1, sigma=0.5, min=0))


def test_a_project_is_migrated_across_its_pieces(tmp_path):
    study = project.create(str(tmp_path / 'study'))
    yaml.dump(
        dict(format=project.FORMAT, version=1),
        tmp_path / 'study' / project.MARKER)
    yaml.dump(
        OBSERVATION | dict(calibration=True),
        tmp_path / 'study' / 'observations' / 'kmos.yaml')
    yaml.dump(
        dict(properties=dict(kmos_calibration=1)),
        tmp_path / 'study' / 'params' / 'flux.yaml')
    changes = migration.migrate_project(study.path)
    assert changes == {
        'observations/kmos.yaml': [
            migration.Change([], "calibration -> flux_scale: one")],
        'params/flux.yaml': [
            migration.Change(
                ['properties'], "kmos_calibration -> kmos_flux_scale")]}
    assert study.piece('observations', 'kmos')['flux_scale'] == 'one'
    assert study.piece('params', 'flux') == dict(
        properties=dict(kmos_flux_scale=1))
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


def test_kinematics_are_migrated_to_typed_ones():
    components = [
        dict(type='disk', kinematics=dict(
            vptraits=dict(type='mass'), equilibrium='self_gravitating',
            equilibrium_density='gas', equilibrium_direction='ver')),
        dict(type='cone', kinematics=dict(vptraits=dict(type='uniform'))),
        dict(type='spheroid', kinematics=dict(type='jeans'))]
    migration.migrate(dict(galaxies=[dict(components=components)]), 2)
    assert [c['kinematics'] for c in components] == [
        dict(type='traits', vptraits=dict(type='mass'), equilibrium=dict(
            type='self_gravitating', density='gas', direction='ver')),
        dict(type='traits', vptraits=dict(type='uniform')),
        dict(type='jeans')]


def test_spectral_axes_are_migrated_to_one_option():
    moments = dict(
        type='pixel_moments', size=[8, 8], spec_step=10,
        spec_rest='5000 Angstrom', spec_convention='log')
    cube = dict(type='pixel_spectra', size=[8, 8, 20], step=[1, 1, 10])
    observations = [
        dict(type='sky', observable=moments),
        dict(type='sky', observable=cube),
        dict(type='sky', observable=dict(type='pixel_spectra'), data=dict(
            data='cube.fits', step=[1, 1, 10], rest='1420 MHz'))]
    migration.migrate(dict(observations=observations), 2)
    assert moments == dict(
        type='pixel_moments', size=[8, 8], spectral=dict(
            step=10, convention='log', rest='5000 Angstrom'))
    # (an axis without a rest had relativistic velocities)
    assert cube == dict(
        type='pixel_spectra', size=[8, 8], step=[1, 1],
        spectral=dict(size=20, step=10, convention='relativistic'))
    assert observations[2]['data'] == dict(
        type='pixel_spectra', data='cube.fits', step=[1, 1],
        spectral=dict(step=10, rest='1420 MHz'))


def test_the_field_of_visibilities_is_migrated_to_an_angle():
    given = dict(type='visibilities', size=[32, 28], step=0.5)
    default = dict(type='visibilities', size=[32, 28])
    changes = migration.migrate(dict(observations=[
        dict(observable=given), dict(observable=default)]), 2)
    assert given == dict(type='visibilities', fov=[16.0, 14.0], step=0.5)
    # (the default step depends on the coverage: the field is by hand)
    assert default == dict(type='visibilities')
    assert any(c.message.startswith("by hand: give fov") for c in changes)


def test_spectral_tables_are_migrated_to_the_options_of_their_class():
    psf = dict(type='moffat', beta=2.5, alpha=dict(
        velocity=[-500, 500], values=dict(values=[1, 2], unit='arcsec')))
    lsf = dict(type='gauss', sigma=dict(table='lsf.ecsv', column='sigma'))
    images = dict(type='images', file='psf.fits', wavelength=dict(
        values=[5000, 7000], unit='Angstrom'))
    migration.migrate(dict(observations=[
        dict(instrument=dict(psf=psf, lsf=lsf)),
        dict(instrument=dict(psf=images))]), 2)
    assert psf['alpha'] == dict(
        points=dict(value=[-500, 500], unit='km/s'), values=[1, 2],
        unit='arcsec')
    assert lsf['sigma'] == dict(file='lsf.ecsv', column='sigma')
    assert images == dict(type='images', file='psf.fits', points=dict(
        value=[5000, 7000], unit='Angstrom'))


def test_radii_of_mass_tables_are_migrated_to_a_quantity():
    table = dict(
        type='table', name='gas', radii=[1, 2], radius_unit='kpc',
        velocities=[50, 80])
    migration.migrate(dict(galaxies=[dict(mass=[table])]), 2)
    assert table == dict(
        type='table', name='gas', radii=dict(value=[1, 2], unit='kpc'),
        velocities=[50, 80])


def test_files_of_image_traits_are_migrated_to_fits_files():
    image = dict(type='image', file='hst.fits', hdu='SCI', size=[3, 2], step=1)
    migration.migrate(dict(galaxies=[dict(components=[dict(
        density=dict(ptraits=[image]))])]), 2)
    assert image == dict(
        type='image', file=dict(file='hst.fits', hdu='SCI'), size=[3, 2],
        step=1)


def test_scale_minimum_of_fitted_noise_is_migrated_to_scale_min():
    noise = dict(moment1=dict(scale=True, scale_minimum=0.5, jitter=True))
    migration.migrate(dict(observations=[dict(
        type='pixel_moments', likelihood=dict(
            type='gaussian', noise=noise))]), 2)
    assert noise == dict(moment1=dict(scale=True, scale_min=0.5, jitter=True))


def test_ranges_of_pdescs_and_solves_are_migrated_to_min_and_max():
    config = dict(
        pdescs=dict(a=dict(type='scalar', minimum=0, maximum=1)),
        params=dict(properties=dict(a=dict(solve=dict(
            type='gaussian', mean=1, sigma=0.5, minimum=0)))))
    migration.migrate(config, 2)
    assert config['pdescs']['a'] == dict(type='scalar', min=0, max=1)
    assert config['params']['properties']['a']['solve'] == dict(
        type='gaussian', mean=1, sigma=0.5, min=0)


def test_coverages_of_visibilities_are_migrated_to_files():
    observable = dict(type='visibilities', coverage='cov.npz', fov=12)
    migration.migrate(dict(observations=[dict(
        type='sky', observable=observable)]), 2)
    assert observable == dict(
        type='visibilities', coverage=dict(file='cov.npz'), fov=12)


def test_data_of_observations_are_migrated_to_their_type():
    data = dict(regions=dict(type='bins', file='bins.fits'))
    migration.migrate(dict(observations=[dict(
        type='sky', observable=dict(type='region_moments'), data=data)]), 2)
    assert data == dict(
        type='region_moments', regions=dict(type='bins', file='bins.fits'))


def test_parallel_evaluations_are_migrated_to_workers_of_threads():
    config = dict(fitter=dict(type='cmaes'), parallel=dict(drivers=['a']))
    migration.migrate(config, 2)
    assert list(config) == ['version', 'fitter', 'workers']
    assert config['workers'] == dict(type='threads', drivers=['a'])


def test_datasets_have_the_types_of_their_observables():
    # (on which the migration of the data of observations relies)
    for type_, cls in observable_parser.registered_classes().items():
        assert cls.dataset_class.type() == type_
