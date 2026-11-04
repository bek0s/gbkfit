"""
Tests for the migration of configurations (gbkfit.migration): the
loaders refuse configurations of other versions of the format, and the
migration rewrites those of older ones (files, with their comments, and
projects) and leaves the current ones as they are.
"""

import importlib
import json
import subprocess
import sys

import pytest
import ruamel.yaml

from gbkfit import migration, project, setup
from gbkfit.config import ConfigError
from gbkfit.observation import observable_parser


yaml = ruamel.yaml.YAML(typ='safe')

OBSERVATION = dict(
    type='sky', name='kmos', line_of_sight=dict(type='flat'),
    observable=dict(
        type='pixel_spectra', size=[4, 4],
        spectral=dict(size=5, convention='relativistic')))


@pytest.mark.parametrize('version, message', [
    (None, "has no version: 'version: 4'"),
    (3, "is of version 3 of the format, not 4"),
    (5, "is of version 5 of the format, not 4")])
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
    (dict(version=5), None, "versions of the format are 1 to 4")])
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
        "version: 4\n"
        "observations:\n"
        "- type: sky\n"
        "  name: kmos\n"
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
        type='disk', name='b', clouds=dict(cflux=1e-3, seed=2))


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
    # The rule of version 2 alone: that of version 4 removes geometry
    migration.v2.OwnGeometries().rewrite(
        dict(galaxies=[dict(components=components)]))
    assert [c.get('geometry') == 'own' for c in components] == owns


def test_components_of_version_2_keep_the_geometry_of_their_galaxy():
    components = [dict(type='disk', name='a'), dict(type='point', name='b')]
    migration.migrate(dict(galaxies=[dict(components=components)]), 2)
    assert all('geometry' not in c for c in components)


def test_observations_without_a_type_are_migrated_to_the_sky():
    observation = dict(observable=dict(type='pixel_brightness'))
    migration.migrate(dict(observations=[observation, 'named']), 1)
    assert list(observation) == ['type', 'observable']
    assert observation['type'] == 'sky'


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
            type='self_gravitating', mass='gas', direction='ver')),
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


def test_each_version_has_the_module_of_its_rules():
    # The rules of version N rewrite configurations of version N - 1, in
    # the order of the versions
    modules = [
        importlib.import_module(f'gbkfit.migration.v{version}')
        for version in range(2, project.FORMAT_VERSION + 1)]
    for version, module in enumerate(modules, 2):
        assert module.RULES
        assert {rule.version() for rule in module.RULES} == {version}
    assert migration.RULES == tuple(
        rule for module in modules for rule in module.RULES)


def test_a_mass_table_alone_is_migrated_too():
    # Only the tables of a list of components used to be migrated
    table = dict(type='table', radii=[1, 2], radius_unit='kpc')
    changes = migration.migrate(dict(galaxies=[dict(mass=table)]), 2)
    assert table == dict(type='table', radii=dict(value=[1, 2], unit='kpc'))
    assert ['galaxies', 0, 'mass'] in [change.path for change in changes]


def test_the_spectral_axes_of_migrated_configurations_are_left_alone():
    # A project whose migration stopped is migrated again from its old
    # version: the spatial step that stays used to replace the spectral axis
    observation = dict(type='sky', observable=dict(
        type='pixel_spectra', size=[4, 4, 5], step=0.5))
    config = dict(observations=[observation])
    for _ in range(2):
        for rule in migration.RULES:
            if rule.version() == 3:
                rule.rewrite(config)
    assert observation['observable'] == dict(
        type='pixel_spectra', size=[4, 4], step=0.5,
        spectral=dict(size=5, step=0.5, convention='relativistic'))


def test_a_file_that_cannot_be_written_is_left_as_it_was(
        tmp_path, monkeypatch):
    # It used to be emptied before it was written
    filename = tmp_path / 'config.yaml'
    text = "observations:\n- {type: sky, scale: 2}\n"
    filename.write_text(text)
    new_yaml = migration.core._yaml

    class Failing:
        def load(self, f):
            return new_yaml().load(f)

        def dump(self, config, f):
            f.write("observ")
            raise RuntimeError("the disk is full")

    monkeypatch.setattr(migration.core, '_yaml', Failing)
    with pytest.raises(RuntimeError, match="disk is full"):
        migration.migrate_file(str(filename), from_version=2)
    assert filename.read_text() == text
    assert [path.name for path in tmp_path.iterdir()] == ['config.yaml']


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


def test_rings_from_the_centre_lose_rnmin():
    disk = dict(type='disk', rnmin=0, rnmax=4, rnsep=1)
    migration.migrate(dict(galaxies=[dict(components=[disk])]), 2)
    assert disk == dict(type='disk', rnmax=4, rnsep=1)
    hole = dict(type='disk', rnmin=1, rnmax=4, rnsep=1)
    changes = migration.migrate(
        dict(galaxies=[dict(components=[hole])]), 2)
    assert any(c.message.startswith('by hand') for c in changes)


def test_datasets_have_the_types_of_their_observables():
    # (on which the migration of the data of observations relies)
    for type_, cls in observable_parser.registered_classes().items():
        assert cls.dataset_class.type() == type_


def test_gauss_hermite_moments_are_migrated_to_h3_and_h4():
    # (they were moment3 and moment4, the names of the central moments)
    data = dict(type='pixel_moments', moment1='v.fits', moment3='h3.fits',
                moment4='h4.fits')
    hermite = dict(observable=dict(type='pixel_moments',
                                   method='gauss_hermite'), data=data)
    central = dict(observable=dict(type='pixel_moments'),
                   data=dict(moment3='m3.fits'))
    migration.migrate(dict(observations=[hermite, central]), 2)
    assert data == dict(type='pixel_moments', moment1='v.fits',
                        h3='h3.fits', h4='h4.fits')
    assert central['data'] == dict(moment3='m3.fits')


# Two disks and a point of version 3 that share the geometry of their
# galaxy, flat (the point its centre and systemic velocity)
DISK = dict(
    type='disk', rnodes=[0, 1, 2], density=dict(ptraits=dict(
        type='exponential')),
    kinematics=dict(
        type='traits', vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform')),
    emission=dict(type='lines'))
SHARED = [
    DISK | dict(name='gas'), DISK | dict(name='ion'),
    dict(type='point', name='agn', emission=dict(type='lines'))]


def test_shared_geometries_are_migrated_to_ties():
    config = dict(
        galaxies=[dict(components=SHARED)],
        params=dict(properties=dict(
            xpos=0.5, ypos=0, vsys=10, posa=dict(value=30, min=0, max=90),
            incl=60, agn_flux='xpos * 2')))
    changes = migration.migrate(config, 3)
    components = config['galaxies'][0]['components']
    assert all('geometry' not in c for c in components)
    properties = config['params']['properties']
    assert properties == dict(
        gas_xpos=0.5, ion_xpos='gas_xpos', agn_xpos='gas_xpos',
        gas_ypos=0, ion_ypos='gas_ypos', agn_ypos='gas_ypos',
        gas_vsys=10, ion_vsys='gas_vsys', agn_vsys='gas_vsys',
        gas_posa=dict(value=30, min=0, max=90), ion_posa='gas_posa',
        gas_incl=60, ion_incl='gas_incl', agn_flux='gas_xpos * 2')
    assert migration.Change(['params', 'properties'], "xpos -> gas_xpos") \
        in changes
    # The migrated configuration has every parameter, of one free value
    from gbkfit.model import galaxy_parser
    from gbkfit.params import ParamSpace
    galaxy = galaxy_parser.load(config['galaxies'][0])
    properties |= {
        name: 1 for name in galaxy.pdescs() if name not in properties}
    space = ParamSpace.from_properties(galaxy.pdescs(), properties)
    assert space.free_properties() == dict(
        gas_posa=dict(value=30, min=0, max=90))


def test_a_shared_warped_geometry_gives_its_disks_its_warps_and_rings():
    geometry = dict(name='warp', tilted=True, rnmax=2, rnsep=1)
    disks = [
        {k: v for k, v in DISK.items() if k != 'rnodes'}
        | dict(name=name, geometry='warp') for name in ('gas', 'ion')]
    config = dict(
        galaxies=[dict(geometries=[geometry], components=disks)],
        params=dict(
            properties={'warp_posa': [30, 31, 32], 'warp_incl[1:]': 60},
            modes=dict(warp_posa=dict(type='offsets', origin=0)),
            priors=[dict(type='smooth', expr='warp_posa', order=2, std=1,
                         positions='warp_rnodes')]))
    migration.migrate(config, 3)
    galaxy = config['galaxies'][0]
    assert 'geometries' not in galaxy
    for disk in galaxy['components']:
        assert disk['tilted'] is True and 'geometry' not in disk
        assert (disk['rnmax'], disk['rnsep']) == (2, 1)
    params = config['params']
    assert params['properties'] == {
        'gas_posa': [30, 31, 32], 'ion_posa': 'gas_posa',
        'gas_incl[1:]': 60, 'ion_incl': 'gas_incl'} | {
        f'ion_{name}': f'gas_{name}' for name in ('xpos', 'ypos', 'vsys')}
    assert params['modes'] == dict(gas_posa=dict(type='offsets', origin=0))
    assert params['priors'][0]['expr'] == 'gas_posa'
    assert params['priors'][0]['positions'] == 'gas_rnodes'


def test_a_geometry_shares_only_the_parameters_it_was_given():
    geometry = dict(name='centre', params=['xpos', 'ypos'])
    config = dict(
        galaxies=[dict(geometries=[geometry], components=[
            component | dict(geometry='centre') for component in SHARED])],
        params=dict(properties=dict(centre_xpos=1, centre_ypos=2)))
    migration.migrate(config, 3)
    assert config['params']['properties'] == dict(
        gas_xpos=1, ion_xpos='gas_xpos', agn_xpos='gas_xpos',
        gas_ypos=2, ion_ypos='gas_ypos', agn_ypos='gas_ypos')


def test_shared_geometries_without_params_are_migrated_by_hand():
    # A galaxy of a project, whose params are other pieces; the geometry
    # of a galaxy of one component without a name changes nothing
    changes = migration.migrate(dict(galaxies=[dict(components=SHARED)]), 3)
    by_hand = [c.message for c in changes if c.message.startswith('by hand')]
    assert "by hand: rename the parameter posa to gas_posa, ion_posa: " \
        "gas_posa" in by_hand
    alone = dict(galaxies=[dict(components=[DISK])])
    assert migration.migrate(alone, 3) == [
        migration.Change(['version'], "3 -> 4")]


def test_own_geometries_are_migrated_to_their_components():
    disk = DISK | dict(name='gas', geometry='own', tilted=True)
    config = dict(galaxies=[dict(components=[disk])])
    migration.migrate(config, 3)
    assert disk == DISK | dict(name='gas', tilted=True)
