"""
Tests for the migration of configurations (gbkfit.migration): the
loaders refuse configurations of other versions of the format, and the
migration rewrites those of older ones (files, with their comments, and
projects) and leaves the current ones as they are.
"""

import copy
import importlib
import json
import subprocess
import sys

import pytest
import ruamel.yaml

from gbkfit import migration, project, setup
from gbkfit.config import FORMAT_VERSION, ConfigError
from gbkfit.observation import observable_parser


yaml = ruamel.yaml.YAML(typ='safe')

# The current version of the format
VERSION = FORMAT_VERSION

OBSERVATION = dict(
    type='sky', name='kmos', line_of_sight=dict(type='flat'),
    observable=dict(
        type='pixel_spectra', size=[4, 4],
        spectral=dict(size=5, convention='relativistic')))


@pytest.mark.parametrize('version, message', [
    (None, f"has no version: 'version: {VERSION}'"),
    (3, f"is of version 3 of the format, not {VERSION}"),
    (VERSION + 1,
     f"is of version {VERSION + 1} of the format, not {VERSION}")])
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
    (dict(version=VERSION + 1), None,
     f"versions of the format are 1 to {VERSION}")])
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
        f"version: {VERSION}\n"
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
        FORMAT_VERSION


def test_a_project_of_the_current_version_is_left_as_it_is(tmp_path):
    study = project.create(str(tmp_path / 'study'))
    for folder in ('galaxies', 'observations', 'params', 'setups'):
        study.create_piece(folder, 'first')
    galaxy = study.piece('galaxies', 'first')
    assert migration.migrate_project(study.path) == {}
    assert study.piece('galaxies', 'first') == galaxy


def test_the_old_disk_types_are_migrated():
    thick = dict(ptraits=dict(type='exponential'), htraits=dict(type='one'))
    components = [
        dict(type='smdisk', name='a'),
        dict(type='mcdisk', name='b', cflux=1e-3, seed=2, density=thick),
        dict(type='smcone', name='c')]
    migration.migrate(dict(galaxies=[dict(components=components)]), 1)
    assert [c['type'] for c in components] == [
        'thin_disk', 'thick_disk', 'cone']
    assert components[1] == dict(
        type='thick_disk', name='b', clouds=dict(cflux=1e-3, seed=2),
        rptraits=thick['ptraits'], rhtraits=thick['htraits'])


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
    # (and then on the components: see test_components_are_flat)
    assert components == [
        dict(type='thin_disk', vptraits=dict(type='mass'), equilibrium=dict(
            type='self_gravitating', mass='gas', direction='ver')),
        dict(type='cone', vptraits=dict(type='uniform')),
        dict(type='jeans_spheroid')]


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
        for version in range(2, FORMAT_VERSION + 1)]
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
    assert disk == dict(type='thin_disk', rnodes=dict(max=4, sep=1))
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
# DISK migrated to the current version: its traits on the disk
FLAT_DISK = dict(
    rnodes=[0, 1, 2], rptraits=dict(type='exponential'),
    vptraits=dict(type='arctan', direction='tan'),
    dptraits=dict(type='uniform'), emission=dict(type='lines'))
SHARED = [
    DISK | dict(name='gas'), DISK | dict(name='ion'),
    dict(type='point', name='agn', emission=dict(type='lines'))]


def test_shared_geometries_are_migrated_to_ties():
    config = dict(
        galaxies=[dict(components=copy.deepcopy(SHARED))],
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
        assert disk['rnodes'] == dict(max=2, sep=1)
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
    changes = migration.migrate(
        dict(galaxies=[dict(components=copy.deepcopy(SHARED))]), 3)
    by_hand = [c.message for c in changes if c.message.startswith('by hand')]
    assert "by hand: rename the parameter posa to gas_posa, ion_posa: " \
        "gas_posa" in by_hand
    alone = dict(galaxies=[dict(components=[copy.deepcopy(DISK)])])
    path = ['galaxies', 0, 'components', 0]
    assert migration.migrate(alone, 3) == [
        migration.Change(path, "type disk -> thin_disk"),
        migration.Change(
            path, "density: {ptraits, htraits} -> rptraits, rhtraits"),
        migration.Change(
            path, "kinematics: {type: traits, ...} -> its options on the "
                  "component"),
        migration.Change(['version'], f"3 -> {VERSION}")]


def test_own_geometries_are_migrated_to_their_components():
    disk = copy.deepcopy(DISK) | dict(name='gas', geometry='own', tilted=True)
    config = dict(galaxies=[dict(components=[disk])])
    migration.migrate(config, 3)
    assert disk == FLAT_DISK | dict(
        type='thin_disk', name='gas', tilted=True)


@pytest.mark.parametrize('old, new', [
    (dict(rnmax=10, rnsep=2), dict(max=10, sep=2)),
    (dict(rnmax=10, rnlen=6), dict(max=10, len=6))])
def test_rings_evenly_spaced_are_migrated_to_one_option(old, new):
    disk = dict(type='disk', name='gas', **old, rstep=0.5)
    spheroid = dict(type='spheroid', kinematics=dict(type='traits', **old))
    config = dict(galaxies=[dict(components=[disk, spheroid])])
    migration.migrate(config, 3)
    assert disk == dict(type='thin_disk', name='gas', rnodes=new, rstep=0.5)
    assert list(disk) == ['type', 'name', 'rnodes', 'rstep']
    assert spheroid == dict(type='spheroid', rnodes=new)


def test_rings_of_several_forms_are_migrated_by_hand():
    disk = dict(type='disk', rnmax=10, rnsep=2, rnodes=[0, 1])
    changes = migration.migrate(dict(galaxies=[dict(components=[disk])]), 3)
    assert any(c.message.startswith('by hand: give rnodes alone')
               for c in changes)


def test_the_rings_of_a_shared_geometry_are_migrated_to_one_option():
    geometry = dict(name='warp', tilted=True, rnmax=2, rnsep=1)
    disk = {k: v for k, v in DISK.items() if k != 'rnodes'} \
        | dict(name='gas', geometry='warp')
    config = dict(galaxies=[dict(geometries=[geometry], components=[disk])])
    migration.migrate(config, 3)
    assert disk['rnodes'] == dict(max=2, sep=1) and 'rnmax' not in disk


def test_a_project_of_one_galaxy_has_its_params_renamed(tmp_path):
    study = project.create(str(tmp_path / 'study'))
    yaml.dump(dict(format=project.FORMAT, version=3),
              tmp_path / 'study' / project.MARKER)
    yaml.dump(dict(components=[DISK | dict(name='gas')]),
              tmp_path / 'study' / 'galaxies' / 'gas.yaml')
    yaml.dump(dict(properties=dict(posa=30)),
              tmp_path / 'study' / 'params' / 'start.yaml')
    migration.migrate_project(study.path)
    assert study.piece('params', 'start') == dict(
        properties=dict(gas_posa=30))


def test_the_galaxies_of_a_project_have_their_params_renamed_by_hand(
        tmp_path):
    # Which params are of which galaxy is in the setups
    study = project.create(str(tmp_path / 'study'))
    yaml.dump(dict(format=project.FORMAT, version=3),
              tmp_path / 'study' / project.MARKER)
    for name in ('gas', 'ion'):
        yaml.dump(dict(components=[DISK | dict(name=name)]),
                  tmp_path / 'study' / 'galaxies' / f'{name}.yaml')
    yaml.dump(dict(properties=dict(posa=30)),
              tmp_path / 'study' / 'params' / 'start.yaml')
    changes = migration.migrate_project(study.path)
    assert study.piece('params', 'start') == dict(properties=dict(posa=30))
    assert migration.Change([], (
        "by hand: rename the parameter posa to gas_posa")) in \
        changes['galaxies/gas.yaml']


THIN_DENSITY = dict(ptraits=dict(type='exponential'))
THICK_DENSITY = THIN_DENSITY | dict(htraits=dict(type='sech2', zmax=8))


@pytest.mark.parametrize('density, new', [
    (THIN_DENSITY, 'thin_disk'),
    (THIN_DENSITY | dict(htraits=None), 'thin_disk'),
    (THICK_DENSITY, 'thick_disk')])
def test_disks_are_thin_or_thick_by_their_type(density, new):
    disk = dict(type='disk', name='gas', density=copy.deepcopy(density))
    mass = dict(type='disk', name='gas', surface=dict(type='exponential'))
    config = dict(galaxies=[dict(components=[disk], mass=[mass])])
    changes = migration.migrate(config, 4)
    expected = THICK_DENSITY if new == 'thick_disk' else THIN_DENSITY
    assert disk == dict(type=new, name='gas') | {
        f'r{key}': value for key, value in expected.items()}
    # (the mass disk is of another family)
    assert mass['type'] == 'disk'
    assert not any(c.message.startswith('by hand') for c in changes)


def test_a_thin_disk_loses_the_options_of_a_thick_disk_that_are_none():
    disk = dict(
        type='disk', density=copy.deepcopy(THIN_DENSITY), zptraits=[],
        opacity=None, kinematics=dict(
            type='traits', vptraits=dict(type='uniform', direction='tan'),
            dptraits=dict(type='uniform'), vhtraits=None, equilibrium=None))
    config = dict(galaxies=[dict(components=[disk])])
    changes = migration.migrate(config, 4)
    assert disk == dict(
        type='thin_disk', rptraits=THIN_DENSITY['ptraits'],
        vptraits=dict(type='uniform', direction='tan'),
        dptraits=dict(type='uniform'))
    assert not any(c.message.startswith('by hand') for c in changes)
    from gbkfit.model import galaxy_parser
    galaxy_parser.load(config['galaxies'][0] | dict(components=[
        disk | dict(rnodes=[0, 1, 2], emission=dict(type='lines'))]))


def test_the_options_of_a_thick_disk_on_a_thin_one_are_changed_by_hand():
    # (they were errors: a disk with them needed density height traits)
    disk = dict(
        type='disk', density=copy.deepcopy(THIN_DENSITY),
        opacity=dict(type='dust'), clouds=dict(cflux=1e-3),
        kinematics=dict(
            type='traits', vptraits=dict(type='uniform', direction='tan'),
            dptraits=dict(type='uniform'),
            equilibrium=dict(type='central')))
    changes = migration.migrate(
        dict(galaxies=[dict(name='a', components=[disk])]), 4)
    assert disk['type'] == 'thin_disk'
    by_hand = [
        (c.path, c.message.split(':')[1]) for c in changes
        if c.message.startswith('by hand')]
    path = ['galaxies', 0, 'components', 0]
    assert by_hand == [
        (path, ' a thin disk takes no opacity'),
        (path, ' a thin disk takes no clouds'),
        (path + ['kinematics'], ' a thin disk takes no equilibrium')]
    # (left as they were, and then on the disk)
    assert disk['opacity'] == dict(type='dust')
    assert disk['equilibrium'] == dict(type='central')


def test_the_hdu_of_the_file_of_a_table_is_in_the_file():
    alpha = dict(file='psf.fits', hdu=1, points='wavelength', column='alpha')
    beta = dict(file='psf.ecsv', points='wavelength', column='beta')
    psf = dict(type='moffat', alpha=alpha, beta=beta)
    config = dict(observations=[dict(instrument=dict(psf=psf))])
    changes = migration.migrate(config, 5)
    assert alpha == dict(
        file=dict(file='psf.fits', hdu=1), points='wavelength',
        column='alpha')
    # (a table without an HDU, or with none, keeps its file as it is)
    assert beta == dict(file='psf.ecsv', points='wavelength', column='beta')
    assert [c.path for c in changes if c.path != ['version']] == [
        ['observations', 0, 'instrument', 'psf', 'alpha']]


def test_the_widths_of_lsfs_get_their_unit():
    # Numbers of km/s of the light, and inline tables without a unit; the
    # Hanning smoothing is of the channels of the data, of every LSF of an
    # instrument, a sum, and the noise
    table = dict(points=dict(value=[0, 1], unit='km/s'), values=[10, 20])
    filed = dict(file='lsf.ecsv', points='wavelength', column='sigma')
    lsf = dict(type='sum', weights=[1, 1], lsfs=[
        dict(type='gauss', sigma=20), dict(type='moffat', alpha=table,
                                          beta=2.5)])
    noise = dict(type='gauss', spectral_correlation=dict(
        type='convolution', lsfs=[
            dict(type='hanning', width=10), dict(type='gauss', sigma=filed)]))
    config = dict(
        observations=[dict(instrument=dict(lsf=lsf))], mock=dict(noise=noise))
    changes = migration.migrate(config, 6)
    assert lsf['lsfs'][0]['sigma'] == '20 km/s'
    assert table['unit'] == 'km/s'
    hanning, gauss = noise['spectral_correlation']['lsfs']
    assert hanning == dict(type='hanning')
    # (the unit of the column of a file is not known: by hand, if none;
    # one given in units is that of the column)
    assert gauss['sigma'] == filed and 'unit' not in filed
    with_units = dict(type='gauss', sigma=filed | dict(units=dict(
        sigma='km/s')))
    assert migration.migrate(dict(observations=[dict(instrument=dict(
        lsf=with_units))]), 6) == migration.migrate(dict(), 6)
    assert [c.path[-1] for c in changes if c.path != ['version']] == [
        'sigma', 'alpha', 'width', 'sigma']


def test_the_lsfs_of_a_project_are_migrated(tmp_path):
    # (and so are its pieces of datasets, PSFs and noise, which earlier
    # migrations left as they were)
    study = project.create(str(tmp_path / 'study'))
    yaml.dump(
        dict(format=project.FORMAT, version=6),
        tmp_path / 'study' / project.MARKER)
    yaml.dump(
        dict(type='gauss', sigma=30), tmp_path / 'study' / 'lsfs' / 'r.yaml')
    changes = migration.migrate_project(study.path)
    assert list(changes) == ['lsfs/r.yaml']
    assert study.piece('lsf', 'r') == dict(
        type='gauss', sigma='30 km/s')


def test_gaussian_beams_are_gaussians():
    # Of the same options, by their beam or a file; in instruments, sums,
    # and the noise
    beam = dict(type='gauss_beam', bmaj=6, bmin=3, bpa=30)
    psf = dict(type='sum', weights=[1, 1], psfs=[
        dict(type='gauss_beam', file='cube.fits'), dict(type='moffat',
                                                       alpha=1, beta=2)])
    config = dict(observations=[
        dict(instrument=dict(psf=beam)), dict(instrument=dict(psf=psf))],
        mock=dict(noise=dict(type='gauss', spatial_correlation=dict(
            type='gauss_beam', bmaj=4, bmin=4))))
    changes = migration.migrate(config, 6)
    assert beam == dict(type='gauss', bmaj=6, bmin=3, bpa=30)
    assert psf['psfs'][0] == dict(type='gauss', file='cube.fits')
    assert config['mock']['noise']['spatial_correlation']['type'] == 'gauss'
    assert len([c for c in changes if c.message == "gauss_beam -> gauss"]) \
        == 3


def test_measured_lsfs_are_profiles():
    # Of the same options (and their steps get their unit); in
    # instruments, sums, and the noise, but not the PSFs of type image
    one = dict(type='image', file='lsf.fits', step=5)
    lsf = dict(type='sum', weights=[1, 1], lsfs=[
        dict(type='images', file='lsfs.fits'), dict(type='gauss', sigma=20)])
    psf = dict(type='image', file='psf.fits')
    config = dict(observations=[
        dict(instrument=dict(lsf=one, psf=psf)),
        dict(instrument=dict(lsf=lsf))],
        mock=dict(noise=dict(type='gauss', spectral_correlation=dict(
            type='image', file='noise.fits'))))
    changes = migration.migrate(config, 6)
    assert one == dict(type='profile', file='lsf.fits', step='5 km/s')
    assert lsf['lsfs'][0] == dict(type='profiles', file='lsfs.fits')
    assert config['mock']['noise']['spectral_correlation']['type'] \
        == 'profile'
    assert psf['type'] == 'image'
    assert [c.message for c in changes if c.path[-1] == 'type'] == [
        'image -> profile', 'images -> profiles', 'image -> profile']


def test_tables_of_pieces_of_projects_have_their_hdu_in_their_file(tmp_path):
    # (a table of a piece of an LSF of a project of version 5)
    study = project.create(str(tmp_path / 'study'))
    yaml.dump(
        dict(format=project.FORMAT, version=5),
        tmp_path / 'study' / project.MARKER)
    yaml.dump(
        dict(type='gauss', sigma=dict(
            file='lsf.fits', hdu=1, points='wavelength', column='sigma',
            units=dict(sigma='km/s'))),
        tmp_path / 'study' / 'lsfs' / 'muse.yaml')
    migration.migrate_project(study.path)
    assert study.piece('lsf', 'muse')['sigma']['file'] == dict(
        file='lsf.fits', hdu=1)


def test_components_are_flat():
    # Version 8: the groups of options without a choice of kind go, and a
    # spheroid's type says how its stars move
    thick = dict(
        type='thick_disk', name='gas',
        density=dict(
            ptraits=dict(type='exponential'),
            htraits=dict(type='sech2', zmax=8)),
        kinematics=dict(
            type='traits', vptraits=dict(type='arctan', direction='tan'),
            dptraits=dict(type='uniform'), equilibrium=dict(type='jeans')),
        emission=dict(type='lines'))
    spheroids = [
        dict(type='spheroid', name='a', kinematics=dict(type='jeans')),
        dict(type='spheroid', name='b', kinematics=dict(
            type='schwarzschild', norbits=200)),
        dict(type='spheroid', name='c', kinematics=dict(
            type='traits', vptraits=dict(type='uniform', direction='tan'),
            dptraits=dict(type='uniform'), rnodes=[0, 10])),
        dict(type='spheroid', name='d'),
        dict(type='spheroid', name='e', kinematics=None)]
    config = dict(galaxies=[dict(components=[thick, *spheroids])])
    changes = migration.migrate(config, 7)
    # (in the order of the options they replace)
    assert list(thick) == [
        'type', 'name', 'rptraits', 'rhtraits', 'vptraits', 'dptraits',
        'equilibrium', 'emission']
    assert thick['rhtraits'] == dict(type='sech2', zmax=8)
    assert spheroids == [
        dict(type='jeans_spheroid', name='a'),
        dict(type='orbit_spheroid', name='b', norbits=200),
        dict(type='spheroid', name='c',
             vptraits=dict(type='uniform', direction='tan'),
             dptraits=dict(type='uniform'), rnodes=[0, 10]),
        dict(type='spheroid', name='d'),
        dict(type='spheroid', name='e')]
    assert not any(c.message.startswith('by hand') for c in changes)


def test_flat_components_keep_their_comments(tmp_path):
    path = tmp_path / 'galaxy.yaml'
    path.write_text(
        'version: 7\n'
        'galaxies:\n'
        '- components:\n'
        '  - type: thin_disk  # the gas\n'
        '    density:\n'
        '      ptraits:\n'
        '        type: exponential\n'
        '    emission:  # its light\n'
        '      type: band\n')
    migration.migrate_file(str(path))
    text = path.read_text()
    assert 'type: thin_disk  # the gas' in text
    assert 'rptraits:\n' in text and 'density' not in text
    assert 'emission:  # its light' in text


def test_trait_parameters_are_named_after_their_kind():
    # Version 8: pt and ht are rpt and rht; unnamed traits are numbered
    # from the second (pt1 -> rpt2); named traits keep their kind (rot ->
    # vpt_rot); the coefficients of line opacity are numbered so too
    disk = dict(
        type='thick_disk', name='gas',
        density=dict(
            ptraits=[dict(type='exponential'), dict(type='nw_uniform')],
            htraits=[dict(type='sech2'), dict(type='sech2')]),
        kinematics=dict(
            type='traits',
            vptraits=[dict(type='arctan', name='rot'),
                      dict(type='uniform', name='inflow')],
            dptraits=dict(type='uniform')),
        opacity=dict(type='line', coefficients=[
            dict(type='uniform'), dict(type='exponential')]))
    params = dict(
        properties=dict(
            gas_pt_a=1, gas_pt1_a=[1, 2], gas_ht_s=1,
            gas_ht1_s='gas_ht_s * 2', gas_rot_vt=100, gas_inflow_vr=3,
            gas_dpt_a=10, gas_op_a=1, gas_op1_a=0.1),
        priors=dict(x='gas_pt1_a[0] + gas_rot_vt'))
    config = dict(galaxies=[dict(components=[disk])], params=params)
    changes = migration.migrate(config, 7)
    assert params == dict(
        properties=dict(
            gas_rpt_a=1, gas_rpt2_a=[1, 2], gas_rht_s=1,
            gas_rht2_s='gas_rht_s * 2', gas_vpt_rot_vt=100,
            gas_vpt_inflow_vr=3, gas_dpt_a=10, gas_op_a=1, gas_op2_a=0.1),
        priors=dict(x='gas_rpt2_a[0] + gas_vpt_rot_vt'))
    assert list(params['properties'])[:2] == ['gas_rpt_a', 'gas_rpt2_a']
    assert not any(c.message.startswith('by hand') for c in changes)


def test_trait_parameters_of_the_galaxies_of_a_project_are_renamed():
    # (their params are in the setups: renamed alike, unless two galaxies
    # rename a prefix in different ways)
    def galaxy(*rptraits):
        return dict(components=[dict(
            type='thin_disk', name='disk',
            density=dict(ptraits=list(rptraits)))])
    one, two = dict(type='uniform'), dict(type='uniform', name='ring')
    params = dict(properties=dict(disk_pt_a=1, disk_ring_a=2))
    config = dict(galaxies=[galaxy(one), galaxy(one, one)], params=params)
    changes = migration.migrate(config, 7)
    assert params == dict(properties=dict(disk_rpt_a=1, disk_ring_a=2))
    assert not any(c.message.startswith('by hand') for c in changes)
    params = dict(properties=dict(disk_pt_a=1, disk_ring_a=2))
    config = dict(galaxies=[galaxy(one), galaxy(two)], params=params)
    changes = migration.migrate(config, 7)
    assert params == dict(properties=dict(disk_rpt_a=1, disk_rpt_ring_a=2))
    # (a conflict: the same old prefix renamed in two ways, here by a
    # trait named pt)
    tied = dict(type='uniform', name='pt')
    params = dict(properties=dict(disk_pt_a=1))
    config = dict(galaxies=[galaxy(one), galaxy(tied)], params=params)
    changes = migration.migrate(config, 7)
    assert params == dict(properties=dict(disk_pt_a=1))
    assert any(
        c.message.startswith('by hand: the galaxies rename the parameters '
                             'disk_pt_*')
        for c in changes)


def test_mass_that_traces_a_component_has_its_own_types():
    # Version 8: a mass disk of the surface of a component is a
    # traced_disk, in its layer unless another is given; light_spheroid
    # is traced_spheroid; a disk of a surface law is left as it is
    mass = [
        dict(type='disk', name='gas',
             surface=dict(type='component', component='hi'),
             vertical=dict(type='component', component='hi')),
        dict(type='disk', name='h2',
             surface=dict(type='component', component='co', rstep=0.5),
             vertical=dict(type='component', component='hi')),
        dict(type='disk', name='dust',
             surface=dict(type='component', component='dust'),
             vertical=dict(type='sech2')),
        dict(type='disk', name='hi',
             surface=dict(type='component', component='hi', rstep=0.5),
             vertical=dict(type='component', component='hi')),
        dict(type='light_spheroid', name='lum', component='bulge'),
        dict(type='disk', name='stars',
             surface=dict(type='exponential'), vertical=dict(type='thin'))]
    changes = migration.migrate(dict(galaxies=[dict(mass=mass)]), 7)
    assert mass == [
        dict(type='traced_disk', name='gas', component='hi'),
        dict(type='traced_disk', name='h2', component='co', rstep=0.5,
             vertical=dict(type='component', component='hi')),
        dict(type='traced_disk', name='dust', component='dust',
             vertical=dict(type='sech2')),
        dict(type='traced_disk', name='hi', component='hi', rstep=0.5),
        dict(type='traced_spheroid', name='lum', component='bulge'),
        dict(type='disk', name='stars',
             surface=dict(type='exponential'), vertical=dict(type='thin'))]
    assert not any(c.message.startswith('by hand') for c in changes)
    from gbkfit.model import mass_component_parser
    mass_component_parser.load(mass)


def test_halos_are_renamed():
    # Version 8: nfw_truncated is truncated_nfw; the inner slope of
    # dekel_zhao is gamma, the radius of a dark fraction rfdm, and the age
    # of an SIDM halo age, in the params sections too
    mass = [
        dict(type='spheroid', name='a', profile=dict(type='nfw_truncated')),
        dict(type='spheroid', name='b', profile=dict(type='dekel_zhao'),
             baryons=['a'], dark_fraction=dict(solve='log_m',
                                               bounds=[9, 15])),
        dict(type='spheroid', name='c', profile=dict(type='nfw'),
             baryons=['a'], response=dict(type='sidm')),
        dict(type='spheroid', name='d', profile=dict(type='nfw'),
             baryons=['a'])]
    params = dict(
        properties=dict(
            mass_b_a=1.0, mass_b_r_fdm=5, mass_c_t_age=10,
            mass_b_alpha=2.0),
        derived=dict(x='mass_b_a * 2 + mass_c_t_age'))
    config = dict(galaxies=[dict(mass=mass)], params=params)
    migration.migrate(config, 7)
    assert mass[0] == dict(
        type='halo', name='a', profile=dict(type='truncated_nfw'))
    # (baryons that nothing used)
    assert 'baryons' not in mass[3] and mass[2]['baryons'] == ['a']
    assert params == dict(
        properties=dict(
            mass_b_gamma=1.0, mass_b_rfdm=5, mass_c_age=10,
            mass_b_alpha=2.0),
        derived=dict(x='mass_b_gamma * 2 + mass_c_age'))


def test_spheroids_of_dark_matter_are_halos():
    # Version 8: a mass spheroid of the profile of a halo, or that
    # responds to baryons or is given by a dark fraction, is a halo, of the
    # parameterisation and the overdensity of its profile; its response
    # is its contraction, or sidm; a dark fraction of its mass or its
    # scale density is its parameterisation; gnedin04 has the parameters
    # a0 and w, at its values before
    mass = [
        dict(type='disk', name='stars', surface=dict(type='exponential'),
             vertical=dict(type='thin')),
        dict(type='spheroid', name='bulge', profile=dict(type='sersic')),
        dict(type='spheroid', name='a',
             profile=dict(type='nfw', parameters='velocity_concentration',
                          overdensity='200m'),
             baryons=['stars'], response=dict(type='gnedin04')),
        dict(type='spheroid', name='b', profile=dict(type='burkert'),
             baryons=['stars'], response=dict(
                 type='sidm', contraction=dict(type='blumenthal86')),
             dark_fraction=dict(solve='v', bounds=[10, 300])),
        dict(type='spheroid', name='c',
             profile=dict(type='einasto', parameters='density_radius'),
             baryons=['stars'],
             dark_fraction=dict(solve='log_rho', bounds=[5, 9]))]
    params = dict(properties=dict(
        mass_a_v=150, mass_a_c=10, mass_b_c=8, mass_b_fdm=0.5,
        mass_b_r_fdm=5, mass_b_sigma_m=1, mass_b_t_age=10))
    config = dict(galaxies=[dict(mass=mass)], params=params)
    changes = migration.migrate(config, 7)
    assert mass[1] == dict(
        type='spheroid', name='bulge', profile=dict(type='sersic'))
    assert mass[2] == dict(
        type='halo', name='a',
        profile=dict(type='nfw', parameterisation='velocity_concentration'),
        overdensity='200m', baryons=['stars'],
        contraction=dict(type='gnedin04'))
    assert mass[3] == dict(
        type='halo', name='b',
        profile=dict(type='burkert',
                     parameterisation='dark_fraction_concentration'),
        baryons=['stars'], contraction=dict(type='blumenthal86'), sidm=True)
    assert mass[4] == dict(
        type='halo', name='c',
        profile=dict(type='einasto', parameterisation='dark_fraction_radius'),
        baryons=['stars'])
    assert list(params['properties']) == [
        'mass_a_v', 'mass_a_c', 'mass_a_a0', 'mass_a_w', 'mass_b_c',
        'mass_b_fdm', 'mass_b_rfdm', 'mass_b_sigma_m', 'mass_b_age']
    assert params['properties']['mass_a_a0'] == 0.85
    assert params['properties']['mass_a_w'] == 0.8
    assert not any(c.message.startswith('by hand') for c in changes)
    from gbkfit.model import mass_component_parser
    mass_component_parser.load(mass)
    # (a dark fraction of another parameter, or a luminous spheroid that
    # responds: by hand)
    mass = [
        dict(type='spheroid', name='a', profile=dict(type='einasto'),
             baryons=['b'], dark_fraction=dict(solve='alpha', bounds=[0, 1])),
        dict(type='spheroid', name='b', profile=dict(type='sersic'),
             baryons=['a'], response=dict(type='blumenthal86'))]
    changes = migration.migrate(dict(galaxies=[dict(mass=mass)]), 7)
    assert [c.message.split(':')[0] for c in changes
            if c.message.startswith('by hand')] == ['by hand', 'by hand']
    assert mass[0]['type'] == 'halo' and mass[1]['type'] == 'spheroid'


def test_the_setups_of_a_project_rename_the_parameters_of_its_pieces(
        tmp_path):
    # A setup that names a galaxy piece, with params of its own; and one
    # with a galaxy of its own, whose params are a piece: both renamed
    study = project.create(str(tmp_path / 'study'))
    yaml.dump(dict(format=project.FORMAT, version=7),
              tmp_path / 'study' / project.MARKER)
    disk = dict(type='thin_disk', name='gas', density=dict(ptraits=[
        dict(type='exponential'), dict(type='uniform')]))
    yaml.dump(dict(components=[disk]),
              tmp_path / 'study' / 'galaxies' / 'gas.yaml')
    yaml.dump(dict(properties=dict(ion_pt_a=1, ion_pt1_a=2)),
              tmp_path / 'study' / 'params' / 'ion.yaml')
    yaml.dump(dict(galaxies='gas', params=dict(properties=dict(
        gas_pt_a=1, gas_pt1_a=2))),
              tmp_path / 'study' / 'setups' / 'named.yaml')
    yaml.dump(dict(galaxies=[dict(components=[disk | dict(name='ion')])],
                   params='ion'),
              tmp_path / 'study' / 'setups' / 'inline.yaml')
    migration.migrate_project(study.path)
    named = yaml.load(tmp_path / 'study' / 'setups' / 'named.yaml')
    assert named == dict(galaxies='gas', params=dict(properties=dict(
        gas_rpt_a=1, gas_rpt2_a=2)))
    assert study.piece('params', 'ion') == dict(
        properties=dict(ion_rpt_a=1, ion_rpt2_a=2))
    inline = yaml.load(tmp_path / 'study' / 'setups' / 'inline.yaml')
    assert inline['galaxies'][0]['components'][0]['rptraits'] == [
        dict(type='exponential'), dict(type='uniform')]
    assert study.piece('galaxies', 'gas')['components'][0]['rptraits'] \
        == [dict(type='exponential'), dict(type='uniform')]


def test_the_cosmology_of_distances_is_that_of_the_configuration():
    # Version 8: the cosmology of the distances of the galaxies is that
    # of the configuration (Planck18 by default); direct is parameter,
    # redshift cosmological, and a redshift of its own a fixed parameter
    # of its angular diameter distance
    import astropy.cosmology
    galaxies = [
        dict(name='a', mass=[dict(type='point', name='bh')],
             distance=dict(type='direct', cosmology='WMAP9')),
        dict(name='b', redshift=0.1, mass=[dict(type='point', name='bh')],
             distance=dict(type='redshift', cosmology='WMAP9')),
        dict(name='c', mass=[dict(type='point', name='bh')],
             distance=dict(type='redshift', redshift=0.02,
                           cosmology='WMAP9'))]
    params = dict(properties=dict(
        a_distance=10, a_mass_bh_log_m=8, b_mass_bh_log_m=8,
        c_mass_bh_log_m=8))
    config = dict(galaxies=galaxies, params=params)
    changes = migration.migrate(config, 7)
    assert [g['distance'] for g in galaxies] == [
        dict(type='parameter'), dict(type='cosmological'),
        dict(type='parameter')]
    assert config['cosmology'] == dict(type='published', name='WMAP9')
    distance = astropy.cosmology.WMAP9.angular_diameter_distance(0.02)
    assert list(params['properties']) == [
        'a_distance', 'a_mass_bh_log_m', 'b_mass_bh_log_m',
        'c_mass_bh_log_m', 'c_distance']
    assert params['properties']['c_distance'] == pytest.approx(
        distance.to_value('Mpc'), rel=1e-6)
    assert not any(c.message.startswith('by hand') for c in changes)
    # (Planck18, the default: no section)
    config = dict(galaxies=[dict(
        mass=[dict(type='point', name='bh')],
        distance=dict(type='direct', cosmology='Planck18'))])
    migration.migrate(config, 7)
    assert 'cosmology' not in config
    # (galaxies of different cosmologies: refused)
    galaxies = [
        dict(name='a', distance=dict(type='direct', cosmology='WMAP9')),
        dict(name='b', distance=dict(type='direct'))]
    with pytest.raises(ConfigError, match="different cosmologies"):
        migration.migrate(dict(galaxies=galaxies), 7)


def test_the_setups_of_a_project_take_the_cosmology_of_their_galaxies(
        tmp_path):
    # Of the galaxy pieces they name; a setup whose galaxies have
    # different cosmologies is refused, and the project is left as it was
    study = project.create(str(tmp_path / 'study'))
    yaml.dump(dict(format=project.FORMAT, version=7),
              tmp_path / 'study' / project.MARKER)
    for name, cosmology in (('far', 'WMAP9'), ('near', 'Planck18')):
        yaml.dump(dict(name=name, mass=[dict(type='point', name='bh')],
                       distance=dict(type='direct', cosmology=cosmology)),
                  tmp_path / 'study' / 'galaxies' / f'{name}.yaml')
    yaml.dump(dict(galaxies='far', params=dict(properties=dict(
        far_distance=10, far_mass_bh_log_m=8))),
              tmp_path / 'study' / 'setups' / 'one.yaml')
    yaml.dump(dict(galaxies=['far', 'near'], params=dict(properties={})),
              tmp_path / 'study' / 'setups' / 'both.yaml')
    files = sorted((tmp_path / 'study').rglob('*.yaml'))
    before = {file: file.read_text() for file in files}
    with pytest.raises(ConfigError, match="different cosmologies"):
        migration.migrate_project(study.path)
    assert {file: file.read_text() for file in files} == before
    (tmp_path / 'study' / 'setups' / 'both.yaml').unlink()
    migration.migrate_project(study.path)
    one = yaml.load(tmp_path / 'study' / 'setups' / 'one.yaml')
    assert one['cosmology'] == dict(type='published', name='WMAP9')
    assert project.open(study.path).piece('galaxies', 'far')['distance'] \
        == dict(type='parameter')


def test_the_distances_of_tables_and_profiles_are_assumed_distances():
    # Version 8: the distance D0 that their radii were measured at is
    # their assumed_distance: of mass tables, profile observations and
    # their data, and data pieces of profiles
    table = dict(type='table', name='disk', file='rotmod.dat',
                 radius=dict(column='col1', unit='kpc'), distance=10,
                 velocity=dict(column='col5'))
    profile = dict(type='profile', name='rc', distance=10, data=dict(
        file='rc.ecsv', radius=dict(column='radius'), distance=10,
        vcirc=dict(column='vcirc')))
    config = dict(galaxies=[dict(mass=[table])], observations=[profile])
    migration.migrate(config, 7)
    assert 'distance' not in table and table['assumed_distance'] == 10
    assert profile['assumed_distance'] == 10
    assert profile['data']['assumed_distance'] == 10
    pieces = migration.rules.PiecesView(data=[
        dict(type='profile', file='rc.ecsv', distance=10),
        dict(type='pixel_spectra', data='cube.fits')])
    migration.v8.AssumedDistances().rewrite(pieces)
    assert pieces['data'] == [
        dict(type='profile', file='rc.ecsv', assumed_distance=10),
        dict(type='pixel_spectra', data='cube.fits')]
