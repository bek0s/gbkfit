"""
The leaves of configurations that name files or other parts (see
gbkfit.config.config_leaves and gbkfit.setup.setup_leaves).
"""

from collections.abc import Sequence

from gbkfit.config import ConfigLeaf, DataFile, config_leaves, with_files
from gbkfit.model import Galaxy
from gbkfit.setup import setup_leaves


GALAXY = dict(
    components=[dict(
        type='thin_disk', name='gas', rnodes=dict(max=10, sep=2),
        rptraits=dict(type='exponential'),
        vptraits=dict(type='mass'), dptraits=dict(type='uniform'),
        emission=dict(type='lines', lines=['hi21']))],
    mass=[dict(
        type='traced_disk', name='gas_mass', component='gas',
        vertical=dict(type='sech2'))],
    distance=dict(type='parameter'))


def test_a_mass_disk_names_the_component_it_follows():
    leaves = list(config_leaves(GALAXY, Galaxy))
    assert leaves == [
        ConfigLeaf(('mass', 0, 'component'), 'components', 'gas')]


def test_data_files_are_found_in_every_form():
    files = ['a.fits', dict(file='b.fits', hdu=1), None]
    leaves = list(config_leaves(files, Sequence[DataFile | None]))
    assert leaves == [
        ConfigLeaf((0,), 'file', 'a.fits'),
        ConfigLeaf((1, 'file'), 'file', 'b.fits')]


def test_files_are_changed_in_a_copy():
    files = ['a.fits', dict(file='b.fits', hdu=1)]
    changed = with_files(files, Sequence[DataFile], lambda f: f'x/{f}')
    assert changed == ['x/a.fits', dict(file='x/b.fits', hdu=1)]
    assert files == ['a.fits', dict(file='b.fits', hdu=1)]
    assert with_files('a.fits', DataFile, str.upper) == 'A.FITS'


def test_the_leaves_of_a_setup_carry_their_section():
    config = dict(
        galaxies=GALAXY,
        observations=[dict(type='pixel_moments')],
        datasets=[dict(
            type='pixel_moments',
            moment0=dict(data='m0.fits', error='m0_e.fits'))],
        likelihoods=dict(
            maps=dict(type='user', file='noise.py', func='f')))
    leaves = list(setup_leaves(config))
    assert ConfigLeaf(
        ('galaxies', 'mass', 0, 'component'), 'components', 'gas') in leaves
    assert ConfigLeaf(
        ('datasets', 0, 'moment0', 'data'), 'file',
        'm0.fits') in leaves
    assert ConfigLeaf(
        ('likelihoods', 'maps', 'file'), 'file', 'noise.py') in leaves


def test_what_cannot_be_read_has_no_leaves():
    assert list(config_leaves(dict(type='nonsense'), Galaxy)) == []
    assert list(config_leaves(42, DataFile)) == []


def test_a_galaxy_is_described_alone_with_its_problems():
    from gbkfit.setup import describe_galaxy
    described = describe_galaxy(GALAXY)
    assert described.galaxy is not None and not described.problems
    broken = dict(GALAXY, mass=[dict(GALAXY['mass'][0], vertical=dict(
        type='nonsense'))])
    described = describe_galaxy(broken)
    assert described.galaxy is None
    assert [p.path for p in described.problems] == [['mass', 0, 'vertical']]
