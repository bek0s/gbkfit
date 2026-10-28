"""
Tests of the light of disks from observed images (see traits.RPTraitImage):
the face-on light that their galaxy makes from the image at the geometry
of the disk.
"""

import logging

import numpy as np
import pytest

from gbkfit.model import galaxy_parser
from gbkfit.observation import ObservationGroup, observation_parser
from gbkfit.utils import fitsutils
from gbkfit.utils.parseutils import ConfigError


# An image of 64 x 64 pixels of 0.2 arcsec, at RA 150 and Dec 2 degrees
IMAGE = dict(
    type='sky', line_of_sight=dict(type='flat'),
    observable=dict(
        type='pixel_brightness', size=[64, 64], step=0.2, rval=[150.0, 2.0]))
GEOMETRY = dict(xpos=0.3, ypos=-0.2, posa=30, incl=60)
EXPONENTIAL = dict(type='exponential')
EXPONENTIAL_PARAMS = dict(pt_a=1, pt_s=1.5)


def disk(ptraits, **options):
    """A galaxy of a thin disk of the given density polar traits."""
    return dict(components=[dict(
        type='disk', loose=False, tilted=False, rnodes=[0, 6],
        density=dict(ptraits=ptraits), emission=dict(type='band'))
        | options])


def image_of(galaxy, params, observation=IMAGE):
    """The image of a galaxy, and its observation group."""
    group = ObservationGroup(
        galaxy_parser.load(galaxy), observation_parser.load(dict(observation)))
    return group, group.plan().model_h(params)[0]['brightness']['d'].copy()


def write_truth(path, rval=None):
    """
    Write the image of the exponential disk to a FITS file, at the world
    coordinates of IMAGE (or of the given reference RA and Dec), and
    return it.
    """
    group, truth = image_of(disk(EXPONENTIAL), GEOMETRY | EXPONENTIAL_PARAMS)
    coords = group.observations()[0].observable().grid().coords
    if rval is not None:
        coords = coords._replace(rval=tuple(rval))
    fitsutils.write_data(str(path), truth, coords, None, True)
    return truth


def test_images_are_read_by_the_kernels_of_maps(tmp_path):
    # Their cells first (the kernels of the map read only those), of the
    # constants of the map
    from gbkfit.model.components import traits
    write_truth(tmp_path / 'truth.fits')
    image = traits.RPTraitImage.from_file(
        str(tmp_path / 'truth.fits'), size=[3, 2], step=0.1, offsets=True)
    map_ = traits.RPTraitMap(size=[3, 2], step=0.1)
    assert image.uid() == map_.uid()
    assert image.consts() == map_.consts()
    assert list(image.params_sm()) == ['m', 'a', 'dx', 'dy']
    assert image.params_sm()['m'].size() == 6
    assert image.from_galaxy().params == ('m',)


@pytest.mark.parametrize('incl', [60, 30, 120])
def test_a_thin_disk_of_an_image_is_the_image(tmp_path, incl):
    # At its own inclination, and at any other (seen from below beyond 90
    # degrees: its light was negative there): the light is the image
    # (to the cells of 0.1 arcsec, finer than the pixels, at the centre of
    # the exponential)
    truth = write_truth(tmp_path / 'truth.fits')
    trait = dict(
        type='image', file=str(tmp_path / 'truth.fits'), size=[120, 120],
        step=0.1)
    _, model = image_of(disk(trait), GEOMETRY | dict(incl=incl, pt_a=1))
    assert model.sum() == pytest.approx(truth.sum(), rel=2e-3)
    assert np.sqrt(np.mean((model - truth) ** 2)) < 5e-3 * truth.max()
    assert np.abs(model - truth).max() < 5e-2 * truth.max()


def test_offsets_move_the_image(tmp_path):
    # An image whose world coordinates put it 0.6 arcsec east of where it
    # is: its offset of 0.6 arcsec to the west puts it back
    ra = 150.0 + 0.6 / 3600 / np.cos(np.radians(2.0))
    truth = write_truth(tmp_path / 'truth.fits', rval=(ra, 2.0))
    trait = dict(
        type='image', file=str(tmp_path / 'truth.fits'), size=[120, 120],
        step=0.1, offsets=True)
    params = GEOMETRY | dict(pt_a=1, pt_dy=0)
    _, moved = image_of(disk(trait), params | dict(pt_dx=0.6))
    _, unmoved = image_of(disk(trait), params | dict(pt_dx=0))
    assert np.sqrt(np.mean((moved - truth) ** 2)) < 5e-3 * truth.max()
    assert np.sqrt(np.mean((unmoved - truth) ** 2)) > 5e-2 * truth.max()


def test_the_mass_takes_the_light_of_the_image(tmp_path):
    # The pressure support of a disk in its mass takes its density: that of
    # its image, at its geometry, as that of the exponential disk it is
    # (which lowers its rotation by 3 to 5 km/s, at 100 Mpc)
    write_truth(tmp_path / 'truth.fits')
    mass = dict(
        mass=[dict(type='spheroid', name='halo', profile=dict(type='nfw'))],
        distance=dict(type='direct'))
    kinematics = dict(
        type='traits', vptraits=dict(type='mass', pressure_support='jeans'),
        dptraits=dict(type='uniform'))
    params = GEOMETRY | dict(
        vsys=0, distance=100, mass_halo_log_m=12, mass_halo_c=10, dpt_a=15)
    image = dict(
        type='image', file=str(tmp_path / 'truth.fits'), size=[120, 120],
        step=0.1)
    origin = (150.0, 2.0)
    rotation = {}
    for name, ptraits, light in (
            ('exponential', EXPONENTIAL, EXPONENTIAL_PARAMS),
            ('image', image, dict(pt_a=1))):
        galaxy = galaxy_parser.load(
            disk(ptraits, kinematics=kinematics) | mass)
        bound = galaxy.plan_functions(origin).bind(params | light)
        radii = np.array([1.0, 2.0, 4.0])
        rotation[name] = bound['rotation'](radii)
        vcirc = bound['vcirc'](radii * bound['kpc_per_arcsec'])
        assert np.all(vcirc - rotation[name] > 3)
    np.testing.assert_allclose(
        rotation['image'], rotation['exponential'], atol=0.5)


# A profile of the circular velocity at 1 kpc
PROFILE = dict(
    type='profile', functions=['vcirc'], radii=[1.0], radius_unit='kpc')


def rotating_disk_of_image(path):
    """
    A galaxy of a thin disk whose light is the image at path, rotating in
    a point mass.
    """
    trait = dict(type='image', file=str(path), size=[8, 8], step=0.2)
    galaxy = disk(trait) | dict(
        mass=[dict(type='point', name='bh')], distance=dict(type='direct'))
    galaxy['components'][0]['kinematics'] = dict(
        type='traits', vptraits=dict(type='mass'),
        dptraits=dict(type='uniform'))
    return galaxy


def test_images_need_an_observation_on_the_sky(tmp_path):
    write_truth(tmp_path / 'truth.fits')
    galaxy = rotating_disk_of_image(tmp_path / 'truth.fits')
    with pytest.raises(ConfigError, match="an observation on the sky"):
        ObservationGroup(
            galaxy_parser.load(galaxy), observation_parser.load(PROFILE)
        ).plan()


def test_profiles_of_galaxies_of_images(tmp_path):
    # With an observation on the sky, a profile of the galaxy takes its
    # origin on the sky, as the galaxy does: the circular velocity of the
    # point mass, sqrt(G M / r)
    write_truth(tmp_path / 'truth.fits')
    galaxy = rotating_disk_of_image(tmp_path / 'truth.fits')
    group = ObservationGroup(
        galaxy_parser.load(galaxy),
        observation_parser.load([dict(IMAGE), PROFILE]))
    params = GEOMETRY | dict(
        vsys=0, pt_a=1, dpt_a=10, distance=10, mass_bh_log_m=9)
    vcirc = group.plan().model_h(params)[1]['vcirc']['d']
    assert vcirc == pytest.approx([np.sqrt(4.30092e-6 * 1e9)], rel=1e-4)


def test_images_warn_of_their_blur_and_their_cells(tmp_path, caplog):
    # The image blurred as much as the data (its light then blurred by
    # both, 41% more), and cells larger than its pixels
    write_truth(tmp_path / 'truth.fits')
    trait = dict(
        type='image', file=str(tmp_path / 'truth.fits'), size=[30, 30],
        step=0.4, psf=dict(type='gauss', sigma=0.3))
    observation = IMAGE | dict(
        instrument=dict(psf=dict(type='gauss', sigma=0.3)))
    with caplog.at_level(logging.WARNING):
        image_of(disk(trait), GEOMETRY | dict(pt_a=1), observation)
    assert "larger than the pixels of its image" in caplog.text
    assert "41% more than the data" in caplog.text


def test_images_off_the_model_are_warned_of(tmp_path, caplog):
    # An image whose world coordinates put it far from the observation (2
    # degrees east) covers none of the map of its trait: the disk has no
    # light
    write_truth(tmp_path / 'truth.fits', rval=(152.0, 2.0))
    trait = dict(
        type='image', file=str(tmp_path / 'truth.fits'), size=[30, 30],
        step=0.4)
    with caplog.at_level(logging.WARNING):
        _, model = image_of(disk(trait), GEOMETRY | dict(pt_a=1))
    assert "covers none of its map" in caplog.text
    assert not np.any(model)
