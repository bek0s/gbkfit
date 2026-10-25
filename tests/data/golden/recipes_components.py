"""
The recipes of the golden cases of components and traits (see
make_golden.Case): each kind of component and the traits of disks.
"""

import pathlib
from typing import Any

import numpy as np


# --- Pieces of the recipes ---------------------------------------------------

_ROTATION = dict(
    vptraits=dict(type='arctan', direction='tan'),
    dptraits=dict(type='uniform'))

_THICK = dict(
    ptraits=dict(type='exponential'), htraits=dict(type='sech2', zmax=4))

_LINES = dict(type='lines')

# Halpha and [NII] 6583
_HALPHA_NII = dict(type='lines', lines=[
    dict(name='ha', rest='6562.8 Angstrom'),
    dict(name='nii', rest='6583.45 Angstrom')])

_PSF = dict(type='gauss', sigma=0.6)
_LSF = dict(type='gauss', sigma=20)


def _disk(**options: Any) -> dict[str, Any]:
    """A disk of rings to 8 arcsec that emits lines, with the given options."""
    return dict(
        type='disk', loose=False, tilted=False, rnodes=list(range(0, 9)),
        density=dict(ptraits=dict(type='exponential')),
        kinematics=_ROTATION, emission=_LINES) | options


def _cube(
        size: list[int], step: list[float], line_of_sight: str = 'flat',
        **observable: Any
) -> dict[str, Any]:
    """
    An observation of a spectral cube, float64, with a psf and an lsf,
    and the given options of its observable.
    """
    return dict(
        type='sky', line_of_sight=dict(type=line_of_sight), dtype='float64',
        observable=dict(type='pixel_spectra', size=size, step=step)
        | observable,
        instrument=dict(psf=_PSF, lsf=_LSF))


def _moments(
        size: list[int], step: list[float], **observable: Any
) -> dict[str, Any]:
    """
    An observation of moment maps, float64, with a psf and an lsf, and the
    given options of its observable.
    """
    return dict(
        type='sky', line_of_sight=dict(type='flat'), dtype='float64',
        observable=dict(
            type='pixel_moments', size=size, step=step, spec_size=60,
            spec_step=10, spec_convention='relativistic') | observable,
        instrument=dict(psf=_PSF, lsf=_LSF))


def _image(
        size: list[int], step: list[float], line_of_sight: str = 'flat',
        **options: Any
) -> dict[str, Any]:
    """
    An observation of an image, float64, with a psf, and the given
    options.
    """
    return dict(
        type='sky', line_of_sight=dict(type=line_of_sight), dtype='float64',
        observable=dict(type='pixel_brightness', size=size, step=step),
        instrument=dict(psf=_PSF)) | options


def _prefixed(name: str, properties: dict[str, Any]) -> dict[str, Any]:
    """Return parameter properties with the prefix of a component."""
    return {f'{name}_{key}': value for key, value in properties.items()}


def _without(properties: dict[str, Any], *names: str) -> dict[str, Any]:
    """Return parameter properties without those of the given names."""
    return {key: value for key, value in properties.items()
            if key not in names}


def _free(value: Any, low: float, high: float) -> dict[str, Any]:
    """Return the property of a free parameter."""
    return dict(value=value, min=low, max=high)


# The geometry and the rotation of the disks, true and trial
_DISK_TRUTH = dict(
    xpos=0.27, ypos=-0.38, posa=40, incl=55, vsys=10, pt_a=1, pt_s=2.5,
    vpt_rt=1.5, vpt_vt=150, dpt_a=25)

_DISK_TRIAL = dict(
    xpos=0.21, ypos=-0.33, posa=_free(43, 0, 90), incl=_free(50, 10, 80),
    vsys=_free(6, -50, 50), pt_a=1, pt_s=2.7, vpt_rt=1.7,
    vpt_vt=_free(140, 50, 300), dpt_a=_free(30, 5, 80))


# --- A1: a thick disk and dust, on a grid and through radiative transfer -----

def _thick_disk(name: str, **roles: Any) -> dict[str, Any]:
    """
    A thick disk of rings to 8 arcsec of its own geometry, of the given
    roles, unmoving.
    """
    return dict(
        type='disk', name=name, geometry='own', loose=False, tilted=False,
        rnodes=list(range(0, 9)), density=_THICK) | roles


_GAS = _disk(name='gas', geometry='own', density=_THICK)
_DUST = _thick_disk('dust', opacity=dict(type='dust'))

_DUST_TRUTH = dict(
    xpos=0.27, ypos=-0.38, posa=40, incl=55, pt_a=0.4, pt_s=3, ht_s=0.4)

_DUST_TRIAL = _DUST_TRUTH | dict(pt_a=_free(0.3, 0, 2), ht_s=0.5)

_THICK_DISK_TRANSFER = dict(
    name='thick_disk_transfer',
    doc="A thick disk of gas and one of dust, seen in two cubes: on a grid "
        "(the dust dims the light behind it) and through radiative "
        "transfer (the march along each line of sight).",
    galaxies=[dict(components=[_GAS, _DUST])],
    observations=[
        _cube([20, 20, 32], [0.8, 0.8, 20], 'grid'),
        _cube([20, 20, 32], [0.8, 0.8, 20], 'transfer')],
    truth=_prefixed('gas', _DISK_TRUTH | dict(ht_s=0.5))
    | _prefixed('dust', _DUST_TRUTH),
    trial=_prefixed('gas', _DISK_TRIAL | dict(ht_s=_free(0.6, 0.1, 2)))
    | _prefixed('dust', _DUST_TRIAL),
    noises=[dict(type='gauss', rms=1.5e-4)] * 2)


# --- A2: a Monte Carlo disk --------------------------------------------------

_MCDISK_CUBE = dict(
    name='mcdisk_cube',
    doc="A thick Monte Carlo disk of clouds of a fixed seed in a cube, "
        "float64, on a grid.",
    galaxies=[dict(components=[
        _disk(type='disk', clouds=dict(cflux=1e-4, seed=7), density=_THICK)])],
    observations=[
        _cube([24, 24, 40], [0.8, 0.8, 15], 'grid')],
    truth=_DISK_TRUTH | dict(ht_s=0.5),
    trial=_DISK_TRIAL | dict(ht_s=_free(0.6, 0.1, 2)),
    noises=[dict(type='gauss', rms=1.5e-4)])


# --- A3: a warped disk -------------------------------------------------------

_WARPED_DISK = dict(
    name='warped_disk',
    doc="A loose and tilted disk (centre, systemic velocity, position "
        "angle and inclination of each ring), with a radial harmonic "
        "flow, a vertical flow and an anisotropic dispersion, in "
        "Gauss-Hermite maps (h3, h4); the disk fills the maps (fits of "
        "spectra of noise alone are wild).",
    galaxies=[dict(components=[_disk(
        geometry='own', loose=True, tilted=True, rnodes=[0, 4, 8, 12, 16],
        kinematics=dict(
            vptraits=[
                dict(type='arctan', direction='tan'),
                dict(type='nw_harmonic', direction='rad', order=2),
                dict(type='uniform', direction='ver')],
            dptraits=[
                dict(type='uniform', direction='rad'),
                dict(type='uniform', direction='tan'),
                dict(type='uniform', direction='ver')]))])],
    observations=[_moments(
        [20, 20], [0.6, 0.6], method='gauss_hermite',
        orders=[0, 1, 2, 3, 4])],
    truth=dict(
        xpos=[0.27, 0.27, 0.32, 0.38, 0.47],
        ypos=[-0.38, -0.38, -0.43, -0.49, -0.58],
        vsys=[10, 10, 11, 12, 14], posa=[40, 40, 42, 46, 52],
        incl=[55, 55, 57, 60, 64], pt_a=1, pt_s=4, vpt_rt=1.5,
        vpt_vt=150, vpt1_a=[0, 15, 20, 15, 10], vpt1_p=[30, 30, 35, 40, 45],
        vpt2_vv=8, dpt_a=30, dpt1_a=20, dpt2_a=12),
    trial=dict(
        xpos=[0.27, 0.27, 0.32, 0.38, 0.47],
        ypos=[-0.38, -0.38, -0.43, -0.49, -0.58],
        vsys=[_free(9, -50, 50), 10, 11, 12, 14],
        posa=[_free(41, 0, 90), 40, 42, 46, _free(51, 0, 90)],
        incl=[55, 55, _free(56, 10, 80), 60, 64], pt_a=1, pt_s=4,
        vpt_rt=1.5, vpt_vt=_free(147, 50, 300), vpt1_a=[0, 12, 20, 18, 10],
        vpt1_p=[30, 30, 35, 40, 45], vpt2_vv=_free(5, -50, 50),
        dpt_a=_free(28, 5, 80), dpt1_a=20, dpt2_a=14),
    noises=[dict(type='gauss', rms=1e-4)],
    likelihoods=[dict(type='gaussian', missing_residual=3)])


# --- A4: density as a sum of traits, and selection ---------------------------

# The densities of the cells of a map of 4 x 4 cells, row by row
_MAP_TRUTH = [
    0.05, 0.10, 0.10, 0.05,
    0.10, 0.30, 0.20, 0.10,
    0.10, 0.20, 0.25, 0.10,
    0.05, 0.10, 0.10, 0.05]

_MAP_TRIAL = [
    0.05, 0.10, 0.10, 0.05,
    0.10, 0.20, 0.20, 0.10,
    0.10, 0.20, 0.35, 0.10,
    0.05, 0.10, 0.10, 0.05]

_BLOBS_MAPS_SELECTION = dict(
    name='blobs_maps_selection',
    doc="The moment maps of a disk whose density is an exponential, three "
        "blobs (a mixture) and a map of 4 x 4 cells, within a range of "
        "radii and of azimuths (selection traits).",
    galaxies=[dict(components=[_disk(
        density=dict(ptraits=[
            dict(type='exponential'),
            dict(type='mixture_exponential', nblobs=3),
            dict(type='map', size=[4, 4], step=2)]),
        sptraits=[dict(type='rrange'), dict(type='azrange')])])],
    observations=[_moments([24, 24], [0.6, 0.6], flux_floor=0.03)],
    truth=_DISK_TRUTH | dict(
        pt1_r=[2, 4, 5.5], pt1_t=[30, 160, 260], pt1_a=[0.8, 0.6, 0.5],
        pt1_s=[0.6, 0.8, 0.5], pt1_q=[0.6, 0.8, 1], pt1_p=[0, 30, 0],
        pt2_a=_MAP_TRUTH, spt_rmin=0.5, spt_rmax=7.5, spt1_p=0, spt1_s=300),
    trial=_DISK_TRIAL | dict(
        pt1_r=[2, 4, 5.5], pt1_t=[30, 150, 260],
        pt1_a=[_free(0.6, 0, 5), 0.6, 0.5], pt1_s=[0.6, 0.8, 0.6],
        pt1_q=[0.6, 0.8, 1], pt1_p=[0, 30, 0], pt2_a=_MAP_TRIAL,
        spt_rmin=0.5, spt_rmax=7.2, spt1_p=10, spt1_s=300),
    noises=[dict(type='gauss', rms=1e-4)],
    likelihoods=[dict(type='gaussian', missing_residual=3)])


# --- A5: a disk and an outflow cone, in two lines ----------------------------

_CONE = dict(
    type='cone', name='cone', geometry='own', rnodes=[0.5, 2, 4],
    density=dict(
        ptraits=dict(type='uniform'), htraits=dict(type='gauss', zmax=1)),
    kinematics=dict(
        vptraits=dict(type='uniform', direction='out'),
        dptraits=dict(type='uniform')),
    emission=_HALPHA_NII)

_CONE_TRUTH = dict(
    xpos=0.27, ypos=-0.38, posa=40, incl=55, vsys=10, wall=30, pt_a=0.3,
    ht_s=0.3, vpt_vo=250, dpt_a=60, nii_ratio=0.8)

_DISK_CONE_LINES = dict(
    name='disk_cone_lines',
    doc="A disk and an outflow cone, each emitting Halpha and [NII] 6583 "
        "(a ratio each), in a cube of a spectral axis of velocities of "
        "Halpha, with a solved flux scale for each spectrum.",
    galaxies=[dict(components=[
        _disk(name='disk', geometry='own', emission=_HALPHA_NII), _CONE])],
    observations=[_cube(
        [24, 24, 40], [0.6, 0.6, 50], 'grid', rval=[0, 0, 450],
        rest='6562.8 Angstrom') | dict(
            name='cube', flux_scale='per_spectrum')],
    truth=_prefixed('disk', _DISK_TRUTH | dict(nii_ratio=0.3))
    | _prefixed('cone', _CONE_TRUTH) | dict(cube_flux_scale=1),
    trial=_prefixed('disk', _DISK_TRIAL | dict(nii_ratio=_free(0.35, 0, 2)))
    | _prefixed('cone', _CONE_TRUTH | dict(
        wall=_free(35, 5, 60), vpt_vo=_free(220, 0, 500),
        nii_ratio=_free(0.7, 0, 2)))
    | dict(cube_flux_scale=dict(solve=dict(
        type='gaussian', mean=1, sigma=0.5, minimum=0))),
    noises=[dict(type='gauss', rms=1.5e-4)])


# --- A6: a point and particles, in an image ----------------------------------

def _write_particles(folder: pathlib.Path) -> None:
    """
    Write the table of the particles of a disk (particles.npz): radii of
    an exponential surface density of scale 2 arcsec (to 8 arcsec), a
    thickness of 0.3 arcsec, a rotation of 150 km/s (2 / pi) atan(R / 1.5),
    and fluxes of 0.05 each; float32, compressed.
    """
    rng = np.random.default_rng(3)
    radius = rng.gamma(2, 2.0, 1000)
    radius = radius[radius < 8]
    theta = rng.uniform(0, 2 * np.pi, radius.size)
    speed = 150 * 2 / np.pi * np.arctan(radius / 1.5)
    columns = dict(
        x=radius * np.cos(theta), y=radius * np.sin(theta),
        z=rng.normal(0, 0.3, radius.size),
        vx=-speed * np.sin(theta), vy=speed * np.cos(theta),
        vz=np.zeros(radius.size), flux=np.full(radius.size, 0.05))
    np.savez_compressed(
        folder / 'particles.npz',
        **{key: value.astype(np.float32) for key, value in columns.items()})


_BAND = dict(type='band')

_POINTS_PARTICLES_BAND = dict(
    name='points_particles_band',
    doc="An image of a point (a nucleus) and of the particles of a table "
        "(the stars of a disk), which emit in a band.",
    galaxies=[dict(components=[
        dict(type='point', name='nucleus', geometry='own', emission=_BAND),
        dict(type='particles', name='stars', geometry='own',
             file='particles.npz', emission=_BAND)])],
    observations=[_image([32, 32], [0.5, 0.5])],
    truth=dict(
        nucleus_xpos=0.27, nucleus_ypos=-0.38, nucleus_flux=2,
        stars_xpos=0.27, stars_ypos=-0.38, stars_posa=40, stars_incl=55),
    trial=dict(
        nucleus_xpos=0.21, nucleus_ypos=-0.38,
        nucleus_flux=_free(1.6, 0, 10), stars_xpos=0.27, stars_ypos=-0.38,
        stars_posa=_free(45, 0, 90), stars_incl=_free(50, 10, 80)),
    noises=[dict(type='gauss', rms=0.02)],
    prepare=_write_particles)


# --- A7: light in a band of a physical unit, dimmed by dust ------------------

_STARS_TRUTH = dict(
    xpos=0.27, ypos=-0.38, posa=40, incl=55, pt_a=1, pt_s=2.5, ht_s=0.5)

_DUST_EXTINCTION_BAND = dict(
    name='dust_extinction_band',
    doc="An image in the band sdss_r of a thick disk of stars whose light "
        "is in MJy/sr (as the galaxy at z = 0.25 emits it), seen in data "
        "in uJy/arcsec2, dimmed by a thick disk of dust of the extinction "
        "law of Cardelli et al.",
    galaxies=[dict(redshift=0.25, components=[
        _thick_disk('stars', emission=dict(
            type='band', bands=['sdss_r'], unit='MJy/sr')),
        _thick_disk('dust', opacity=dict(
            type='dust', law=dict(type='ccm')))])],
    observations=[_image([32, 32], [0.5, 0.5], 'grid', band='sdss_r')],
    # (the mock data are in the unit of the stars, but read as data in
    # uJy/arcsec2: 1 MJy/sr emitted at z = 0.25 is 12.03 uJy/arcsec2 of the
    # data, hence pt_a near 1 / 12.03 at the trial values)
    data_options=[dict(unit='uJy/arcsec2')],
    truth=_prefixed('stars', _STARS_TRUTH) | _prefixed('dust', _DUST_TRUTH),
    trial=_prefixed('stars', _STARS_TRUTH | dict(
        posa=_free(43, 0, 90), pt_a=_free(0.09, 0, 1), ht_s=0.6))
    | _prefixed('dust', _DUST_TRIAL),
    noises=[dict(type='gauss', rms=0.01)])


# --- A8: light from an observed image ----------------------------------------

# The RA and Dec (degrees) of the image and the cube
_SKY = [150.0, 2.0]


def _write_image(folder: pathlib.Path) -> None:
    """
    Write the image of a thin exponential disk (image.fits, float32), as
    GBKFIT makes it: 40 x 40 pixels of 0.4 arcsec at _SKY, of the
    geometry of the disk of the case.
    """
    from gbkfit.driver.drivers.host import DriverHost
    from gbkfit.model import galaxy_parser
    from gbkfit.observation import ObservationGroup, observation_parser
    from gbkfit.utils import fitsutils
    galaxy = dict(components=[dict(
        type='disk', loose=False, tilted=False, rnodes=list(range(0, 9)),
        density=dict(ptraits=dict(type='exponential')),
        emission=dict(type='band'))])
    observation = dict(
        type='sky', line_of_sight=dict(type='flat'),
        observable=dict(
            type='pixel_brightness', size=[40, 40], step=0.4, rval=_SKY))
    group = ObservationGroup(
        galaxy_parser.load(galaxy), observation_parser.load(observation))
    params = dict(xpos=0.27, ypos=-0.38, posa=40, incl=55, pt_a=1, pt_s=2.5)
    image = group.plan(DriverHost()).model_h(params)[0]['brightness']['d']
    coords = group.observations()[0].observable().grid().coords
    fitsutils.write_data(
        str(folder / 'image.fits'), image.astype(np.float32), coords, None,
        True)


_IMAGE_LIGHT = dict(
    name='image_light',
    doc="A cube at an RA and Dec of a disk whose light is that of an "
        "observed image there (an image trait, with offsets), made by its "
        "galaxy at the geometry of the disk.",
    galaxies=[dict(components=[_disk(density=dict(ptraits=dict(
        type='image', file='image.fits', size=[40, 40], step=0.4,
        offsets=True)))])],
    observations=[_cube(
        [24, 24, 40], [0.6, 0.6, 15], rval=[*_SKY, 0])],
    truth=_without(_DISK_TRUTH, 'pt_s') | dict(pt_dx=0, pt_dy=0),
    trial=_without(_DISK_TRIAL, 'pt_s') | dict(
        pt_a=1.1, pt_dx=_free(0.1, -1, 1), pt_dy=0),
    noises=[dict(type='gauss', rms=1.5e-4)],
    prepare=_write_image)


CASES: list[dict[str, Any]] = [
    _THICK_DISK_TRANSFER,
    _MCDISK_CUBE,
    _WARPED_DISK,
    _BLOBS_MAPS_SELECTION,
    _DISK_CONE_LINES,
    _POINTS_PARTICLES_BAND,
    _DUST_EXTINCTION_BAND,
    _IMAGE_LIGHT,
]
