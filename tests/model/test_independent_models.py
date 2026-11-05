"""
Tests of the first-science model against independent computations:
models written in numpy from the physics, not from the code of GBKFIT,
compared in km/s by the moments of the spectra of bright pixels. The
golden and regression tests catch changes; these catch errors common to
the pieces of the model (e.g. a dispersion added twice). Their margins
are 100 to 1000 times below the errors of the data of high-z IFUs (5 to
20 km/s).
"""

import functools

import gbkfit.params
import numpy as np
import pytest
import scipy.signal
import scipy.special

from gbkfit.model import Extras
from modelutils import group_plan


# The speed of light (km/s)
C = 299792.458

# A cube of N x N pixels of STEP arcsec and NZ channels of DV km/s
N, STEP, NZ, DV = 41, 0.5, 121, 10.0

# A disk: its position angle and inclination (deg), the scale length of
# its exponential brightness (arcsec), its arctan rotation (turnover
# radius, arcsec; velocity, km/s) and its dispersion (km/s)
PA, INCL, RD, RT, VT, SIG = 130.0, 55.0, 3.0, 1.5, 200.0, 50.0

# A Gaussian PSF (sigma, arcsec) and LSF (sigma, km/s)
PSF_SIGMA, LSF_SIGMA = 0.6, 30.0

# The thick disk: its position angle, the scale height of its sech2
# (arcsec) and where it ends (arcsec), and its dispersion (km/s)
THICK_PA, HS, ZMAX, THICK_SIG = 30.0, 1.0, 6.0, 40.0


def _disk(**options):
    """A disk of rings to 40 arcsec of arctan rotation that emits a line."""
    return dict(
        type='thin_disk', loose=False, tilted=False,
        rnodes=list(np.arange(0, 41, 1.0)),
        density=dict(ptraits=dict(type='exponential')),
        kinematics=dict(
            type='traits', vptraits=dict(type='arctan', direction='tan'),
            dptraits=dict(type='uniform')),
        emission=dict(type='lines')) | options


def _cube(driver, galaxy, observable, line_of_sight, properties):
    """Return the model spectra of a galaxy in a cube, in float64."""
    plan = group_plan([dict(
        driver=dict(type=driver.type()), galaxy=galaxy,
        observable=observable | dict(dtype='float64'),
        line_of_sight=line_of_sight)])
    params = gbkfit.params.ParamSpace.from_properties(
        plan.group().pdescs(), properties)
    return np.asarray(
        plan.model_h(params.evaluate())[0]['spectra']['d'], float)


def _moments(cube, velocities):
    """Return the flux, the mean velocity and the dispersion of spectra."""
    flux = cube.sum(0)
    mean = np.tensordot(velocities, cube, 1) / flux
    square = np.tensordot(velocities ** 2, cube, 1) / flux
    return flux, mean, np.sqrt(square - mean ** 2)


def _velocities(nz, step):
    """Return the velocities of the centres of nz channels about 0."""
    return (np.arange(nz) - (nz - 1) / 2) * step


def _sky(factor):
    """
    Return the sky coordinates x and y (arcsec, x to the west) of the
    pixels of the cube, each split into factor x factor points.
    """
    x = (np.arange(N * factor) - (N * factor - 1) / 2) * STEP / factor
    return np.meshgrid(x, x)


def _bright(flux, radius):
    """Return the pixels within a radius (arcsec) of 5% of the peak flux."""
    y, x = (np.indices(flux.shape) - (N - 1) / 2) * STEP
    inner = np.hypot(x, y) < radius
    return inner & (flux > 0.05 * flux[inner].max())


def _rotation(radius):
    """Return the arctan rotation velocity (km/s) at radii (arcsec)."""
    return VT * 2 / np.pi * np.arctan(radius / RT)


@functools.cache
def _thin_disk():
    """
    Return the cube of the thin disk seen with the PSF and the LSF, made
    independently: on 9 x 9 points of each pixel, its brightness and its
    line (a Gaussian of the dispersion and of the LSF in quadrature,
    integrated over each channel), convolved with the PSF by FFT, and
    averaged over each pixel.
    """
    factor = 9
    x, y = _sky(factor)
    pa, incl = np.radians(PA), np.radians(INCL)
    along = -x * np.sin(pa) + y * np.cos(pa)
    across = (-x * np.cos(pa) - y * np.sin(pa)) / np.cos(incl)
    radius = np.hypot(along, across)
    cos_azimuth = np.divide(
        along, radius, out=np.zeros_like(radius), where=radius > 0)
    vlos = _rotation(radius) * np.sin(incl) * cos_azimuth
    sigma = np.hypot(SIG, LSF_SIGMA)
    edges = (np.arange(NZ + 1) - NZ / 2) * DV
    cdf = scipy.special.ndtr((edges[:, None, None] - vlos) / sigma)
    cube = np.exp(-radius / RD) * np.diff(cdf, axis=0) / DV
    half = int(6 * PSF_SIGMA * factor / STEP)
    k = np.arange(-half, half + 1) * STEP / factor
    psf = np.exp(-0.5 * (k[:, None] ** 2 + k ** 2) / PSF_SIGMA ** 2)
    cube = scipy.signal.fftconvolve(
        cube, psf[None] / psf.sum(), mode='same', axes=(1, 2))
    return cube.reshape(NZ, N, factor, N, factor).mean(axis=(2, 4))


@pytest.mark.parametrize('oversampling, tolerance, flux_tolerance', [
    (1, 1.0, 0.03), (3, 0.1, 0.005)])
def test_a_thin_disk_in_a_cube(
        driver, oversampling, tolerance, flux_tolerance):
    # With a PSF and an LSF; sampled at the centres of its pixels, the
    # cusp of the exponential at its centre takes 2% of its flux there
    cube = _cube(
        driver, dict(components=[_disk()]),
        dict(type='pixel_spectra', size=[N, N], step=[STEP, STEP],
             spectral=dict(size=NZ, step=DV, convention='relativistic'),
             oversampling=[oversampling, oversampling, 1],
             psf=dict(type='gauss', sigma=PSF_SIGMA),
             lsf=dict(type='gauss', sigma=LSF_SIGMA)),
        dict(type='flat'),
        dict(vsys=0, xpos=0, ypos=0, posa=PA, incl=INCL, pt_a=1, pt_s=RD,
             vpt_rt=RT, vpt_vt=VT, dpt_a=SIG))
    velocities = _velocities(NZ, DV)
    flux, mean, dispersion = _moments(cube, velocities)
    true_flux, true_mean, true_dispersion = _moments(_thin_disk(), velocities)
    bright = _bright(true_flux, 7)
    ratio = flux[bright] / true_flux[bright]
    np.testing.assert_allclose(ratio, ratio.mean(), rtol=flux_tolerance)
    np.testing.assert_allclose(
        mean[bright], true_mean[bright], atol=tolerance)
    np.testing.assert_allclose(
        dispersion[bright], true_dispersion[bright], atol=tolerance)


@functools.cache
def _thick_disk(incl):
    """
    Return the flux, the mean velocity and the dispersion of the thick
    disk at an inclination (deg), without PSF or LSF, made independently:
    integrated along the line of sight of 6 x 6 points of each pixel, in
    steps of 0.05 arcsec, and summed over each pixel; the dispersion has
    the variance of a channel, DV^2 / 12.
    """
    factor, ds = 6, 0.05
    x, y = _sky(factor)
    pa, incl = np.radians(THICK_PA), np.radians(incl)
    along = -x * np.sin(pa) + y * np.cos(pa)
    across = -x * np.cos(pa) - y * np.sin(pa)
    flux, flux_v, flux_v2 = (np.zeros_like(x) for _ in range(3))
    for s in np.arange(-30, 30, ds) + ds / 2:
        in_plane = across * np.cos(incl) + s * np.sin(incl)
        height = -across * np.sin(incl) + s * np.cos(incl)
        radius = np.hypot(along, in_plane)
        density = (np.exp(-radius / RD) / np.cosh(height / HS) ** 2
                   * (np.abs(height) <= ZMAX))
        cos_azimuth = np.divide(
            along, radius, out=np.zeros_like(radius), where=radius > 0)
        v = _rotation(radius) * np.sin(incl) * cos_azimuth
        flux += density
        flux_v += density * v
        flux_v2 += density * (v ** 2 + THICK_SIG ** 2)

    def summed(a):
        return a.reshape(N, factor, N, factor).sum(axis=(1, 3))
    mean = summed(flux_v) / summed(flux)
    square = summed(flux_v2) / summed(flux) + DV ** 2 / 12
    return summed(flux), mean, np.sqrt(square - mean ** 2)


@pytest.mark.parametrize('incl', [75.0, 88.0])
def test_a_thick_disk_in_a_cube(driver, incl):
    # An exponential of sech2 heights, seen through it on a grid along
    # the line of sight (of 0.1 arcsec), its pixels split 3 x 3
    cube = _cube(
        driver,
        dict(components=[_disk(type='thick_disk', density=dict(
            ptraits=dict(type='exponential'),
            htraits=dict(type='sech2', zmax=ZMAX)))]),
        dict(type='pixel_spectra', size=[N, N], step=[STEP, STEP],
             spectral=dict(size=NZ, step=DV, convention='relativistic'),
             oversampling=[3, 3, 1]),
        dict(type='grid', depth=dict(extent=60, step=0.1)),
        dict(vsys=0, xpos=0, ypos=0, posa=THICK_PA, incl=incl, pt_a=1,
             pt_s=RD, ht_s=HS, vpt_rt=RT, vpt_vt=VT, dpt_a=THICK_SIG))
    flux, mean, dispersion = _moments(cube, _velocities(NZ, DV))
    true_flux, true_mean, true_dispersion = _thick_disk(incl)
    bright = _bright(true_flux, 8)
    ratio = flux[bright] / true_flux[bright]
    np.testing.assert_allclose(ratio, ratio.mean(), rtol=0.01)
    np.testing.assert_allclose(mean[bright], true_mean[bright], atol=0.15)
    np.testing.assert_allclose(
        dispersion[bright], true_dispersion[bright], atol=0.15)


def test_a_galaxy_at_redshift_1_5(driver):
    # On a spectral axis linear in wavelength at Halpha x 2.5 (1.64 um,
    # channels of 1.8e-4 um, as KMOS), the line centres and widths of a
    # disk at z = 1.5, turned into velocities of its frame (relativistic,
    # and c dw / w), are those of the same disk at z = 0 on an axis of
    # velocities (the thin disk tested above): to 0.1 km/s
    z, rest, dw, nz = 1.5, 0.656461, 1.8e-4, 61
    centre = rest * (1 + z)
    disk = _disk(
        rnodes=list(np.arange(0, 21, 0.5)),
        emission=dict(type='lines', lines=['halpha']))
    observable = dict(
        type='pixel_spectra', size=[31, 31], step=[0.2, 0.2],
        psf=dict(type='gauss', sigma=0.25), lsf=dict(type='gauss', sigma=35))
    properties = dict(
        vsys=0, xpos=0, ypos=0, posa=PA, incl=INCL, pt_a=1, pt_s=1,
        vpt_rt=0.5, vpt_vt=VT, dpt_a=SIG)
    cube = _cube(
        driver, dict(redshift=z, components=[disk]),
        observable | dict(spectral=dict(
            size=nz, step=C * dw / centre, rval=0, convention='optical',
            rest=f'{centre} um')),
        dict(type='flat'), properties)
    flux, wavelength, width = _moments(
        cube, centre + (np.arange(nz) - (nz - 1) / 2) * dw)
    ratio = (wavelength / centre) ** 2
    mean = C * (ratio - 1) / (ratio + 1)
    dispersion = np.sqrt(
        (C * width / wavelength) ** 2 - (C * dw / wavelength) ** 2 / 12)
    at_rest = _cube(
        driver, dict(components=[_disk(rnodes=disk['rnodes'])]),
        observable | dict(spectral=dict(
            size=301, step=3, convention='relativistic')),
        dict(type='flat'), properties)
    rest_flux, rest_mean, rest_dispersion = _moments(
        at_rest, _velocities(301, 3))
    rest_dispersion = np.sqrt(rest_dispersion ** 2 - 3 ** 2 / 12)
    y, x = (np.indices(flux.shape) - 15) * 0.2
    bright = (np.hypot(x, y) < 2) & (flux > 0.05 * flux.max())
    np.testing.assert_allclose(mean[bright], rest_mean[bright], atol=0.1)
    np.testing.assert_allclose(
        dispersion[bright], rest_dispersion[bright], atol=0.1)


@pytest.mark.parametrize('posa, incl', [(0, 0), (30, 60)])
def test_an_elliptical_blob(driver, posa, incl):
    # A Gaussian blob of a disk (at radius r and azimuth t, of sigma s,
    # axis ratio q, its major axis at p from the radial direction), as
    # traits of mixture_gauss give it, brighter on the sky by 1 / cos(i)
    r, t, s, q, p = 1.5, 40.0, 0.4, 0.5, 20.0
    n, step = 121, 0.05
    plan = group_plan([dict(
        driver=dict(type=driver.type()),
        galaxy=dict(components=[_disk(
            rnodes=list(range(0, 10)),
            density=dict(ptraits=dict(type='mixture_gauss', nblobs=1)),
            kinematics=dict(
                type='traits', vptraits=dict(type='uniform', direction='tan'),
                dptraits=dict(type='uniform')))]),
        observable=dict(
            type='pixel_spectra', size=[n, n], step=[step, step],
            spectral=dict(size=21, step=5, convention='relativistic')),
        line_of_sight=dict(type='flat'))])
    params = gbkfit.params.ParamSpace.from_properties(
        plan.group().pdescs(), dict(
            vsys=0, xpos=0, ypos=0, posa=posa, incl=incl, vpt_vt=0, dpt_a=10,
            pt_r=r, pt_t=t, pt_a=1, pt_s=s, pt_q=q, pt_p=p))
    extra = Extras()
    plan.model_h(params.evaluate(), extra)
    image = extra.get('bdata', component='component0').data.reshape(n, n)
    y, x = (np.indices((n, n)) - (n - 1) / 2) * step
    pa, incl = np.radians(posa), np.radians(incl)
    along = -x * np.sin(pa) + y * np.cos(pa)
    across = (-x * np.cos(pa) - y * np.sin(pa)) / np.cos(incl)
    dx = along - r * np.cos(np.radians(t))
    dy = across - r * np.sin(np.radians(t))
    major = np.radians(t + p + 90)
    u = dx * np.cos(major) + dy * np.sin(major)
    w = -dx * np.sin(major) + dy * np.cos(major)
    expected = np.exp(
        -(u ** 2 + w ** 2 / q ** 2) / (2 * s ** 2)) / np.cos(incl)
    np.testing.assert_allclose(image, expected, rtol=0, atol=1e-5)
