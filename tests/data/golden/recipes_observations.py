"""
The recipes of the golden cases of observables, instruments and
likelihoods (see make_golden.Case): mostly a thin disk, seen in each way.
"""

import pathlib
from typing import Any

import numpy as np


_C_KMS = 299792.458


# --- Shared pieces -------------------------------------------------------

def _thin_disk(**options: Any) -> dict[str, Any]:
    """A thin disk of rings, emitting lines, with the given options."""
    return dict(
        type='disk', loose=False, tilted=False, rnodes=list(range(0, 13)),
        density=dict(ptraits=dict(type='exponential')),
        kinematics=dict(
            type='traits', vptraits=dict(type='arctan', direction='tan'),
            dptraits=dict(type='uniform')),
        emission=dict(type='lines')) | options


def _named(prefix: str, properties: dict[str, Any]) -> dict[str, Any]:
    """Return parameter properties with their names prefixed (prefix_name)."""
    return {f'{prefix}_{name}': value for name, value in properties.items()}


def _sky(observable: dict[str, Any], **options: Any) -> dict[str, Any]:
    """An observation on the sky, flat, float64, of an observable."""
    return dict(
        type='sky', line_of_sight=dict(type='flat'), dtype='float64',
        observable=observable) | options


def _gauss(rms: float) -> dict[str, Any]:
    """Gaussian noise of a dispersion."""
    return dict(type='gauss', rms=rms)


_PSF = dict(type='gauss', sigma=0.6)

_LSF = dict(type='gauss', sigma=20)

_DISK = [dict(components=[_thin_disk()])]

_DISK_TRUTH = dict(
    xpos=0.3, ypos=-0.4, posa=40, incl=55, vsys=10, pt_a=1, pt_s=3,
    vpt_rt=2, vpt_vt=150, dpt_a=25)

_DISK_TRIAL = dict(
    xpos=0.2, ypos=-0.3, posa=dict(value=42, min=0, max=90),
    incl=dict(value=50, min=10, max=80), vsys=dict(value=8, min=-50, max=50),
    pt_a=1, pt_s=3.2, vpt_rt=2.2, vpt_vt=dict(value=140, min=50, max=300),
    dpt_a=dict(value=30, min=5, max=80))

_CUBE = _sky(
    dict(
        type='pixel_spectra', size=[24, 24], step=[0.5, 0.5],
        spectral=dict(size=40, step=20, convention='relativistic')),
    instrument=dict(psf=_PSF, lsf=_LSF))

_MOMENTS = _sky(
    dict(
        type='pixel_moments', size=[24, 24],
        spectral=dict(size=60, step=10, convention='relativistic')),
    instrument=dict(psf=_PSF, lsf=_LSF))


# A flattened Sersic spheroid of stars, of Jeans kinematics, whose mass
# (with that of an NFW halo) holds it up
_STARS = dict(
    type='spheroid', name='stars', rmax=8, emission=dict(type='stars'),
    profile=dict(type='sersic'), shape=dict(type='flattened'),
    kinematics=dict(type='jeans'))

_STARS_TRUTH = dict(
    xpos=0.3, ypos=-0.4, posa=40, incl=60, vsys=10, distance=20,
    stars_ie=1, stars_re=2, stars_n=2, stars_q=0.6, stars_beta_z=0.2,
    stars_kappa=0.8, mass_lum_upsilon=1e10, mass_halo_log_m=12,
    mass_halo_c=10)

_STARS_TRIAL = _STARS_TRUTH | dict(
    posa=dict(value=42, min=0, max=90), incl=dict(value=55, min=30, max=85),
    vsys=dict(value=8, min=-50, max=50),
    stars_beta_z=dict(value=0.1, min=-0.5, max=0.5),
    stars_kappa=dict(value=0.7, min=0, max=1.2),
    mass_lum_upsilon=dict(value=1.2e10, min=1e9, max=1e11),
    mass_halo_log_m=dict(value=11.8, min=10, max=14))


def _mge_stars(kinematics: dict[str, Any]) -> list[dict[str, Any]]:
    """
    A galaxy of a spheroid of stars of two Gaussians of light, of the
    given kinematics in the gravity of its light and a black hole.
    """
    return [dict(
        components=[dict(
            type='spheroid', name='stars', rmax=20,
            emission=dict(type='stars'), profile=dict(type='mge', ngauss=2),
            kinematics=kinematics)],
        mass=[dict(type='light_spheroid', name='lum', component='stars'),
              dict(type='point', name='bh')],
        distance=dict(type='direct', cosmology='Planck18'))]


_MGE_STARS_TRUTH = dict(
    xpos=0.3, ypos=-0.4, posa=40, incl=60, vsys=10, distance=16,
    stars_i0=[50.0, 10.0], stars_sigma=[1.0, 4.0], stars_q=[0.8, 0.8],
    mass_lum_upsilon=1e8, mass_bh_log_m=8.5)


def _small_cube(**options: Any) -> dict[str, Any]:
    """
    An observation of a cube smaller than _CUBE (for cases of several
    observations), with options of its observable.
    """
    return _sky(
        dict(
            type='pixel_spectra', size=[20, 20], step=[0.5, 0.5],
            spectral=dict(size=40, step=20, convention='relativistic'))
        | options, instrument=dict(psf=_PSF, lsf=_LSF))


# A grism of one pixel (1 arcsec) along x per channel of 30 km/s at 4 um
# (a channel spans 4 um * 30 / c)
_GRISM = dict(
    spectral=dict(size=21, step=30, rest='4 um'), step=[1, 1],
    trace=dict(type='polynomial', wave='4 um', dx=[0, _C_KMS / (4 * 30)]))


def _band_disk() -> dict[str, Any]:
    """A small thin disk emitting in a band (no kinematics), for images."""
    return dict(
        type='disk', loose=False, tilted=False,
        rnodes=[round(0.1 * i, 1) for i in range(13)],
        density=dict(ptraits=dict(type='exponential')),
        emission=dict(type='band'))


# --- Files written by prepare --------------------------------------------

def _write_voronoi_bins(folder: pathlib.Path) -> None:
    """
    Write bins.fits: the Voronoi bins of a 24 x 24 grid of 0.5 arcsec, of
    seeded generators denser near the centre; no bin beyond 5.5 arcsec.
    """
    from gbkfit.utils import fitsutils, gridutils
    rng = np.random.default_rng(3)
    radius = 5.5 * rng.uniform(0, 1, 40) ** 0.75
    angle = rng.uniform(0, 2 * np.pi, 40)
    x = (np.arange(24) - 11.5) * 0.5
    xx, yy = np.meshgrid(x, x)
    index = np.argmin(np.hypot(
        xx[..., None] - radius * np.cos(angle),
        yy[..., None] - radius * np.sin(angle)), axis=-1)
    inside = np.hypot(xx, yy) <= 5.5
    # (the bins numbered from 0 without gaps)
    index = np.where(
        inside, np.searchsorted(np.unique(index[inside]), index), -1)
    fitsutils.write_data(
        str(folder / 'bins.fits'), index,
        gridutils.Coords((0.5, 0.5), (11.5, 11.5), (0, 0), 0), overwrite=True)


def _write_uv_coverage(folder: pathlib.Path) -> None:
    """
    Write coverage.npz: 60 random baselines of up to 15 km, of 16 HI
    channels of 20 km/s, of a phase centre 3.6 arcsec from the reference
    pixel.
    """
    from gbkfit.dataset import UVCoverage
    hi = 1420.405752e6
    velocities = 140 - 20 * np.arange(16)
    uvw = np.random.default_rng(1).uniform(-1, 1, (60, 3)) \
        * [15e3, 15e3, 5e3]
    UVCoverage(
        uvw, hi * (1 - velocities / _C_KMS), (150.001, -30.0005), 'LSRK',
        f'{hi} Hz').dump(str(folder / 'coverage.npz'), overwrite=True)


def _write_detection_noise(folder: pathlib.Path) -> None:
    """
    Write detected_noise.fits: the error of the cube of the moment maps
    of _MOMENTS (24 x 24 pixels of 0.5 arcsec), 0.0005 where the disk of
    the truth is within 5.5 arcsec of its centre in its plane, and
    missing (NaN) beyond, where the spectra are blanked as
    non-detections.
    """
    from astropy.io import fits
    truth = _DISK_TRUTH
    x = (np.arange(24) - 11.5) * 0.5
    xx, yy = np.meshgrid(x - truth['xpos'], x - truth['ypos'])
    posa, incl = np.radians(truth['posa']), np.radians(truth['incl'])
    along = -np.sin(posa) * xx + np.cos(posa) * yy
    across = (np.cos(posa) * xx + np.sin(posa) * yy) / np.cos(incl)
    error = np.where(np.hypot(along, across) <= 5.5, 0.0005, np.nan)
    fits.writeto(folder / 'detected_noise.fits', error, overwrite=True)


_USER_LIKELIHOODS = '''\
import numpy as np


def cauchy_residuals(item, data, model, error, mask, params):
    """The residuals of Cauchy noise of the scale s times the errors."""
    x = (model - data) / (params['s'] * error)
    return np.sign(x) * np.sqrt(2 * np.log1p(x ** 2))


def student_residuals(item, data, model, error, mask, params):
    """
    The residuals of Student's t noise of nu degrees of freedom and the
    scale of the errors; a value the model lacks is 3 errors off.
    """
    nu = params['nu']
    x = np.where(np.isnan(model), 3.0, (model - data) / error)
    return np.sign(x) * np.sqrt((nu + 1) * np.log1p(x ** 2 / nu))


def jitter_log_likelihood(item, data, model, error, mask, params):
    """
    The log-likelihood of Gaussian noise of the errors and a jitter in
    quadrature; a value the model lacks is 3 sigma off.
    """
    sigma = np.hypot(error, params['jitter'])
    r = np.where(np.isnan(model), 3.0, (model - data) / sigma)
    kept = np.isfinite(data) if mask is None else mask != 0
    return -0.5 * np.sum(r[kept] ** 2 + np.log(2 * np.pi * sigma[kept] ** 2))
'''


def _write_user_likelihoods(folder: pathlib.Path) -> None:
    """Write user_likelihoods.py: the functions of the user likelihoods."""
    (folder / 'user_likelihoods.py').write_text(_USER_LIKELIHOODS)


def _user_likelihood(func: str, form: str, *params: str) -> dict[str, Any]:
    """A user likelihood of a function of user_likelihoods.py."""
    return dict(type='user', file='user_likelihoods.py', func=func,
                form=form, params=list(params))


# --- The recipes ---------------------------------------------------------

_LENS_TRUTH = dict(
    xpos=0.1, ypos=-0.05, posa=40, incl=55, pt_a=1, pt_s=0.2,
    hst_lens_main_theta_e=1.0, hst_lens_main_slope=2.1,
    hst_lens_main_q=0.8, hst_lens_main_pa=30, hst_lens_main_x=0,
    hst_lens_main_y=0,
    hst_lens_m4_theta_e='hst_lens_main_theta_e', hst_lens_m4_a=0.02,
    hst_lens_m4_pa='hst_lens_main_pa', hst_lens_m4_x='hst_lens_main_x',
    hst_lens_m4_y='hst_lens_main_y',
    hst_lens_shear_gamma=0.05, hst_lens_shear_pa=70)

_LENS_TRIAL = _LENS_TRUTH | dict(
    pt_s=dict(value=0.22, min=0.05, max=1),
    posa=dict(value=45, min=0, max=90),
    hst_lens_main_theta_e=dict(value=1.03, min=0.5, max=2),
    hst_lens_main_q=dict(value=0.78, min=0.5, max=1),
    hst_lens_shear_gamma=dict(value=0.06, min=0, max=0.2))


CASES: list[dict[str, Any]] = [
    dict(
        name='thin_disk_cube',
        doc="A thin disk in a cube, float64: the plain path from a "
            "configuration to the chi-square.",
        galaxies=_DISK, observations=[_CUBE],
        truth=_DISK_TRUTH, trial=_DISK_TRIAL,
        noises=[_gauss(0.002)]),
    dict(
        name='thin_disk_moments',
        doc="The moment maps of a thin disk, float32 (the default dtype), "
            "with weights of the items and a residual for missing values.",
        galaxies=_DISK, observations=[_MOMENTS | dict(dtype='float32')],
        truth=_DISK_TRUTH, trial=_DISK_TRIAL,
        noises=[_gauss(0.002)],
        likelihoods=[dict(
            type='gaussian', weights=dict(moment1=2, moment2=0.5),
            missing_residual=3)]),
    dict(
        name='log_lambda_cube',
        doc="A cube of an axis uniform in ln w (FITS WAVE-LOG, log "
            "velocities) of a galaxy at z = 0.03 that emits Halpha and "
            "[NII] 6583.",
        galaxies=[dict(redshift=0.03, components=[_thin_disk(
            emission=dict(type='lines', lines=['halpha', 'nii6583']))])],
        observations=[_sky(
            dict(
                type='pixel_spectra', size=[20, 20], step=[0.5, 0.5],
                rval=[0, 0],
                spectral=dict(
                    size=50, step=30, rval=9300, convention='log',
                    rest='6564.61 Angstrom')),
            instrument=dict(psf=_PSF, lsf=_LSF))],
        truth=_DISK_TRUTH | dict(nii6583_ratio=0.3),
        trial=_DISK_TRIAL | dict(
            nii6583_ratio=dict(value=0.35, min=0, max=2)),
        noises=[_gauss(0.0005)]),
    dict(
        name='slit_spectra',
        doc="The spectra of a long slit rotated on the sky, wider than "
            "its step.",
        galaxies=_DISK,
        observations=[_sky(
            dict(
                type='slit_spectra', size=[40], step=[0.5],
                spectral=dict(size=40, step=20, convention='relativistic'),
                rota=30, slit_width=1.5),
            instrument=dict(psf=_PSF, lsf=_LSF))],
        truth=_DISK_TRUTH, trial=_DISK_TRIAL,
        noises=[_gauss(0.0005)]),
    dict(
        name='slitless_rows',
        doc="Slitless spectra of a polynomial trace along x, with a solved "
            "flux scale per row (along the dispersion).",
        galaxies=_DISK,
        observations=[_sky(
            dict(type='slitless_spectra', size=[32, 24]) | _GRISM,
            instrument=dict(psf=_PSF), name='grism', flux_scale='per_row')],
        truth=_DISK_TRUTH | dict(grism_flux_scale=1),
        trial=_DISK_TRIAL | dict(grism_flux_scale=dict(solve=dict(
            type='gaussian', mean=1, sigma=0.5, minimum=0))),
        noises=[_gauss(0.02)]),
    dict(
        name='msa_shutters',
        doc="Slitless spectra behind a slitlet of three MSA shutters: two "
            "nods subtracted (A - B, B - A) and aligned on the sky (sky "
            "and image offsets, weights), blurred by the LSF behind the "
            "aperture.",
        galaxies=_DISK,
        observations=[_sky(
            dict(type='slitless_spectra', size=[32, 24], sky_sampling=2,
                 aperture=dict(
                     type='shutters', size=[1, 2], pitch=2.5, count=3,
                     centre=[0.5, 0]),
                 exposures=[
                     dict(sky_offset=[0, 2.5]),
                     dict(sky_offset=[0, -2.5], weight=-1),
                     dict(sky_offset=[0, -2.5], image_offset=[0, 5]),
                     dict(sky_offset=[0, 2.5], image_offset=[0, 5],
                          weight=-1)]) | _GRISM,
            instrument=dict(psf=_PSF, lsf=dict(type='gauss', sigma=40)))],
        truth=_DISK_TRUTH, trial=_DISK_TRIAL,
        noises=[_gauss(0.005)]),
    dict(
        name='regions',
        doc="One galaxy in two observations of regions: the moments of "
            "Voronoi bins, and the spectra of apertures (a circle, an "
            "ellipse, a rectangle and the field).",
        galaxies=_DISK,
        observations=[
            _sky(dict(
                type='region_moments',
                regions=dict(type='bins', file='bins.fits'),
                spectral=dict(size=60, step=10, convention='relativistic')),
                 instrument=dict(psf=_PSF, lsf=_LSF)),
            _sky(dict(
                type='region_spectra', size=[24, 24], step=[0.5, 0.5],
                spectral=dict(size=40, step=20, convention='relativistic'),
                regions=dict(type='apertures', apertures=[
                          dict(type='circle', x=1, y=-1, radius=1.5),
                          dict(type='ellipse', x=0, y=0, a=4, b=2, posa=40),
                          dict(type='rectangle', x=-1, y=1, length=8,
                               width=1, posa=130),
                          dict(type='field')])),
                 instrument=dict(psf=_PSF, lsf=_LSF))],
        truth=_DISK_TRUTH, trial=_DISK_TRIAL,
        noises=[_gauss(0.0005), _gauss(0.005)],
        likelihoods=[dict(type='gaussian', missing_residual=3), None],
        prepare=_write_voronoi_bins),
    dict(
        name='stars_regions',
        doc="The kinematics of the stars of a flattened Sersic spheroid in "
            "Voronoi bins, as pPXF gives them (log velocities, a PSF and no "
            "LSF), as Jeans models are fitted to them: the mean and the "
            "dispersion of each bin (see prep.gauss_hermite_moments). Its "
            "Jeans kinematics in the gravity of its light and an NFW halo.",
        galaxies=[dict(
            components=[_STARS],
            mass=[dict(type='light_spheroid', name='lum', component='stars'),
                  dict(type='spheroid', name='halo', profile=dict(type='nfw'))],
            distance=dict(type='direct', cosmology='Planck18'))],
        observations=[_sky(
            dict(
                type='region_moments',
                regions=dict(type='bins', file='bins.fits'), orders=[1, 2],
                method='moments',
                spectral=dict(size=101, step=10, convention='log')),
            line_of_sight=dict(type='grid', depth=dict(step=0.5)),
            lines=['stars'], instrument=dict(psf=_PSF))],
        truth=_STARS_TRUTH, trial=_STARS_TRIAL,
        noises=[_gauss(0.0002)],
        likelihoods=[dict(type='gaussian', missing_residual=3)],
        prepare=_write_voronoi_bins),
    dict(
        name='stars_orbits',
        doc="The kinematics of the stars of a flattened spheroid in "
            "Voronoi bins, as pPXF gives them (V, sigma, h3 and h4; log "
            "velocities, a PSF and no LSF), as Schwarzschild models are "
            "fitted to them: the weights of their orbits solved, of a prior "
            "whose width the AICp chooses. The data are of its rotating "
            "Jeans kinematics in the gravity of its light and a black hole.",
        galaxies=_mge_stars(dict(type='schwarzschild', norbits=200,
                                 turns=20)),
        mock_galaxies=_mge_stars(dict(type='jeans')),
        observations=[_sky(
            dict(
                type='region_moments',
                regions=dict(type='bins', file='bins.fits'),
                orders=[1, 2, 3, 4], method='gauss_hermite',
                spectral=dict(size=101, step=10, convention='log')),
            line_of_sight=dict(type='grid', depth=dict(step=0.5)),
            lines=['stars'], instrument=dict(psf=_PSF))],
        truth=_MGE_STARS_TRUTH | dict(stars_beta_z=0.0, stars_kappa=0.5),
        trial=_MGE_STARS_TRUTH | dict(
            posa=dict(value=42, min=0, max=90),
            mass_lum_upsilon=dict(value=1.2e8, min=1e7, max=1e9),
            mass_bh_log_m=dict(value=8.3, min=6, max=10),
            stars_weights=dict(solve=dict(
                type='gaussian', mean=1, sigma_aicp=[0.1, 100], minimum=0,
                likelihood='profiled'))),
        noises=[_gauss(0.03)],
        likelihoods=[dict(type='gaussian', missing_residual=3)],
        prepare=_write_voronoi_bins),
    dict(
        name='visibilities',
        doc="The visibilities of HI channels at random baselines, "
            "smoothed by a Hanning correlator response.",
        galaxies=_DISK,
        observations=[_sky(
            dict(type='visibilities', coverage='coverage.npz',
                 fov=[32, 28], step=1, rval=[150, -30]),
            instrument=dict(lsf=dict(type='hanning', width=20)))],
        truth=_DISK_TRUTH, trial=_DISK_TRIAL,
        noises=[_gauss(0.01)],
        prepare=_write_uv_coverage),
    dict(
        name='lens_foreground',
        doc="The image of a source disk behind a foreground lens of three "
            "masses: an elliptical power law, its multipole of order 4 "
            "(tied to it by expressions) and an external shear.",
        galaxies=[dict(components=[_band_disk()])],
        observations=[_sky(
            dict(type='pixel_brightness', size=[40, 40], step=[0.1, 0.1]),
            name='hst', instrument=dict(psf=dict(type='gauss', sigma=0.08)),
            foreground=dict(lens=dict(
                source_size=[40, 40], source_step=[0.05, 0.05], masses=[
                    dict(type='epl', name='main'),
                    dict(type='multipole', order=4, name='m4'),
                    dict(type='shear', name='shear')])))],
        truth=_LENS_TRUTH, trial=_LENS_TRIAL,
        noises=[_gauss(0.01)]),
    dict(
        name='instrument_full',
        doc="A cube seen through a Moffat PSF whose alpha varies with "
            "velocity, an LSF whose sigma does, and a primary beam; "
            "oversampled, median filtered, with pixel weights of the azimuth "
            "in the disk, smoothed.",
        galaxies=[dict(components=[_thin_disk(name='gas', geometry='own')])],
        observations=[_sky(
            dict(
                type='pixel_spectra', size=[16, 16], step=[0.75, 0.75],
                spectral=dict(size=32, step=25, convention='relativistic'),
                smooth_weights=True, median_size=3),
            oversampling=[2, 2, 1],
            pixel_weights=dict(component='gas', traits=dict(
                type='azimuth_power', power=1)),
            instrument=dict(
                psf=dict(type='moffat', beta=2.5, alpha=dict(
                    velocity=[-500, 0, 500], values=[1.2, 0.9, 0.6])),
                lsf=dict(type='gauss', sigma=dict(
                    velocity=[-500, 500], values=[12, 30])),
                primary_beam=dict(type='gauss', fwhm=8, x=1, y=-0.5)))],
        truth=_named('gas', _DISK_TRUTH),
        trial=_named('gas', _DISK_TRIAL),
        noises=[_gauss(0.0002)],
        likelihoods=[dict(type='gaussian', missing_residual=3)]),
    dict(
        name='upper_limits_fitted_noise',
        doc="The moment maps of a named observation with a flux scale, "
            "of spectra blanked beyond the detection: upper limits on its "
            "masked moment 0, and a fitted noise (scale and jitter) of its "
            "moment 1.",
        galaxies=_DISK,
        observations=[_MOMENTS | dict(name='halpha', flux_scale='one')],
        truth=_DISK_TRUTH | dict(halpha_flux_scale=1.2),
        trial=_DISK_TRIAL | dict(
            halpha_flux_scale=dict(value=1.0, min=0.5, max=2),
            halpha_noise_scale_moment1=dict(value=1.3, min=0.5, max=5),
            halpha_noise_jitter_moment1=dict(value=2.0, min=0, max=20)),
        noises=[dict(type='gauss', error='detected_noise.fits')],
        likelihoods=[dict(
            type='gaussian', missing_residual=3,
            upper_limits=dict(moment0=dict(limit=0.12, error=0.04)),
            noise=dict(moment1=dict(
                scale=True, scale_minimum=0.5, jitter=True)))],
        prepare=_write_detection_noise),
    dict(
        name='correlated_noise',
        doc="A cube of noise correlated on the sky by the beam and along "
            "the channels, declared on the data: the likelihood whitens "
            "its residuals.",
        galaxies=_DISK,
        observations=[_sky(
            dict(
                type='pixel_spectra', size=[16, 16], step=[1, 1],
                spectral=dict(size=30, step=20, convention='relativistic')),
            instrument=dict(psf=dict(type='gauss', sigma=1.2), lsf=_LSF))],
        truth=_DISK_TRUTH, trial=_DISK_TRIAL,
        noises=[_gauss(0.0002) | dict(
            spatial_correlation='instrument',
            spectral_correlation=dict(type='gauss', sigma=25))],
        data_options=[dict(
            spatial_correlation=dict(type='gauss', sigma=1.2),
            spectral_correlation=dict(type='gauss', sigma=25))]),
    dict(
        name='user_likelihoods',
        doc="Three observations of user likelihoods of parameters: a cube "
            "and moment maps of residuals (Cauchy and Student's t noise), "
            "and moment maps of a log-likelihood (Gaussian noise of a "
            "free jitter).",
        galaxies=_DISK,
        observations=[
            _small_cube() | dict(name='cube'),
            _MOMENTS | dict(name='maps'),
            _MOMENTS | dict(name='jittered')],
        truth=_DISK_TRUTH,
        trial=_DISK_TRIAL | dict(
            cube_s=1.5, maps_nu=4,
            jittered_jitter=dict(value=0.5, min=0, max=10)),
        noises=[_gauss(0.0005)] * 3,
        likelihoods=[
            _user_likelihood('cauchy_residuals', 'residuals', 's'),
            _user_likelihood('student_residuals', 'residuals', 'nu'),
            _user_likelihood(
                'jitter_log_likelihood', 'log_likelihood', 'jitter')],
        prepare=_write_user_likelihoods),
    dict(
        name='offsets_two_observations',
        doc="One galaxy in a cube and in moment maps whose reference "
            "pixels are 1.4 arcsec apart on the sky (aligned), the maps "
            "with fitted offsets of position and velocity.",
        galaxies=_DISK,
        observations=[
            _small_cube(rval=[150, -30, 0]) | dict(name='kmos'),
            _MOMENTS | dict(
                name='sinfoni', offsets=dict(position=True, velocity=True),
                observable=_MOMENTS['observable']
                | dict(rval=[150.0004, -29.9998]))],
        truth=_DISK_TRUTH | dict(
            sinfoni_dx=0.3, sinfoni_dy=-0.2, sinfoni_dv=5),
        trial=_DISK_TRIAL | dict(
            sinfoni_dx=dict(value=0.1, min=-2, max=2),
            sinfoni_dy=dict(value=0, min=-2, max=2),
            sinfoni_dv=dict(value=0, min=-30, max=30)),
        noises=[_gauss(0.0005), _gauss(0.0005)],
        likelihoods=[None, dict(type='gaussian', missing_residual=3)]),
]
