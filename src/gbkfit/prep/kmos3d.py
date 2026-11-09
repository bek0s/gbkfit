"""
The PSF and the LSF of the cubes of KMOS3D (Wisnioski et al. 2019), from
the keywords of their headers.
"""

import astropy.io.fits as fits
import astropy.table
import astropy.units as u
import numpy as np

from gbkfit.instrument import PSFMoffat
from gbkfit.io import parse_wcs, spectral_points
from .resolution import resolving_power_lsf_table


__all__ = [
    'kmos3d_lsf_table',
    'kmos3d_psf'
]


# The number of wavelengths of the table of an LSF, across the channels
_WAVELENGTHS = 50


def kmos3d_lsf_table(
        primary: fits.Header, cube: fits.Header
) -> astropy.table.QTable:
    """
    Return the LSF of a cube of KMOS3D: a Gaussian of its resolving power,
    a polynomial of the wavelength in micron (ESO K3D RES COEFF0, ..., of
    order ESO K3D RES ORDER, of its primary header), as a table of its
    FWHM against wavelength (see resolving_power_lsf_table), across the
    channels of the cube.

    Parameters
    ----------
    primary : astropy.io.fits.Header
        The primary header of the file.
    cube : astropy.io.fits.Header
        The header of the cube (of the flux).

    Returns
    -------
    QTable
        The table: the wavelength (micron), and the FWHM (km/s).

    Raises
    ------
    ValueError
        If the primary header has no resolving power of KMOS3D, or the
        cube has no spectral axis.
    """
    order = primary.get('ESO K3D RES ORDER')
    if order is None:
        raise ValueError(
            "the primary header has no resolving power of KMOS3D (ESO K3D "
            "RES ORDER and COEFF0, ...)")
    coefficients = [primary[f'ESO K3D RES COEFF{k}'] for k in range(order + 1)]
    wcs = parse_wcs('the cube', cube)
    if wcs.wcs.spec < 0:
        raise ValueError("the cube has no spectral axis")
    points = spectral_points(wcs, cube[f'NAXIS{wcs.wcs.spec + 1}'])
    wavelengths = np.linspace(*points[[0, -1]].to_value(u.um), _WAVELENGTHS)
    return resolving_power_lsf_table(
        wavelengths * u.um,
        np.polynomial.polynomial.polyval(wavelengths, coefficients))


def kmos3d_psf(header: fits.Header) -> PSFMoffat:
    """
    Return the PSF of a cube of KMOS3D: the Moffat fitted to the image of
    its PSF (ESO K3D PSF MOFFAT, of the header of the image), by its FWHM
    (see PSFMoffat.from_fwhm).

    Its position angle (PA) is north through east, as the major axes of
    the images show (of the 240 cubes of the field UDS), and its axis
    ratio the minor axis over the major (FWHM_MIN / FWHM_MAJ; AXRAT is
    the inverse).

    Parameters
    ----------
    header : astropy.io.fits.Header
        The header of the image of the PSF.

    Returns
    -------
    PSFMoffat
        The PSF.

    Raises
    ------
    ValueError
        If the header has no Moffat of KMOS3D, or its options are invalid.
    """
    key = 'ESO K3D PSF MOFFAT'
    if f'{key} FWHM_MAJ' not in header:
        raise ValueError(
            f"the header has no PSF of KMOS3D ({key} FWHM_MAJ, ...)")
    major, minor = header[f'{key} FWHM_MAJ'], header[f'{key} FWHM_MIN']
    return PSFMoffat.from_fwhm(
        float(major), float(header[f'{key} BETA']), float(minor / major),
        float(header[f'{key} PA']))
