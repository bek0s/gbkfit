"""
The PSF and the LSF of the cubes of KMOS3D (Wisnioski et al. 2019), from
the keywords of their headers.
"""

from typing import Any

import astropy.io.fits as fits
import astropy.table
import astropy.units as u
import astropy.wcs
import numpy as np

from .resolution import resolving_power_table


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
    FWHM against wavelength (see resolving_power_table), across the
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
    wcs = astropy.wcs.WCS(cube)
    if wcs.wcs.spec < 0:
        raise ValueError("the cube has no spectral axis")
    spectral = wcs.sub([wcs.wcs.spec + 1])
    size = cube[f'NAXIS{wcs.wcs.spec + 1}']
    ends = spectral.pixel_to_world_values(np.array([0, size - 1]))
    ends = u.Quantity(ends, spectral.wcs.cunit[0]).to_value(u.um)
    wavelengths = np.linspace(*ends, _WAVELENGTHS)
    return resolving_power_table(
        wavelengths * u.um,
        np.polynomial.polynomial.polyval(wavelengths, coefficients))


def kmos3d_psf(header: fits.Header) -> dict[str, Any]:
    """
    Return the PSF of a cube of KMOS3D: the Moffat fitted to the image of
    its PSF (ESO K3D PSF MOFFAT, of the header of the image), as the
    options of a Moffat PSF by its FWHM (see PSFMoffat.from_fwhm).

    Its position angle (PA) is north through east, as the major axes of
    the images show (of the 240 cubes of the field UDS), and its axis
    ratio (AXRAT) the major axis over the minor.

    Parameters
    ----------
    header : astropy.io.fits.Header
        The header of the image of the PSF.

    Returns
    -------
    dict
        The options of the PSF.

    Raises
    ------
    ValueError
        If the header has no Moffat of KMOS3D.
    """
    key = 'ESO K3D PSF MOFFAT'
    if f'{key} FWHM_MAJ' not in header:
        raise ValueError(
            f"the header has no PSF of KMOS3D ({key} FWHM_MAJ, ...)")
    major, minor = header[f'{key} FWHM_MAJ'], header[f'{key} FWHM_MIN']
    return dict(
        type='moffat', fwhm=float(major), beta=float(header[f'{key} BETA']),
        ratio=float(minor / major), posa=float(header[f'{key} PA']))
