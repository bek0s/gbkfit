"""
The tables of the PSF and the LSF of MUSE, of their varying widths
along the spectral axis.
"""

import astropy.io.fits as fits
import astropy.table
import astropy.units as u
import numpy as np


__all__ = [
    'muse_lsf_table',
    'muse_psf_table'
]


# The wavelengths (Angstrom) of the tables of MUSE (its range, and more)
_MUSE_WAVELENGTHS = np.arange(4600.0, 9401.0, 50.0)

# The names of the modes of the PSFs of MUSE (FSFMODE) that MPDAF writes as
# numbers
_MOFFAT_MODES = {2: 'MOFFAT2', '2': 'MOFFAT2'}


def muse_psf_table(header: fits.Header) -> astropy.table.QTable:
    """
    Return the PSF of a MUSE cube (from the keywords of its header: the
    modes MOFFAT1 and 2 of MPDAF, its first field) as a table of the
    options fwhm and beta of a Moffat PSF against wavelength (see
    PSFMoffat.from_fwhm).

    Raises
    ------
    ValueError
        If the header has no PSF of MUSE.
    """
    wavelengths = _MUSE_WAVELENGTHS
    mode = header.get('FSFMODE')
    mode = _MOFFAT_MODES.get(mode, mode)
    fields = [f for f in (*range(10), 99) if f'FSF{f:02d}BET' in header
              or f'FSF{f:02d}FNC' in header]
    if mode not in ('MOFFAT1', 'MOFFAT2') or not fields:
        raise ValueError(
            f"the header has no PSF of MUSE (FSFMODE MOFFAT1, or 2); "
            f"FSFMODE is {header.get('FSFMODE')!r}")
    key = f'FSF{fields[0]:02d}'
    if mode == 'MOFFAT1':
        fwhm = header[f'{key}FWA'] + header[f'{key}FWB'] * wavelengths
        beta = np.full_like(wavelengths, header[f'{key}BET'])
    else:
        # (polynomials of the wavelength normalised to [-0.5, 0.5] in the
        # range FSFLB1 to FSFLB2, the coefficient of the highest power
        # first)
        lb1, lb2 = header['FSFLB1'], header['FSFLB2']
        x = (wavelengths - lb1) / (lb2 - lb1) - 0.5
        fwhm = np.polyval([
            header[f'{key}F{k:02d}']
            for k in range(header[f'{key}FNC'])], x)
        beta = np.polyval([
            header[f'{key}B{k:02d}']
            for k in range(header[f'{key}BNC'])], x)
    return astropy.table.QTable(dict(
        wavelength=wavelengths * u.Angstrom, fwhm=fwhm * u.arcsec,
        beta=beta * u.dimensionless_unscaled))


def muse_lsf_table() -> astropy.table.QTable:
    """
    Return the LSF of MUSE (a Gaussian whose FWHM is that of Bacon et al.
    2017, eq. 8) as a table of its FWHM against wavelength.
    """
    w = _MUSE_WAVELENGTHS
    fwhm = 5.866e-8 * w ** 2 - 9.187e-4 * w + 6.040
    return astropy.table.QTable(dict(
        wavelength=w * u.Angstrom, fwhm=fwhm * u.Angstrom))
