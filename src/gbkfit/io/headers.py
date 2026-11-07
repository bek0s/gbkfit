"""
FITS headers and their world coordinates: the record of what a file of
any format states of an array (see formats), in the keywords of FITS.
"""

import contextlib
import re
import warnings
from collections.abc import Iterator

import astropy.io.fits
import astropy.wcs
import numpy as np

import gbkfit.units


__all__ = [
    'centre_missing_crpix',
    'has_wcs',
    'header_shape',
    'header_unit',
    'ignoring_wcs_fixes',
    'parse_wcs',
    'with_wcs'
]


# The keywords of the world coordinates of each axis, or pair of axes
_WCS_KEYWORDS = re.compile(
    r'(WCSAXES'
    r'|(CRPIX|CRVAL|CDELT|CUNIT|CTYPE|CROTA)\d+'
    r'|(PC|CD|PV|PS)\d+_\d+)$')

# Further keywords of the axes of world coordinates, which another
# orientation of them leaves stale: of each axis (its name, errors and
# period), of the alternate descriptions of the world coordinates (e.g.
# CTYPE1A), and of their distortions (SIP, and those of WCS Paper IV),
# which the model leaves out (see oriented_header)
_AXIS_KEYWORDS = re.compile(
    r'((CNAME|CRDER|CSYER|CPERI)\d+[A-Z]?'
    r'|(WCSAXES|(CRPIX|CRVAL|CDELT|CUNIT|CTYPE|CROTA)\d+'
    r'|(PC|CD|PV|PS)\d+_\d+|WCSNAME)[A-Z]'
    r'|(A|B|AP|BP)_(ORDER|DMAX|\d+_\d+)'
    r'|(CPDIS|CQDIS|DP|DQ|CPERR|CQERR|D2IMDIS|D2IMERR|D2IM)\d+(\..*)?)$')

# The keywords of the rest of the spectral axis, also in their old forms
_REST_KEYWORDS = ('RESTWAV', 'RESTFRQ', 'RESTWAVE', 'RESTFREQ')

# The keywords of the length of each axis
_NAXIS_KEYWORDS = re.compile(r'NAXIS\d+')

# The keywords of the unit of each world axis
_CUNIT_KEYWORDS = re.compile(r'CUNIT\d+')


def has_wcs(header: astropy.io.fits.Header) -> bool:
    """Return whether a header has world coordinates."""
    return any(_WCS_KEYWORDS.match(key) for key in header)


def with_wcs(
        header: astropy.io.fits.Header, wcs: astropy.wcs.WCS
) -> astropy.io.fits.Header:
    """
    Return a copy of a header with the world coordinates of a WCS, and its
    rest (if any), instead of its own (and without the other keywords of
    its axes, see _AXIS_KEYWORDS).
    """
    header = header.copy()
    for key in [
            key for key in header
            if _WCS_KEYWORDS.match(key) or _AXIS_KEYWORDS.match(key)
            or key in _REST_KEYWORDS]:
        del header[key]
    header.update(wcs.to_header())
    return header


def header_shape(header: astropy.io.fits.Header) -> tuple[int, ...]:
    """
    Return the shape of the data of a FITS header (in the order of numpy:
    NAXISn first).
    """
    ndim = header.get('NAXIS', 0)
    return tuple(header.get(f'NAXIS{i}', 0) for i in range(ndim, 0, -1))


def header_unit(header: astropy.io.fits.Header) -> str | None:
    """
    Return the unit of the data of a header, as it gives it (BUNIT), if
    it does.
    """
    unit = header.get('BUNIT')
    if unit is None or not str(unit).strip():
        return None
    return str(unit).strip()


def centre_missing_crpix(
        header: astropy.io.fits.Header,
        shape: tuple[int, ...]
) -> astropy.io.fits.Header:
    """
    Return a header with the reference pixel of each axis that has none
    and no world position (neither CTYPEn nor CRVALn; e.g. only the size
    of its pixels) at its centre: the convention of the model for data
    without world coordinates. An axis with a world position keeps the
    reference pixel of FITS, 0, as other tools read it.

    Parameters
    ----------
    header : astropy.io.fits.Header
        The header.
    shape : tuple[int, ...]
        The shape of its data (numpy order).

    Returns
    -------
    astropy.io.fits.Header
        A copy of the header, with the missing CRPIXn.
    """
    header = header.copy()
    for n, size in enumerate(shape[::-1], start=1):
        placed = f'CTYPE{n}' in header or f'CRVAL{n}' in header
        if f'CRPIX{n}' not in header and not placed:
            header[f'CRPIX{n}'] = size / 2 + 0.5
    return header


def with_shape(
        header: astropy.io.fits.Header, shape: tuple[int, ...]
) -> astropy.io.fits.Header:
    """Return a copy of a header of an array of another shape (NAXISn)."""
    header = header.copy()
    for key in [key for key in header if _NAXIS_KEYWORDS.fullmatch(key)]:
        del header[key]
    header['NAXIS'] = len(shape)
    for n, size in enumerate(shape[::-1], start=1):
        header[f'NAXIS{n}'] = size
    return header


def parse_wcs(source: str, header: astropy.io.fits.Header) -> astropy.wcs.WCS:
    """
    Return the world coordinates of a header, or raise ValueError.

    The unit of each axis (CUNITn) must be valid (see
    gbkfit.units.parse_unit): astropy would correct some, e.g. 'HZ'
    to Hz, and no unit is corrected.
    """
    for key in header:
        text = str(header[key]).strip()
        if not _CUNIT_KEYWORDS.fullmatch(key) or not text:
            continue
        try:
            gbkfit.units.parse_unit(text)
        except ValueError:
            raise ValueError(
                f"{source}: {key} = {text!r} is not a valid unit (see the "
                f"FITS standard, section 4.3); correct it in a copy of the "
                f"file") from None
    try:
        return astropy.wcs.WCS(header)
    except Exception as e:
        raise ValueError(
            f"{source}: invalid world coordinates: {e}") from e


def linear_transform(wcs: astropy.wcs.WCS) -> np.ndarray:
    """
    Return the linear transformation from pixels to intermediate world
    coordinates: one row for each world axis, one column for each pixel
    axis.
    """
    return wcs.wcs.get_cdelt()[:, None] * wcs.wcs.get_pc()


@contextlib.contextmanager
def ignoring_wcs_fixes() -> Iterator[None]:
    """
    Leave out, within the block, the warnings of the fixes that astropy
    makes to headers as it reads their world coordinates
    (FITSFixedWarning): astropy cannot fix a header without warning.
    """
    # That changes the warning filters of the whole process for a moment,
    # which can only show or hide a warning of another thread (see the
    # type adapters of options, gbkfit.config.types, for the fix:
    # context-aware warnings)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', astropy.wcs.FITSFixedWarning)
        yield
