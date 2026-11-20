"""
FITS headers and their world coordinates: the record of what a file of
any format states of an array (see formats), in the keywords of FITS.
"""

import contextlib
import re
import warnings
from collections.abc import Iterator, Sequence

import astropy.io.fits
import astropy.wcs
import numpy as np

import gbkfit.units
from gbkfit.config import InputError


__all__ = [
    'data_unit',
    'has_wcs',
    'header_shape',
    'header_unit',
    'ignoring_wcs_fixes',
    'parse_wcs',
    'shifted_header'
]


# The keywords of the world coordinates of each axis, or pair of axes
_WCS_KEYWORDS = re.compile(
    r'WCSAXES'
    r'|(CRPIX|CRVAL|CDELT|CUNIT|CTYPE|CROTA)\d+'
    r'|(PC|CD|PV|PS)\d+_\d+')

# Further keywords of the axes of world coordinates, which another
# orientation of them leaves stale: of each axis (its name, errors and
# period), of the alternate descriptions of the world coordinates (e.g.
# CTYPE1A), and of their distortions (SIP, and those of WCS Paper IV),
# which the model leaves out (see _oriented_header)
_AXIS_KEYWORDS = re.compile(
    r'(CNAME|CRDER|CSYER|CPERI)\d+[A-Z]?'
    r'|(WCSAXES|(CRPIX|CRVAL|CDELT|CUNIT|CTYPE|CROTA)\d+'
    r'|(PC|CD|PV|PS)\d+_\d+|WCSNAME)[A-Z]'
    r'|(A|B|AP|BP)_(ORDER|DMAX|\d+_\d+)'
    r'|(CPDIS|CQDIS|DP|DQ|CPERR|CQERR|D2IMDIS|D2IMERR|D2IM)\d+(\..*)?')

# The keywords of the rest of the spectral axis, also in their old forms
_REST_KEYWORDS = ('RESTWAV', 'RESTFRQ', 'RESTWAVE', 'RESTFREQ')

# The types of spectral axes that wcslib reads only of a rest: FELO of
# AIPS (optical velocities linear in frequency), and those of FITS whose
# algorithm code is of another kind than their values (e.g. VOPT-F2W)
_TYPES_OF_A_REST = re.compile(r'FELO.*|.{4}-[FWVA]2[FWVA]')


# The keywords of the length of each axis
_NAXIS_KEYWORDS = re.compile(r'NAXIS\d+')

# The keywords of the unit of each world axis
_CUNIT_KEYWORDS = re.compile(r'CUNIT\d+')


def _option_error(message: str, *path: str) -> InputError:
    """
    Return an error of a header that an option of its data fixes, at its
    path in the options (e.g. 'spectral', 'unit'): where the data are read
    from a configuration, its path there (see gbkfit.config.config_path,
    which adds the path of the data before it), and where a header is
    described, the option of the problem (see describe_fits).
    """
    return InputError(message, path)


def has_wcs(header: astropy.io.fits.Header) -> bool:
    """Return whether a header has world coordinates."""
    return any(_WCS_KEYWORDS.fullmatch(key) for key in header)


def _with_wcs(
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
            if _WCS_KEYWORDS.fullmatch(key) or _AXIS_KEYWORDS.fullmatch(key)
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
    # (a BUNIT without a value is None)
    unit = str(header.get('BUNIT') or '').strip()
    return unit or None


def data_unit(
        source: str,
        header: astropy.io.fits.Header,
        unit: str | None = None
) -> gbkfit.units.AnyUnit | None:
    """
    Return the unit of the values of a header: that given (the option
    unit of a dataset), which replaces that of the header, or else that of
    the header (BUNIT), if any. Units are read as FITS writes them (see
    gbkfit.units.parse_unit), never their legacy spellings (e.g.
    'JY/BEAM').

    Raises
    ------
    InputError
        Of the option unit, if the unit given is not a unit, or, none
        given, BUNIT is not.
    """
    if unit is not None:
        try:
            return gbkfit.units.parse_unit(unit)
        except ValueError:
            raise _option_error(f"unit {unit!r} invalid", 'unit') from None
    text = header_unit(header)
    if text is None:
        return None
    try:
        return gbkfit.units.parse_unit(text)
    except ValueError:
        raise _option_error(
            f"{source}: BUNIT {text!r} invalid", 'unit') from None


def _centre_missing_crpix(
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


def shifted_header(
        header: astropy.io.fits.Header, offset: Sequence[int]
) -> astropy.io.fits.Header:
    """
    Return the header of data whose first pixel is the pixel at the given
    offset (on each numpy axis) of the data of a header: positive where
    the data are cropped, negative where they are padded.
    """
    if not has_wcs(header):
        return header
    wcs = astropy.wcs.WCS(header)
    wcs.wcs.crpix -= np.asarray(offset)[::-1]
    return _with_wcs(header, wcs)


def _with_shape(
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
    Return the world coordinates of a header, or raise InputError (of the
    failures of wcslib, a short message, see _wcs_failure).

    The unit of each axis (CUNITn) must be valid (see
    gbkfit.units.parse_unit): astropy would correct some, e.g. 'HZ'
    to Hz, and no unit is corrected (that of a spectral axis is given
    instead, see model_header, which checks it first). A CUNITn that is
    not text (e.g. the
    number 0, of the PSF images of KMOS3D) is no unit: astropy leaves it
    out, with a warning, as an axis without a unit.
    """
    for key in filter(_CUNIT_KEYWORDS.fullmatch, header):
        value = header[key]
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            gbkfit.units.parse_unit(value.strip())
        except ValueError:
            raise InputError(
                f"{source}: {key} {value.strip()!r} invalid") from None
    try:
        return astropy.wcs.WCS(header)
    except ValueError as e:
        message, path = _wcs_failure(header, str(e))
        raise InputError(f"{source}: {message}", path) from None


# The lines of wcslib that say where it failed (e.g. "ERROR 4 in
# wcs_types() at line 3121 of file cextern/wcslib/C/wcs.c:"), and the
# failures it says after them that have short messages (see _wcs_failure)
_WCSLIB_WHERE = re.compile(r'ERROR \d+ in \w+\(\) at line \d+ of file .*')
_WCSLIB_FAILURES = (
    (re.compile(r'.*matrix is singular'), 'singular'),
    (re.compile(r'Unrecognized projection code \((\S+) in (CTYPE\d+)\)'),
     'projection'),
    (re.compile(
        r'Inconsistent projection types \(expected (\S+), got (\S+) in '
        r'(CTYPE\d+)\)'), 'pair'),
    (re.compile(r'Unmatched celestial axes'), 'unpaired'),
    (re.compile(
        r"In (CUNIT\d+) : Mismatched units type .*: have '(.*)', "
        r"want '.*'"), 'unit'),
    (re.compile(r'Missing required rest frequency or wavelength'), 'rest'),
    (re.compile(r'Invalid projection: (.*)'), 'invalid'))

# The prefixes of the types of celestial axes (CTYPE)
_CELESTIAL_PREFIXES = (
    'RA--', 'DEC-', 'GLON', 'GLAT', 'ELON', 'ELAT', 'SLON', 'SLAT', 'HPLN',
    'HPLT')


def _wcs_failure(
        header: astropy.io.fits.Header, text: str
) -> tuple[str, tuple[str, ...]]:
    """
    Return the short message of a failure of wcslib (astropy's error text)
    to read the world coordinates of a header, in its keywords (e.g.
    "CTYPE1 'RA---XXX' invalid"), and the path of the option that fixes
    it, if any: of the failures it knows (see _WCSLIB_FAILURES); of
    another, its first line that is not where wcslib failed.
    """
    lines = [
        line.strip().rstrip('.') for line in text.splitlines()
        if line.strip() and not _WCSLIB_WHERE.fullmatch(line.strip())]
    for line in lines:
        for pattern, kind in _WCSLIB_FAILURES:
            if match := pattern.fullmatch(line):
                return _wcs_failure_message(header, kind, match.groups())
    return (lines[0] if lines else 'world coordinates invalid'), ()


def _wcs_failure_message(
        header: astropy.io.fits.Header, kind: str, groups: tuple[str, ...]
) -> tuple[str, tuple[str, ...]]:
    """
    Return the short message of a known failure of wcslib (see
    _wcs_failure) of its kind and the groups of its pattern.
    """
    def ctype(key: str) -> str:
        return f"{key} {str(header.get(key, '')).strip()!r}"
    ctypes = [
        key for key in header
        if re.fullmatch(r'CTYPE\d+', key)
        and str(header[key]).strip().upper().startswith(_CELESTIAL_PREFIXES)]
    if kind == 'singular':
        return 'CDELT/PC/CD matrix singular', ()
    if kind == 'projection':
        return f"{ctype(groups[1])} invalid", ()
    if kind == 'pair':
        return f"{ctype(groups[2])} mismatched ({groups[0]} expected)", ()
    if kind == 'unpaired':
        return ', '.join(map(ctype, ctypes)) + ' unpaired', ()
    if kind == 'unit':
        number = groups[0].removeprefix('CUNIT')
        return (
            f"{groups[0]} {groups[1]!r} not of "
            f"{str(header.get(f'CTYPE{number}', '')).strip()}"), ()
    if kind == 'rest':
        spectral = next((
            str(header[key]).strip() for key in header
            if re.fullmatch(r'CTYPE\d+', key)
            and _TYPES_OF_A_REST.fullmatch(str(header[key]).strip())), None)
        said = 'rest missing' if spectral is None \
            else f"{spectral} axis: rest missing"
        return said, ('spectral', 'rest')
    return f"projection invalid: {groups[0]}", ()


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
