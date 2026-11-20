"""
The steps and the references of the axes of FITS headers that GBKFIT
reads (see reference_errors): of the axes on the sky and of the spectral
axis, CDELTn (or a CD matrix), CRPIXn and CRVALn. FITS gives each a
default where it is missing (a step of 1, and 0), which GBKFIT never
takes: a missing one is an error at the option of a dataset that states
it instead (step, rpix and rval, and those of its spectral axis), unless
the dataset states it.
"""

import re
from collections.abc import Collection

import astropy.io.fits

from gbkfit.config import InputError
from .headers import _CELESTIAL_PREFIXES
from .spectral import spectral_axis_index, unstated_spectral_axis


__all__ = [
    'reference_errors'
]


# The keywords of a CD matrix
_CD_KEYWORDS = re.compile(r'CD\d+_\d+')


def reference_errors(
        source: str,
        header: astropy.io.fits.Header,
        shape: tuple[int, ...],
        stated: Collection[str] = ()
) -> list[InputError]:
    """
    Return the errors of the steps and the references of the axes on the
    sky and of the spectral axis of a header (of an array of the given
    shape, numpy order) that it leaves out, each at the option of a
    dataset that states it instead, unless the dataset states it (stated:
    the names of those options it states: 'step', 'rpix', 'rval', or
    those of its spectral axis, 'spectral.step', ...):

    - a missing CDELTn (without a CD matrix), or with a CD matrix, a
      CDELTn too that is neither 1 nor CDn_n (FITS, and wcslib, leave it
      out; other tools read it) or no CDn_j of the axis: of the option
      step;
    - a missing CRPIXn: of rpix, and a missing CRVALn: of rval; but of
      rval and of rpix, if the other is stated alone: one reference
      stated alone takes the other from the header (see
      coords_from_header), which then needs both.

    Axes of length 1, or beyond those of the array, are not looked at:
    reading leaves them out. A step stated is that of the grid of the
    dataset; the header gives the direction of its axis (that of FITS
    without CDELTn).
    """
    naxis = int(header.get('WCSAXES', header.get('NAXIS', 0)) or 0)

    def is_own(axis: int) -> bool:
        return axis < len(shape) and shape[len(shape) - 1 - axis] > 1
    sky = [
        axis for axis in range(naxis)
        if is_own(axis) and str(header.get(f'CTYPE{axis + 1}', ''))
        .strip().upper().startswith(_CELESTIAL_PREFIXES)]
    spectral = _spectral_axis(header, shape)
    groups = [(sky, '')]
    if spectral is not None and is_own(spectral):
        groups.append(([spectral], 'spectral.'))
    errors = []
    for axes, prefix in groups:
        for axis in axes:
            errors += _axis_errors(source, header, axis + 1, prefix, stated)
    return errors


def _spectral_axis(
        header: astropy.io.fits.Header, shape: tuple[int, ...]
) -> int | None:
    """
    Return the spectral axis of a header (FITS order, from 0): of its
    type, or of a type FITS does not define, or of none (see
    unstated_spectral_axis); None if it has none.
    """
    axis = spectral_axis_index(header, shape)
    if axis is None:
        found = unstated_spectral_axis(header, shape)
        axis = None if found is None else found[0]
    return axis


def _axis_errors(
        source: str,
        header: astropy.io.fits.Header,
        n: int,
        prefix: str,
        stated: Collection[str]
) -> list[InputError]:
    """
    Return the errors of the step and the references of the axis n (FITS
    order, from 1) of a header (see reference_errors), at the options of
    a prefix ('' for the sky, 'spectral.' for the spectral axis).
    """
    def error(message: str, option: str) -> InputError:
        return InputError(
            f"{source}: {message}", tuple(f'{prefix}{option}'.split('.')))

    def is_stated(option: str) -> bool:
        return f'{prefix}{option}' in stated
    errors = []
    cd = [key for key in header if _CD_KEYWORDS.fullmatch(key)]
    cdelt = header.get(f'CDELT{n}')
    diagonal = header.get(f'CD{n}_{n}', 0.0)
    if is_stated('step'):
        pass
    elif cd and cdelt is not None and cdelt not in (1, diagonal):
        # (wcslib leaves CDELTn out, as FITS says; other tools read it: of
        # 1, or of the diagonal of the matrix, they agree)
        errors.append(error(
            f"CDELT{n} {cdelt:g} with CD{n}_{n} {diagonal:g}", 'step'))
    elif cd and cdelt is None \
            and not any(key.startswith(f'CD{n}_') for key in cd):
        errors.append(error(f"CD{n}_{n} missing", 'step'))
    elif not cd and cdelt is None:
        errors.append(error(f"CDELT{n} missing", 'step'))
    both = is_stated('rpix') and is_stated('rval')
    for key, option, other in (
            (f'CRPIX{n}', 'rpix', 'rval'), (f'CRVAL{n}', 'rval', 'rpix')):
        if key not in header and not both:
            errors.append(error(
                f"{key} missing", other if is_stated(option) else option))
    return errors
