"""
What the options of a dataset state of the headers of its files (see
statement_errors): of their spectral axis, its type, unit and the
convention of its velocities, and its rest; the steps and references of
their axes; and the unit of their values.
"""

from collections.abc import Collection

import astropy.io.fits

from gbkfit.config import InputError
from gbkfit.grid import Rest, make_rest
from .headers import _option_error, data_unit
from .references import reference_errors
from .spectral import _statement_errors


__all__ = [
    'blocks_reading',
    'statement_errors'
]


# The options of the steps and references of the axes on the sky (see
# blocks_reading)
_SKY_OPTIONS = (['step'], ['rpix'], ['rval'])


def statement_errors(
        source: str,
        header: astropy.io.fits.Header,
        shape: tuple[int, ...],
        *,
        rest: Rest = None,
        velocity: str | None = None,
        spectral_type: str | None = None,
        spectral_unit: str | None = None,
        unit: str | None = None,
        stated: Collection[str] = ()
) -> list[InputError]:
    """
    Return the errors of what a header (of an array of the given shape,
    numpy order) needs stated, or of what is stated of it that is not
    valid, each at the option of a dataset that fixes it: together, of
    each check that does not depend on another (see describe_fits).

    - Its spectral axis: its type, its unit and the convention of its
      velocities, where its header does not say them (see
      _statement_errors); and the rest given, if it is not one.
    - The steps and references of its axes that it leaves out (see
      reference_errors), but those of the options stated (the names of
      the options of the grid of a dataset, e.g. 'step' or
      'spectral.rval').
    - The unit of its values (see data_unit): the one given, or else
      BUNIT.
    """
    errors = _statement_errors(
        source, header, shape, spectral_type, velocity, spectral_unit)
    try:
        make_rest(rest)
    except ValueError:
        errors.append(
            _option_error(f"rest {rest!r} invalid", 'spectral', 'rest'))
    errors += reference_errors(source, header, shape, stated)
    try:
        data_unit(source, header, unit)
    except InputError as e:
        errors.append(e)
    return errors


def blocks_reading(error: InputError, sky: bool = False) -> bool:
    """
    Return whether an error of statement_errors leaves the world
    coordinates of its header unread (they would be read of what FITS
    takes for what is missing, or not at all): all but that of the unit of
    its values; of those of the axes on the sky alone (sky, see
    describe_fits), those of their steps and references.
    """
    if sky:
        return error.config_path in _SKY_OPTIONS
    return error.config_path != ['unit']
