import os.path
from collections.abc import Mapping, Sequence
from typing import Any

import astropy.units as u
import numpy as np
from astropy.table import Table

from gbkfit.utils import parseutils, unitutils
from gbkfit.utils.parseutils import ConfigError
from . import _resampling
from .base import Dataset
from .data import Data


__all__ = [
    'DatasetProfile',
    'check_profile_radii'
]


# The axes of its blocks of residuals (see resample_blocks)
_BLOCK_AXES = ('the radii',)

# The options of a data item in its configuration (see load)
_ITEM_OPTIONS = ('column', 'error', 'mask', 'unit')


def check_profile_radii(
        radii: u.Quantity, distance: u.Quantity | None
) -> tuple[u.Quantity, u.Quantity | None]:
    """
    Return the radii of a profile (see DatasetProfile) and their distance,
    as 1d arrays of floats and a length; or raise ConfigError if the radii
    are not angles or lengths, finite and at least 0, or the distance is
    not a positive length, or is given for radii in angles.
    """
    if not isinstance(radii, u.Quantity) or radii.ndim != 1 \
            or not (radii.unit.is_equivalent(u.arcsec)
                    or radii.unit.is_equivalent(u.kpc)):
        raise ConfigError(
            f"the radii are a vector of angles (e.g. arcsec) or lengths "
            f"(e.g. kpc), with their unit; they are {radii!r}")
    radii = radii.astype(float)
    if not np.all(np.isfinite(radii) & (radii >= 0)):
        raise ConfigError(
            f"the radii must be finite and at least 0; they are {radii}")
    if distance is not None:
        if radii.unit.is_equivalent(u.arcsec):
            raise ConfigError(
                "a distance is that of radii in lengths, measured at it; "
                "the radii are angles")
        if not isinstance(distance, u.Quantity) or distance.size != 1 \
                or not distance.unit.is_equivalent(u.Mpc) \
                or not distance > 0:
            raise ConfigError(
                f"the distance must be a positive length; it is "
                f"{distance!r}")
    return radii, distance


class DatasetProfile(Dataset):
    """
    Values of functions of a galaxy at radii (see ObservationProfile; e.g.
    its rotation curve): a table of radii, and of the values of each
    function, its data items, named after the function (e.g. vcirc).

    The radii are angles (e.g. arcsec), or lengths (e.g. kpc): measured at
    a distance, if given (their angles are the lengths over it), or
    physical (e.g. galactocentric radii in the Milky Way).

    Its configuration has the file of the table (any that astropy reads,
    see Table.read: e.g. ECSV, FITS, CSV or text; the columns of a text
    file without a header are named col1, col2, ...), its format if
    astropy cannot tell it, the column of the radii (radius: {column,
    unit}; by default the unit of the column), the distance (Mpc), if any,
    and each data item, by name: {column, error (a column, or one error
    for all the values), mask (a column: 0 for the values to leave out),
    unit (by default that of the column)}.

    Parameters
    ----------
    items : Mapping of str to Data
        The values of each function, a vector of one value per radius; at
        least one.
    radii : Quantity
        The radii.
    distance : Quantity, optional
        The distance of radii in lengths that were measured at one.

    Raises
    ------
    ConfigError
        If the items do not have one value per radius, or the radii or the
        distance are invalid (see check_profile_radii).
    """

    ndim = 1

    @staticmethod
    def type() -> str:
        return 'profile'

    @classmethod
    def load(
            cls, info: dict[str, Any], prefix: str = ''
    ) -> 'DatasetProfile':
        info = dict(info)
        options = {
            key: info.pop(key) for key in ('file', 'format', 'radius',
                                           'distance') if key in info}
        options = parseutils.parse_options(
            options, required={'file', 'radius'},
            optional={'format', 'distance'})
        with parseutils.config_path('file'):
            filename, _ = parseutils.parse_file(options['file'])
            try:
                table = Table.read(prefix + filename, format=options.get(
                    'format'))
            except (OSError, ValueError) as e:
                raise ConfigError(
                    f"cannot read the table {filename}: {e}") from e
        with parseutils.config_path('radius'):
            radius = parseutils.parse_options(
                options['radius'], required={'column'}, optional={'unit'})
            radii = _column(table, radius['column'], radius.get('unit'))
            if radii.unit is u.dimensionless_unscaled:
                raise ConfigError(
                    "the radii need their unit (e.g. unit: kpc), which their "
                    "column does not have")
        distance = options.get('distance')
        items = {}
        for key, item in info.items():
            with parseutils.config_path(key):
                items[key] = _item(table, item)
        return cls(
            items, radii, None if distance is None else distance * u.Mpc)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Dump the dataset to its configuration, writing its table to an
        ECSV file ('{prefix}profile.ecsv'): the radii ('radius'), and of
        each data item, its values ('{item}'), errors ('{item}_error') and
        mask ('{item}_mask').
        """
        filename = f'{prefix}profile.ecsv'
        table = Table(dict(radius=self._radii))
        info = dict(
            type=self.type(),
            file=filename if dump_path else os.path.basename(filename),
            radius=dict(column='radius'))
        if self._distance is not None:
            info['distance'] = float(self._distance.to_value(u.Mpc))
        for key, item in self.items():
            unit = item.unit() or u.dimensionless_unscaled
            table[key] = item.data() * unit
            info[key] = dict(column=key, mask=f'{key}_mask')
            if item.error() is not None:
                table[f'{key}_error'] = item.error() * unit
                info[key]['error'] = f'{key}_error'
            table[f'{key}_mask'] = item.mask()
        table.write(filename, format='ascii.ecsv', overwrite=overwrite)
        return info

    def __init__(
            self,
            items: Mapping[str, Data],
            radii: u.Quantity,
            distance: u.Quantity | None = None
    ):
        super().__init__(items)
        radii, distance = check_profile_radii(radii, distance)
        if self.shape() != radii.shape:
            raise ConfigError(
                f"the data items have {self.shape()[0]} values, but there "
                f"are {radii.size} radii")
        self._radii = radii
        self._distance = distance

    def radii(self) -> u.Quantity:
        """Return the radii."""
        return self._radii

    def distance(self) -> u.Quantity | None:
        """
        Return the distance of radii in lengths that were measured at one,
        if given.
        """
        return self._distance

    def check_blocks(self, size: Sequence[float]) -> None:
        """
        Raise ConfigError unless the size is of blocks of radii: a positive
        integer for each of the radii (see resample_blocks).
        """
        _resampling.block_size(size, _BLOCK_AXES)

    def resample_blocks(
            self,
            residuals: Mapping[str, np.ndarray],
            size: Sequence[float],
            rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """
        Return residuals of its data redrawn in blocks of radii (see
        Dataset.resample_blocks), of the size (radii): runs of consecutive
        radii, in their order (e.g. of 3, for the points of a rotation
        curve closer than the beam that measured them).
        """
        return _resampling.resample_blocks(
            residuals, size, _BLOCK_AXES, rng)


def _column(
        table: Table, name: Any, unit: str | None
) -> u.Quantity:
    """
    Return a column of a table, by name, as floats with a unit: the unit
    given, or else that of the column (none if it has none), and NaN where
    it is masked. Raise ConfigError if the table has no such column, or
    its values are not numbers.
    """
    if name not in table.colnames:
        raise ConfigError(
            f"the table has no column {name!r}; its columns are "
            f"{table.colnames}")
    column = table[name]
    try:
        values = np.ma.filled(
            np.ma.asarray(column, dtype=float), np.nan)
    except (TypeError, ValueError):
        raise ConfigError(f"the column {name!r} is not of numbers") from None
    if unit is not None:
        return values * unitutils.parse_unit(unit)
    return values * (column.unit or u.dimensionless_unscaled)


def _item(table: Table, info: Any) -> Data:
    """
    Return a data item of a table from its configuration (see
    DatasetProfile): its values, errors and mask, in its unit.
    """
    if not isinstance(info, Mapping):
        raise ConfigError(
            f"a data item has its column ('column'), and optionally its "
            f"error ('error': a column, or one error), mask ('mask': a "
            f"column) and unit ('unit'); it is {info!r}")
    info = parseutils.parse_options(
        info, required={'column'}, optional=set(_ITEM_OPTIONS) - {'column'})
    values = _column(table, info['column'], info.get('unit'))
    unit = None if values.unit is u.dimensionless_unscaled else values.unit
    error = info.get('error')
    if isinstance(error, (int, float)):
        error = np.full(values.shape, float(error))
    elif error is not None:
        with parseutils.config_path('error'):
            error = _column(table, error, None)
            if error.unit is not u.dimensionless_unscaled:
                try:
                    error = error.to(values.unit)
                except u.UnitConversionError:
                    raise ConfigError(
                        f"the errors are in {error.unit}, but the values in "
                        f"{values.unit}") from None
            error = error.value
    mask = None
    if info.get('mask') is not None:
        with parseutils.config_path('mask'):
            mask = np.nan_to_num(_column(table, info['mask'], None).value)
    return Data(values.value, mask, error, unit)
