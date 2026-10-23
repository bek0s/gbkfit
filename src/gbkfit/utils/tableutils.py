"""
Tables of configurations (e.g. rotation curves): reading them, by
astropy, and their columns, with units.
"""

from typing import Any

import astropy.units as u
import numpy as np
from astropy.table import Table

from gbkfit.utils import parseutils, unitutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'read_column',
    'read_table'
]


def read_table(
        file: Any, format_: str | None = None, prefix: str = ''
) -> Table:
    """
    Read a table of a file option (see parseutils.parse_file): any that
    astropy reads (see Table.read: e.g. ECSV, FITS, CSV or text; the
    columns of a text file without a header are named col1, col2, ...),
    in the given format, or one astropy tells; the prefix is prepended
    to its filename (as to the files of datasets, see fits_file).

    Raises
    ------
    ConfigError
        If the file cannot be read.
    """
    filename = prefix + parseutils.parse_file(file)[0]
    try:
        return Table.read(filename, format=format_)
    except (OSError, ValueError) as e:
        raise ConfigError(f"cannot read the table {filename}: {e}") from e


def read_column(table: Table, name: Any, unit: str | None) -> u.Quantity:
    """
    Return a column of a table, by name, as floats with a unit: the unit
    given, or else that of the column (dimensionless if it has none), and
    NaN where it is masked.

    Raises
    ------
    ConfigError
        If the table has no such column, its values are not numbers, or
        the unit is not a unit.
    """
    if name not in table.colnames:
        raise ConfigError(
            f"the table has no column {name!r}; its columns are "
            f"{table.colnames}")
    column = table[name]
    try:
        values = np.ma.filled(np.ma.asarray(column, dtype=float), np.nan)
    except (TypeError, ValueError):
        raise ConfigError(f"the column {name!r} is not of numbers") from None
    if unit is not None:
        return values * unitutils.parse_unit(unit)
    return values * (column.unit or u.dimensionless_unscaled)
