"""
Tables of configurations (e.g. rotation curves): reading them, by
astropy, and their columns, with units.
"""

from typing import Any

import astropy.units as u
import numpy as np
from astropy.io.registry import IORegistryError
from astropy.table import Table

from gbkfit.utils import parseutils, unitutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'read_column',
    'read_table'
]


def read_table(file: Any, format_: str | None = None) -> Table:
    """
    Read a table from a file: any format that astropy reads (see
    Table.read), e.g. ECSV, FITS, CSV or text. The columns of a text file
    without a header are named col1, col2, etc.

    Parameters
    ----------
    file : Any
        The file, as a file option (see parseutils.parse_file).
    format_ : str, optional
        The format of the file (e.g. 'ascii.csv'); by default, the one
        astropy tells from the file.

    Returns
    -------
    Table
        The table.

    Raises
    ------
    ConfigError
        If the file cannot be read, or astropy cannot tell its format.
    """
    filename = parseutils.parse_file(file)[0]
    try:
        return Table.read(filename, format=format_)
    except IORegistryError:
        raise ConfigError(
            f"cannot tell the format of the table {filename}: give it "
            f"(format, e.g. ascii.csv)") from None
    except (OSError, ValueError) as e:
        raise ConfigError(f"cannot read the table {filename}: {e}") from e


def read_column(table: Table, name: Any, unit: str | None) -> u.Quantity:
    """
    Return a column of a table, by name, as floats with a unit, and NaN
    where it is masked.

    Parameters
    ----------
    table : Table
        The table.
    name : Any
        The name of the column.
    unit : str or None
        The unit of the values. If the column has a unit of its own, its
        values are converted to this one; if it has none, its values are
        taken to be in it. If None, the unit of the column, or
        dimensionless if it has none.

    Returns
    -------
    Quantity
        The values.

    Raises
    ------
    ConfigError
        If the table has no such column, its values are not numbers, the
        unit is not a unit, or the unit of the column cannot be converted
        to it.
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
    if unit is None:
        return values * (column.unit or u.dimensionless_unscaled)
    given = unitutils.parse_unit(unit)
    if column.unit is None:
        return values * given
    try:
        return (values * column.unit).to(given)
    except u.UnitConversionError:
        raise ConfigError(
            f"the column {name!r} is in "
            f"{unitutils.format_unit(column.unit)!r}, which cannot be "
            f"converted to {unit!r}") from None
