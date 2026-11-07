"""
Tables of configurations (e.g. rotation curves): reading them, by
astropy, and their columns, with units.
"""

from typing import Any

import astropy.io.fits
import astropy.units as u
import numpy as np
from astropy.io.fits.connect import is_fits
from astropy.io.registry import IORegistryError
from astropy.table import Table

import gbkfit.units
from gbkfit.config import parse_file


__all__ = [
    'read_column',
    'read_table'
]


def read_table(
        file: Any, format_: str | None = None, hdu: int | str | None = None
) -> Table:
    """
    Read a table from a file: any format that astropy reads (see
    Table.read), e.g. ECSV, FITS, CSV or text. The columns of a text file
    without a header are named col1, col2, etc.

    Parameters
    ----------
    file : Any
        The file, as a file option (see parse_file).
    format_ : str, optional
        The format of the file (e.g. 'ascii.csv'); by default, the one
        astropy tells from the file.
    hdu : int or str, optional
        The HDU of a FITS file; by default, its first HDU of a table.

    Returns
    -------
    Table
        The table.

    Raises
    ------
    OSError
        If the file cannot be read (e.g. it is missing).
    ValueError
        If astropy cannot tell its format, or cannot read it as a table
        of that format.
    """
    filename = parse_file(file)[0]
    options = {} if hdu is None else dict(hdu=hdu)
    try:
        table = Table.read(filename, format=format_, **options)
    except IORegistryError:
        raise ValueError(
            f"cannot tell the format of the table {filename}: give it "
            f"(format, e.g. ascii.csv)") from None
    except ValueError as e:
        raise ValueError(f"cannot read the table {filename}: {e}") from e
    if format_ == 'fits' or format_ is None and is_fits(
            'read', filename, None):
        _read_fits_units(table, filename, hdu)
    return table


def _read_fits_units(
        table: Table, filename: str, hdu: int | str | None
) -> None:
    """
    Give the columns of a table of a FITS file (of the HDU, or else of its
    first HDU of a table, as astropy reads it) the units of their TUNITn
    as gbkfit.units.parse_unit reads them: astropy reads them by its own
    precedence, e.g. 'Jy/beam km/s' as Jy / (beam km s).

    Raise ValueError if a TUNITn is not a unit.
    """
    with astropy.io.fits.open(filename) as hdus:
        if hdu is None:
            hdu = next(
                i for i, item in enumerate(hdus) if isinstance(item, (
                    astropy.io.fits.BinTableHDU, astropy.io.fits.TableHDU)))
        for column in hdus[hdu].columns:
            if not column.unit:
                continue
            try:
                table[column.name].unit = gbkfit.units.parse_unit(
                    column.unit)
            except ValueError:
                raise ValueError(
                    f"{filename}: the unit of the column {column.name!r}, "
                    f"{column.unit!r}, is not a valid unit (see the FITS "
                    f"standard, section 4.3); correct it in a copy of the "
                    f"file"
                ) from None


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
    ValueError
        If the table has no such column, its values are not numbers, the
        unit is not a unit, or the unit of the column cannot be converted
        to it.
    """
    if name not in table.colnames:
        raise ValueError(
            f"the table has no column {name!r}; its columns are "
            f"{table.colnames}")
    column = table[name]
    try:
        values = np.ma.filled(np.ma.asarray(column, dtype=float), np.nan)
    except (TypeError, ValueError):
        raise ValueError(f"the column {name!r} is not of numbers") from None
    if unit is None:
        return values * (column.unit or u.dimensionless_unscaled)
    given = gbkfit.units.parse_unit(unit)
    if column.unit is None:
        return values * given
    try:
        return (values * column.unit).to(given)
    except u.UnitConversionError:
        raise ValueError(
            f"the column {name!r} is in "
            f"{gbkfit.units.format_unit(column.unit)!r}, which cannot be "
            f"converted to {unit!r}") from None
