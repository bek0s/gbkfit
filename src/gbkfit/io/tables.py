"""
Tables (e.g. of rotation curves): reading them from files, and their
columns with their units.
"""

from typing import Any

import astropy.io.fits
import astropy.units as u
import numpy as np
from astropy.io.fits.connect import is_fits
from astropy.io.registry import IORegistryError
from astropy.table import Table
from astropy.utils.data import get_readable_fileobj

import gbkfit.units


__all__ = [
    'read_column',
    'read_table'
]


def read_table(
        filename: str, format_: str | None = None,
        hdu: int | str | None = None
) -> Table:
    """
    Read a table from a file.

    The file is of any format that astropy reads, e.g. ECSV, FITS, CSV
    or text (see astropy.table.Table.read). The columns of a text file
    without a header are named col1, col2, etc. The units of a FITS
    file (TUNITn) are read by gbkfit.units.parse_unit.

    Parameters
    ----------
    filename : str
        The path of the file.
    format_ : str, optional
        The format of the file (e.g. 'ascii.csv'); by default, the one
        astropy tells from the file.
    hdu : int or str, optional
        The HDU of a FITS file, by index or by name; by default, its
        first HDU of a table.

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
        of that format, or if the HDU is not that of a table in a FITS
        file.
    """
    # Whether the file is FITS: as the format says, or else as astropy
    # tells from its first bytes
    if format_ is None:
        with get_readable_fileobj(filename, encoding='binary') as file:
            fits = is_fits('read', filename, file)
    else:
        fits = format_ == 'fits'
    options = {}
    if fits:
        # The units are read below, so astropy does not warn of those
        # that it cannot read
        hdu = _table_hdu(filename, hdu)
        options.update(hdu=hdu, unit_parse_strict='silent')
    elif hdu is not None:
        raise ValueError(
            f"{filename}: an HDU is given, but the file is not FITS")
    try:
        table = Table.read(filename, format=format_, **options)
    except IORegistryError:
        raise ValueError(
            f"cannot tell the format of the table {filename}: give it "
            f"(format, e.g. ascii.csv)") from None
    except ValueError as e:
        raise ValueError(f"cannot read the table {filename}: {e}") from e
    if fits:
        _read_fits_units(table, filename, hdu)
    return table


def _table_hdu(filename: str, hdu: int | str | None) -> int:
    """
    Return the index of the HDU of a table in a FITS file: the HDU
    given (by index or by name), or else the first HDU of a table.

    Raise ValueError if there is no table in the HDU given (also if it
    is missing), or none in the file.
    """
    with astropy.io.fits.open(filename) as hdus:
        tables = [
            i for i, item in enumerate(hdus) if isinstance(
                item, (astropy.io.fits.BinTableHDU, astropy.io.fits.TableHDU))]
        if hdu is None:
            if not tables:
                raise ValueError(f"{filename}: there is no table in the file")
            return tables[0]
        # astropy finds an HDU by name, and returns an index unchecked:
        # one out of range is not one of the tables, below
        try:
            index = hdus.index_of(hdu)
        except KeyError:
            index = None
    if index not in tables:
        raise ValueError(
            f"{filename}: there is no table in HDU {hdu!r}; the tables are "
            f"in the HDUs {tables}")
    return index


def _read_fits_units(table: Table, filename: str, hdu: int) -> None:
    """
    Set the units of the columns of a table read from the HDU of a FITS
    file (by index) to their TUNITn, as gbkfit.units.parse_unit reads
    them: astropy reads them by its own precedence, e.g. 'Jy/beam km/s'
    as Jy / (beam km s).

    Raise ValueError if a TUNITn is not a unit.
    """
    with astropy.io.fits.open(filename) as hdus:
        for column in hdus[hdu].columns:
            if not column.unit:
                continue
            try:
                unit = gbkfit.units.parse_unit(column.unit)
            except ValueError:
                raise ValueError(
                    f"{filename}: the unit of the column {column.name!r}, "
                    f"{column.unit!r}, is not a valid unit "
                    f"(see the FITS standard, section 4.3); correct it in "
                    f"a copy of the file") from None
            table[column.name].unit = unit


def read_column(table: Table, name: Any, unit: str | None) -> u.Quantity:
    """
    Return a column of a table as floats with a unit.

    The values that the column masks are NaN.

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
        unit is not a unit, or the column is in a unit of another kind.
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
