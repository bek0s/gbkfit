"""
Tables (e.g. of rotation curves): reading them from files, and their
columns with their units.
"""

import warnings

import astropy.io.fits
import astropy.units as u
import numpy as np
from astropy.io.fits.connect import is_fits
from astropy.io.registry import IORegistryError
from astropy.table import Table
from astropy.table.meta import YamlParseError, get_header_from_yaml
from astropy.utils.data import get_readable_fileobj
from astropy.utils.masked import Masked

import gbkfit.units
from gbkfit.config import DataFile, split_data_file


__all__ = [
    'read_column',
    'read_table'
]


def read_table(file: DataFile, format_: str | None = None) -> Table:
    """
    Read a table from a file.

    The file is of any format that astropy reads, e.g. ECSV, FITS, CSV
    or text (see astropy.table.Table.read). The columns of a text file
    without a header are named col1, col2, etc.

    The units of the columns of a FITS or an ECSV file are read by
    gbkfit.units.parse_unit: astropy reads them by its own precedence,
    e.g. 'Jy/beam km/s' as Jy / (beam km s).

    Parameters
    ----------
    file : DataFile
        The file, and of a FITS file the HDU of the table, by index or by
        name (by default, its first HDU of a table).
    format_ : str, optional
        The format of the file (e.g. 'ascii.csv'); by default, the one
        astropy tells from the file.

    Returns
    -------
    Table
        The table.

    Raises
    ------
    OSError
        If the file cannot be read (e.g. it is missing).
    ValueError
        If the format is not one that astropy reads, or cannot be told;
        if the file is not a table of its format (e.g. an ECSV header
        that lists no columns); if an HDU is given for a table not read
        as FITS, or holds no table; or if the unit of a column is not a
        unit.
    """
    filename, hdu = split_data_file(file)
    # Whether the file is FITS, as astropy tells from its first bytes
    fits = False
    if format_ in (None, 'fits'):
        with get_readable_fileobj(filename, encoding='binary') as file:
            fits = is_fits('read', filename, file)
        if format_ == 'fits' and not fits:
            raise ValueError(f"{filename}: not a FITS file")
    # The units of the columns of a FITS or an ECSV file replace those
    # astropy reads, and its warnings about those (e.g. that 'erg/s/cm2'
    # has two '/') are not shown. They are read before the table, so
    # that an ECSV header astropy fails on obscurely (e.g. one without
    # columns) gets a clear error instead.
    units = {}
    options = {}
    if fits:
        hdu = _table_hdu(filename, hdu)
        units = _fits_units(filename, hdu)
        options['hdu'] = hdu
    elif hdu is not None:
        raise ValueError(
            f"{filename}: an HDU is given, but the table is not read as "
            f"FITS")
    else:
        # Those of an ECSV file, whatever format reads it, e.g. 'ascii'
        units = _ecsv_units(filename)
    try:
        with warnings.catch_warnings():
            if units:
                warnings.simplefilter('ignore', u.UnitsWarning)
            table = Table.read(filename, format=format_, **options)
    except IORegistryError:
        if format_ is None:
            raise ValueError(
                f"{filename}: cannot tell the format of the table: give it "
                f"(format, e.g. ascii.csv)") from None
        raise ValueError(
            f"{filename}: astropy reads no tables of the format {format_!r} "
            f"(e.g. 'fits', 'ascii.ecsv' or 'ascii.csv')") from None
    except (ValueError, TypeError, KeyError, IndexError) as e:
        # astropy's readers fail on some files they cannot read with
        # other errors than ValueError, e.g. a TypeError for an ECSV
        # header that lists no columns
        raise ValueError(
            f"{filename}: cannot read the table: {e}") from None
    for name, unit in units.items():
        if name in table.colnames:
            table[name].unit = unit
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


def _fits_units(filename: str, hdu: int) -> dict[str, gbkfit.units.AnyUnit]:
    """
    Return the units of the columns of the table of an HDU of a FITS
    file (by index), by name: their TUNITn, as read by _unit.

    Raise ValueError if a TUNITn is not a unit.
    """
    with astropy.io.fits.open(filename) as hdus:
        return {
            column.name: _unit(filename, column.name, column.unit)
            for column in hdus[hdu].columns if column.unit}


def _ecsv_units(filename: str) -> dict[str, gbkfit.units.AnyUnit]:
    """
    Return the units of the columns of an ECSV file, by name: those of
    its header, as read by _unit. There are none if the file is not
    ECSV, or if its header is not YAML (astropy then says so).

    Raise ValueError if the header does not list the columns, or if a
    unit is not one.
    """
    with get_readable_fileobj(filename, encoding='binary') as file:
        if file.read(7) != b'# %ECSV':
            return {}
        file.seek(0)
        # The lines of the header, without their '#', as astropy reads
        # them (see EcsvHeader.process_lines): blank lines are skipped,
        # and the header ends at the first line that is not a comment
        lines = []
        for line in file:
            line = line.strip()
            if not line:
                continue
            if not line.startswith(b'#'):
                break
            if line[1:]:
                lines.append(line[1:])
    try:
        header = get_header_from_yaml([line.decode() for line in lines])
    except (UnicodeDecodeError, YamlParseError):
        return {}
    # The columns, which astropy reads with no check (it fails with a
    # TypeError where they are missing)
    columns = header.get('datatype') if isinstance(header, dict) else None
    if not isinstance(columns, list) or not all(
            isinstance(column, dict) and 'name' in column
            for column in columns):
        raise ValueError(
            f"{filename}: the ECSV header does not list its columns "
            f"(datatype)")
    return {
        column['name']: _unit(filename, column['name'], str(column['unit']))
        for column in columns if column.get('unit') is not None}


def _unit(filename: str, column: str, text: str) -> gbkfit.units.AnyUnit:
    """
    Return the unit of a column of a table file, as
    gbkfit.units.parse_unit reads its text.

    Raise ValueError if the text is not a unit.
    """
    try:
        return gbkfit.units.parse_unit(text)
    except ValueError as e:
        raise ValueError(
            f"{filename}: the unit of the column {column!r}: {e}; correct "
            f"it in a copy of the file") from None


def read_column(table: Table, name: str, unit: str | None) -> u.Quantity:
    """
    Return a column of a table as floats with a unit.

    The values that the column masks are NaN. The table may also be a
    QTable, whose columns with units are Quantities.

    Parameters
    ----------
    table : Table
        The table.
    name : str
        The name of the column.
    unit : str or None
        The unit of the values, as gbkfit.units.parse_unit reads it. If
        the column has a unit of its own, its values are converted to
        this one; if it has none, its values are taken to be in it. If
        None, the unit of the column, or dimensionless if it has none.

    Returns
    -------
    Quantity
        The values.

    Raises
    ------
    ValueError
        If the table has no such column, its values are not real
        numbers, the unit is not a unit, or the column is in a unit of
        another kind.
    """
    if name not in table.colnames:
        raise ValueError(
            f"the table has no column {name!r}; its columns are "
            f"{table.colnames}")
    column = table[name]
    if np.iscomplexobj(column):
        raise ValueError(f"the column {name!r} is of complex numbers")
    # The values (NaN where masked) and the unit of the column. In a
    # QTable, a column with a unit is a Quantity, masked the astropy
    # way; in a Table, a column holds plain values, masked the numpy
    # way, and its unit apart.
    if isinstance(column, u.Quantity):
        if isinstance(column, Masked):
            column = column.filled(np.nan)
        values, column_unit = column.value.astype(float), column.unit
    else:
        try:
            values = np.ma.filled(
                np.ma.asarray(column, dtype=float), np.nan)
        except (TypeError, ValueError):
            raise ValueError(
                f"the column {name!r} is not of numbers") from None
        column_unit = column.unit
    if unit is None:
        return values * (column_unit or u.dimensionless_unscaled)
    given = gbkfit.units.parse_unit(unit)
    if column_unit is None:
        return values * given
    try:
        return (values * column_unit).to(given)
    except u.UnitConversionError:
        source, target = (
            gbkfit.units.format_unit(x) or 'dimensionless'
            for x in (column_unit, given))
        raise ValueError(
            f"the column {name!r} is in {source!r}, which cannot be "
            f"converted to {target!r}") from None
