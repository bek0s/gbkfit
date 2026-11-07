"""
Tables (e.g. of rotation curves): reading them from files, and their
columns with their units.
"""

import astropy.io.fits
import astropy.units as u
import numpy as np
from astropy.io.fits.connect import is_fits
from astropy.io.registry import IORegistryError
from astropy.table import Table
from astropy.table.meta import YamlParseError, get_header_from_yaml
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
    without a header are named col1, col2, etc.

    The units of the columns of a FITS or an ECSV file are read by
    gbkfit.units.parse_unit: astropy reads them by its own precedence,
    e.g. 'Jy/beam km/s' as Jy / (beam km s).

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
        If astropy cannot tell its format or read it as a table of that
        format, if the file is not FITS but the format or an HDU says it
        is, if the HDU holds no table, or if the unit of a column is not
        a unit.
    """
    # Whether the file is FITS, as astropy tells from its first bytes
    fits = False
    if format_ in (None, 'fits'):
        with get_readable_fileobj(filename, encoding='binary') as file:
            fits = is_fits('read', filename, file)
        if format_ == 'fits' and not fits:
            raise ValueError(f"{filename}: not a FITS file")
    # The units of the columns, in place of those that astropy reads:
    # those of an ECSV file are read first, so that astropy need not
    # warn of one that is not a unit before the error
    units = {}
    options = {}
    if fits:
        hdu = _table_hdu(filename, hdu)
        options.update(hdu=hdu, unit_parse_strict='silent')
    elif hdu is not None:
        raise ValueError(
            f"{filename}: an HDU is given, but the table is not read as "
            f"FITS")
    else:
        # Those of an ECSV file, whatever format reads it, e.g. 'ascii'
        units = _ecsv_units(filename)
    try:
        table = Table.read(filename, format=format_, **options)
    except IORegistryError:
        if format_ is None:
            raise ValueError(
                f"cannot tell the format of the table {filename}: give it "
                f"(format, e.g. ascii.csv)") from None
        raise ValueError(
            f"astropy reads no tables of the format {format_!r} "
            f"(e.g. 'fits', 'ascii.ecsv' or 'ascii.csv')") from None
    except (ValueError, TypeError, KeyError, IndexError) as e:
        # astropy's readers fail on some files they cannot read with
        # other errors than ValueError, e.g. a TypeError for an ECSV
        # header that lists no columns
        raise ValueError(f"cannot read the table {filename}: {e}") from e
    if fits:
        units = _fits_units(filename, hdu)
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


def _fits_units(filename: str, hdu: int) -> dict[str, u.UnitBase]:
    """
    Return the units of the columns of the table of an HDU of a FITS
    file (by index), by name: their TUNITn, as read by _unit.

    Raise ValueError if a TUNITn is not a unit.
    """
    with astropy.io.fits.open(filename) as hdus:
        return {
            column.name: _unit(filename, column.name, column.unit)
            for column in hdus[hdu].columns if column.unit}


def _ecsv_units(filename: str) -> dict[str, u.UnitBase]:
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


def _unit(filename: str, column: str, text: str) -> u.UnitBase:
    """
    Return the unit of a column of a table file, as
    gbkfit.units.parse_unit reads its text.

    Raise ValueError if the text is not a unit.
    """
    try:
        return gbkfit.units.parse_unit(text)
    except ValueError:
        raise ValueError(
            f"{filename}: the unit of the column {column!r}, {text!r}, is "
            f"not a valid unit (see the FITS standard, section 4.3); "
            f"correct it in a copy of the file") from None


def read_column(
        table: Table, name: str, unit: str | u.UnitBase | None
) -> u.Quantity:
    """
    Return a column of a table as floats with a unit.

    The values that the column masks are NaN.

    Parameters
    ----------
    table : Table
        The table.
    name : str
        The name of the column.
    unit : str, Unit or None
        The unit of the values (a text, as gbkfit.units.parse_unit reads
        it). If the column has a unit of its own, its values are
        converted to this one; if it has none, its values are taken to
        be in it. If None, the unit of the column, or dimensionless if
        it has none.

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
    try:
        values = np.ma.filled(np.ma.asarray(column, dtype=float), np.nan)
    except (TypeError, ValueError):
        raise ValueError(f"the column {name!r} is not of numbers") from None
    if unit is None:
        return values * (column.unit or u.dimensionless_unscaled)
    given = gbkfit.units.parse_unit(unit) if isinstance(unit, str) \
        else u.Unit(unit)
    if column.unit is None:
        return values * given
    try:
        return (values * column.unit).to(given)
    except u.UnitConversionError:
        target = gbkfit.units.format_unit(given) or 'dimensionless'
        raise ValueError(
            f"the column {name!r} is in "
            f"{gbkfit.units.format_unit(column.unit)!r}, which cannot be "
            f"converted to {target!r}") from None
