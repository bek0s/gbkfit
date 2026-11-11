"""
The data items of the columns of tables (e.g. of profiles, or of the
moments of regions).
"""

import astropy.units
import numpy as np
from astropy.table import Table

from gbkfit.config import config_path
from gbkfit.io import read_column
from .data import Data, DataColumns


def table_item(table: Table, columns: DataColumns) -> Data:
    """
    Return a data item of the columns of a table: its values, errors and
    mask, in its unit (by default that of its column). Errors without a
    unit are in that of the column of the values (or in the unit given,
    if the column has none).
    """
    with config_path('column'):
        values = read_column(table, columns.column, columns.unit)
    unit = None if values.unit is astropy.units.dimensionless_unscaled \
        else values.unit
    error = columns.error
    if isinstance(error, bool):
        raise ValueError(
            f"the error must be a column, or one error for all the values; "
            f"it is {error}")
    if isinstance(error, int | float):
        if not error > 0:
            raise ValueError(
                f"one error for all the values must be positive; it is "
                f"{error}")
        error = np.full(values.shape, float(error))
    elif error is not None:
        with config_path('error'):
            error = read_column(table, error, None)
            if error.unit is astropy.units.dimensionless_unscaled:
                error = error.value * (
                    table[columns.column].unit or values.unit)
            try:
                error = error.to_value(values.unit)
            except astropy.units.UnitConversionError:
                raise ValueError(
                    f"the errors are in {error.unit}, but the values in "
                    f"{values.unit}") from None
    mask = None
    if columns.mask is not None:
        with config_path('mask'):
            mask = np.nan_to_num(
                read_column(table, columns.mask, None).value)
    return Data(values.value, mask, error, unit)
