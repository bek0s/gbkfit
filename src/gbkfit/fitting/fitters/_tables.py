"""
The tables of parameters whose number of rows a fitter varies (see
ParamTable): which free elements make their rows, and the free values of
the rows a fitter uses.
"""

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np

from gbkfit.params import ParamSpace, param_tables
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'TableLayout',
    'VariedTable',
    'table_layout'
]


@dataclasses.dataclass(frozen=True)
class VariedTable:
    """
    A table of parameters whose number of rows a fitter varies: its name
    (that of its parameters without their column, e.g. 'pt1' of
    'pt1_r'), the names of its free columns, the free elements of the
    rows it varies (rows, columns: their indices in the free values), the
    values of an unused row (the empty row in its columns, the median of
    the prior in the others), and the number of its rows that are always
    used.
    """
    name: str
    columns: tuple[str, ...]
    elements: np.ndarray
    unused: np.ndarray
    always: int

    def rows(self) -> int:
        """Return the number of rows it varies."""
        return len(self.elements)


@dataclasses.dataclass(frozen=True)
class TableLayout:
    """
    How the free values are made of the elements that are always used
    (their indices) and of the rows of the tables whose number of rows a
    fitter varies.
    """
    ndim: int
    fixed: np.ndarray
    tables: tuple[VariedTable, ...]

    def place(
            self,
            fixed: np.ndarray,
            rows: Sequence[np.ndarray],
            fill: float
    ) -> np.ndarray:
        """
        Return the vector of the free elements: the given values of those
        that are always used, and of the rows of each table (rows,
        columns) in its first rows; fill in its other rows.
        """
        values = np.full(self.ndim, fill)
        values[self.fixed] = fixed
        for table, used in zip(self.tables, rows):
            values[table.elements[:len(used)]] = used
        return values

    def unused(self, counts: Sequence[int]) -> np.ndarray:
        """
        Return which free elements are in rows not used, of the given
        number of rows used of each table.
        """
        unused = np.zeros(self.ndim, dtype=bool)
        for table, count in zip(self.tables, counts):
            unused[table.elements[count:]] = True
        return unused

    def unused_values(self) -> np.ndarray:
        """
        Return the values of the free elements in rows not used (NaN for
        the elements always used).
        """
        values = np.full(self.ndim, np.nan)
        for table in self.tables:
            values[table.elements] = table.unused
        return values


def table_layout(
        params: ParamSpace,
        names: Sequence[str],
        medians: Sequence[float],
        min_rows: Mapping[str, int]
) -> TableLayout:
    """
    Return the tables of parameters whose number of rows a fitter can
    vary, and the other free elements (see TableLayout).

    Only the rows of a table that share all their settings are
    interchangeable, and only they can be dropped: a row with settings of
    its own (see ParamSpace.own_settings: e.g. 'pt1_r[0]') is always
    used, and so are the rows whose empty columns are not free (e.g.
    blobs of a fixed amplitude), and all those of a table of a parameter
    with a mode. Of the others, the first are always used too, as many as
    min_rows asks of the table beyond those.

    Parameters
    ----------
    params : ParamSpace
        The parameters.
    names : Sequence of str
        The names of the free elements, in order.
    medians : Sequence of float
        The median of the prior of each free element (the value of an
        unused row in its columns that are not empty).
    min_rows : Mapping of str to int
        The least rows used of tables, by the name of the table.

    Raises
    ------
    ConfigError
        If min_rows names no table whose rows can vary, or asks for more
        rows than a table has.
    """
    index = {name: i for i, name in enumerate(names)}
    own = params.own_settings()
    tables = []
    named = set()
    for table, columns in param_tables(params.pdescs()):
        if any(name in params.modes() for name in columns.values()):
            continue
        size = params.pdescs()[next(iter(columns.values()))].size()
        empty = table.empty()
        # (as ParamSpace names the elements)
        cells = {c: [f'{name}[{i}]' for i in range(size)]
                 for c, name in columns.items()}
        free = tuple(c for c in columns if cells[c][0] in index)
        varied = [
            i for i in range(size)
            if not any(cells[c][i] in own for c in columns)
            and all(cells[c][i] in index for c in empty)
            and tuple(c for c in columns if cells[c][i] in index) == free]
        if not varied or not free:
            continue
        name = next(iter(columns.values()))[:-len(next(iter(columns))) - 1]
        named.add(name)
        least = min_rows.get(name, 0)
        if least > size:
            raise ConfigError(
                f"option 'min_rows' asks for {least} rows of table "
                f"'{name}', which has {size}")
        # (the rows always used of those varied: the fixed elements have
        # them)
        pinned = max(0, least - (size - len(varied)))
        if pinned == len(varied):
            continue
        elements = np.array(
            [[index[cells[c][i]] for c in free] for i in varied[pinned:]])
        unused = np.array([
            empty[c] if c in empty else medians[elements[0, j]]
            for j, c in enumerate(free)])
        tables.append(VariedTable(
            name, free, elements, unused, size - len(elements)))
    if unknown := set(min_rows) - named:
        raise ConfigError(
            f"option 'min_rows' names tables whose rows cannot vary: "
            f"{sorted(unknown)}; those that can: {sorted(named)}")
    in_tables = {int(i) for t in tables for i in t.elements.ravel()}
    fixed = np.array(
        [i for i in range(len(names)) if i not in in_tables], dtype=int)
    return TableLayout(len(names), fixed, tuple(tables))
