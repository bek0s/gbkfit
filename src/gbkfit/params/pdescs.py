"""
The descriptions of parameters: their size, meaning and valid range.

A description has no name: the name of a parameter is its key in the
dict of parameters it belongs to. Models that combine others give the
parameters of each a prefix (e.g. 'vt' of a trait becomes 'hi_vpt_vt' of
a model), and their dicts share the descriptions.

Vector parameters of the same size can be the columns of a table (see
ParamTable), whose rows are e.g. the blobs of a mixture trait.
"""

import abc
import copy
import dataclasses
from collections.abc import Mapping
from typing import Annotated, Any, Self, TypeAlias

import astropy.units
import numpy as np
from annotated_types import Ge

from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError, check_options


__all__ = [
    'ParamColumn',
    'ParamDesc',
    'ParamScalarDesc',
    'ParamTable',
    'ParamVectorDesc',
    'param_tables',
    'ParamUnit',
    'make_param_unit',
    'pdesc_parser',
    'pdescs_with_units',
    'load_pdescs_dict',
    'dump_pdescs_dict'
]


# The unit of the values of a parameter: an astropy unit, or a unit of
# their logarithm (e.g. dex(solMass), see ParamDesc)
ParamUnit: TypeAlias = astropy.units.UnitBase | astropy.units.FunctionUnitBase


class ParamTable:
    """
    A table of parameters: vector parameters of the same size, its
    columns (see ParamColumn), whose elements of one index form a row
    (e.g. a blob of a mixture trait: its radius, azimuth, amplitude...),
    and its empty row: the values of columns with which a row adds nothing
    to the model (e.g. amplitude 0, with a valid size). Every fitter fits
    all its rows, but one that varies their number (see FitterDNest4) uses
    some and writes the empty row into the others. The descriptions of
    its columns share it: a table is told apart from another by its
    identity, whatever the names its columns get in the models that
    combine others.

    Parameters
    ----------
    desc : str
        What a row is, for people (e.g. "blob").
    empty : Mapping[str, float]
        The empty row: values of columns, by column name.
    """

    def __init__(self, desc: str, empty: Mapping[str, float]):
        self._desc = desc
        self._empty = {name: float(value) for name, value in empty.items()}

    def desc(self) -> str:
        """Return what a row is."""
        return self._desc

    def empty(self) -> dict[str, float]:
        """Return the empty row: values of columns, by column name."""
        return dict(self._empty)


@dataclasses.dataclass(frozen=True)
class ParamColumn:
    """
    A column of a table of parameters (see ParamTable): the table, and the
    name of the column in it.
    """
    table: ParamTable
    name: str


class ParamDesc(parseutils.TypedSerializable, abc.ABC):
    """
    The description of a parameter: a scalar, or a vector of elements.

    Parameters
    ----------
    size : int
        The number of its values.
    desc : str, optional
        What it is, for people (e.g. "the core radius").
    unit : str or UnitBase, optional
        The unit of its values (an astropy unit, e.g. 'km/s'), if it has
        one: a label of its values, which stay plain numbers in that unit
        (in configurations, expressions and evaluations), for people and
        outputs.
    min, max : float, optional
        Its valid range: the values for which the models that have it are
        defined (e.g. 0 for a dispersion), not the values that are likely.
        Every element of a vector has the same range.

    Raises
    ------
    ConfigError
        If the unit is not a unit, or min is above max.
    """

    def dump(self) -> dict[str, Any]:
        info: dict[str, Any] = dict(type=self.type())
        if not self.is_scalar():
            info.update(size=self._size)
        if self._desc is not None:
            info.update(desc=self._desc)
        if self._unit is not None:
            info.update(unit=self._unit.to_string())
        if np.isfinite(self._minimum):
            info.update(min=self._minimum)
        if np.isfinite(self._maximum):
            info.update(max=self._maximum)
        return info

    def __init__(
            self,
            size: int,
            desc: str | None,
            min: float | None,
            max: float | None,
            unit: str | ParamUnit | None = None
    ):
        minimum = -np.inf if min is None else float(min)
        maximum = +np.inf if max is None else float(max)
        if not minimum <= maximum:
            raise ConfigError(
                f"the min of a parameter ({minimum}) cannot be above its max "
                f"({maximum})")
        self._size = int(size)
        self._desc = desc
        self._unit = make_param_unit(unit)
        self._minimum = minimum
        self._maximum = maximum

    @abc.abstractmethod
    def is_scalar(self) -> bool:
        """Check whether it is a scalar (a vector has elements)."""
        pass

    def size(self) -> int:
        """Return the number of its values (1 for a scalar)."""
        return self._size

    def desc(self) -> str | None:
        """Return what it is, if given."""
        return self._desc

    def unit(self) -> ParamUnit | None:
        """Return the unit of its values, if it has one."""
        return self._unit

    def with_unit(self, unit: str | ParamUnit | None) -> Self:
        """
        Return a copy of it of another unit: for the models whose parts
        describe parameters whose unit the model knows (e.g. the
        amplitudes of the density traits of a component, in the unit of
        its light).

        Parameters
        ----------
        unit : str or UnitBase, optional
            The unit of its values (see ParamDesc); None if not known.

        Returns
        -------
        ParamDesc
            The copy (of the same column of a table, if any).

        Raises
        ------
        ConfigError
            If the unit is not a unit.
        """
        pdesc = copy.copy(self)
        pdesc._unit = make_param_unit(unit)
        return pdesc

    def minimum(self) -> float:
        """Return the lowest valid value (-inf if none)."""
        return self._minimum

    def maximum(self) -> float:
        """Return the highest valid value (+inf if none)."""
        return self._maximum

    def has_range(self) -> bool:
        """Check whether its valid range is finite on either side."""
        return bool(np.isfinite(self._minimum) or np.isfinite(self._maximum))


class ParamScalarDesc(ParamDesc):
    """
    The description of a scalar parameter.

    Parameters
    ----------
    desc : str, optional
        What it is (see ParamDesc).
    min, max : float, optional
        Its valid range (see ParamDesc).
    unit : str or UnitBase, optional
        The unit of its values (see ParamDesc).

    Raises
    ------
    ConfigError
        If the unit is not a unit, or min is above max.
    """

    @staticmethod
    def type() -> str:
        return 'scalar'

    def __init__(
            self,
            desc: str | None = None,
            min: float | None = None,
            max: float | None = None,
            unit: str | ParamUnit | None = None
    ):
        check_options(ParamScalarDesc.__init__, locals())
        super().__init__(1, desc, min, max, unit)

    def is_scalar(self) -> bool:
        return True


class ParamVectorDesc(ParamDesc):
    """
    The description of a vector parameter, whose elements are its values
    (e.g. one for each radial node of a disk).

    Parameters
    ----------
    size : int
        The number of its elements.
    desc : str, optional
        What it is (see ParamDesc).
    min, max : float, optional
        The valid range of each element (see ParamDesc).
    unit : str or UnitBase, optional
        The unit of the values of its elements (see ParamDesc).

    Raises
    ------
    ConfigError
        If the unit is not a unit, or min is above max.
    """

    @staticmethod
    def type() -> str:
        return 'vector'

    def __init__(
            self,
            size: Annotated[int, Ge(1)],
            desc: str | None = None,
            min: float | None = None,
            max: float | None = None,
            unit: str | ParamUnit | None = None
    ):
        check_options(ParamVectorDesc.__init__, locals())
        super().__init__(size, desc, min, max, unit)
        self._column: ParamColumn | None = None

    def is_scalar(self) -> bool:
        return False

    def column(self) -> ParamColumn | None:
        """Return the column of a table it is, if any."""
        return self._column

    def with_column(self, column: ParamColumn) -> Self:
        """
        Return a copy of it that is a column of a table (see ParamTable):
        for the models that declare tables (not part of configurations).
        """
        pdesc = copy.copy(self)
        pdesc._column = column
        return pdesc


pdesc_parser = parseutils.TypedParser(ParamDesc, [
    ParamScalarDesc,
    ParamVectorDesc])


def param_tables(
        pdescs: Mapping[str, ParamDesc]
) -> list[tuple[ParamTable, dict[str, str]]]:
    """
    Return the tables of parameters (see ParamTable), in the order of
    their first columns: each table, with the name of the parameter of
    each of its columns, by column name.

    Parameters
    ----------
    pdescs : Mapping
        The descriptions of the parameters, by name.
    """
    tables: dict[int, tuple[ParamTable, dict[str, str]]] = {}
    for name, pdesc in pdescs.items():
        column = pdesc.column() if isinstance(pdesc, ParamVectorDesc) \
            else None
        if column is not None:
            table = column.table
            tables.setdefault(id(table), (table, {}))[1][column.name] = name
    return list(tables.values())


def load_pdescs_dict(info: Mapping[str, Any]) -> dict[str, ParamDesc]:
    """
    Load the descriptions of parameters, by name.

    Parameters
    ----------
    info : Mapping
        The configuration of each description, by the name of its
        parameter.

    Returns
    -------
    dict
        The descriptions, by name.

    Raises
    ------
    ConfigError
        If a configuration is invalid.
    """
    return {
        name: parseutils.load_option(
            pdesc_parser.load, info, name, required=True)
        for name in info}


def dump_pdescs_dict(pdescs: Mapping[str, ParamDesc]) -> dict[str, Any]:
    """
    Dump the descriptions of parameters, by name (see load_pdescs_dict).

    Parameters
    ----------
    pdescs : Mapping
        The descriptions, by name.

    Returns
    -------
    dict
        The configuration of each description, by name.
    """
    return {name: pdesc_parser.dump(pdesc) for name, pdesc in pdescs.items()}


def pdescs_with_units(
        pdescs: Mapping[str, ParamDesc],
        units: Mapping[str, str | ParamUnit | None]
) -> dict[str, ParamDesc]:
    """
    Return descriptions of parameters with some of other units (see
    ParamDesc.with_unit): for the models whose parts describe parameters
    whose units the model knows.

    Parameters
    ----------
    pdescs : Mapping
        The descriptions, by name.
    units : Mapping
        The other units (None if not known), by the names of the
        parameters that take them; names that are not in pdescs are
        ignored.

    Returns
    -------
    dict
        The descriptions, by name.

    Raises
    ------
    ConfigError
        If a unit is not a unit.
    """
    return {
        name: pdesc.with_unit(units[name]) if name in units else pdesc
        for name, pdesc in pdescs.items()}


def make_param_unit(
        unit: str | ParamUnit | None
) -> ParamUnit | None:
    """
    Return the unit of the values of a parameter (see ParamDesc). Raise
    ConfigError if it is not a unit.
    """
    if unit is None:
        return None
    try:
        return astropy.units.Unit(unit)
    except (TypeError, ValueError) as e:
        raise ConfigError(
            f"the unit of a parameter must be an astropy unit (e.g. "
            f"'km/s'); it is {unit!r}") from e
