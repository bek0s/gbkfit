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
from typing import Annotated, Any, NamedTuple, Self, TypeAlias

import astropy.units
import numpy as np
from annotated_types import Ge

import gbkfit.units
from gbkfit.config import (
    TypedParser, TypedSerializable, check_options, load_option)


__all__ = [
    'ParamColumn',
    'ParamDesc',
    'ParamScalarDesc',
    'ParamTable',
    'ParamUnit',
    'ParamVectorDesc',
    'ValidRange',
    'default_values',
    'dump_pdescs_dict',
    'load_pdescs_dict',
    'make_param_unit',
    'param_tables',
    'pdesc_parser',
    'pdescs_with_units'
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


class ValidRange(NamedTuple):
    """
    The valid range of a parameter (see ParamDesc): its lowest and highest
    values (-inf and +inf where it has none), and whether each is out of
    it (exclusive).
    """
    min: float = -np.inf
    max: float = np.inf
    min_exclusive: bool = False
    max_exclusive: bool = False

    def bounded(self) -> bool:
        """Check whether it is finite on either side."""
        return bool(np.isfinite(self.min) or np.isfinite(self.max))

    def contains(self, value: float | np.ndarray) -> bool | np.ndarray:
        """Check whether a value is in it (or each of an array)."""
        value = np.asarray(value)
        above = value > self.min if self.min_exclusive else value >= self.min
        below = value < self.max if self.max_exclusive else value <= self.max
        inside = above & below
        return bool(inside) if inside.ndim == 0 else inside

    def __str__(self) -> str:
        # (as in mathematics: open at an exclusive or infinite end)
        low = self.min_exclusive or not np.isfinite(self.min)
        high = self.max_exclusive or not np.isfinite(self.max)
        return (
            f"{'(' if low else '['}{self.min:g}, {self.max:g}"
            f"{')' if high else ']'}")


class ParamDesc(TypedSerializable, abc.ABC):
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
        Its lowest and highest valid values. Its valid range holds the
        values for which the models that have it are defined, not those
        that are likely; every element of a vector has it.
    min_exclusive, max_exclusive : bool, optional
        Whether min and max are out of its valid range (e.g. min 0 for a
        dispersion, and min 0 exclusive for a scale length); false by
        default.
    default : float, optional
        A typical value, in its valid range, with which a model that has
        it can be evaluated before values are chosen (e.g. 200 km/s for a
        rotation speed); every element of a vector has it.
    period : float, optional
        The period of its values, if they repeat (e.g. 360 for an azimuth
        in degrees): a value and that value plus the period give the same
        models.
    title : str, optional
        Its name, for people: a few words in sentence case (e.g. "Core
        radius"), where its name is a symbol (e.g. rc).

    Raises
    ------
    ValueError
        If the unit is not a unit, a bound is exclusive but not given, the
        valid range is empty, the default is not finite or outside it, or
        the period is not positive.
    """

    def dump(self) -> dict[str, Any]:
        info: dict[str, Any] = dict(type=self.type())
        if not self.is_scalar():
            info.update(size=self._size)
        if self._title is not None:
            info.update(title=self._title)
        if self._desc is not None:
            info.update(desc=self._desc)
        if self._unit is not None:
            info.update(unit=gbkfit.units.format_unit(self._unit))
        valid = self._valid
        if np.isfinite(valid.min):
            info.update(min=valid.min)
            if valid.min_exclusive:
                info.update(min_exclusive=True)
        if np.isfinite(valid.max):
            info.update(max=valid.max)
            if valid.max_exclusive:
                info.update(max_exclusive=True)
        if self._default is not None:
            info.update(default=self._default)
        if self._period is not None:
            info.update(period=self._period)
        return info

    def __init__(
            self,
            size: int,
            desc: str | None = None,
            min: float | None = None,
            max: float | None = None,
            min_exclusive: bool = False,
            max_exclusive: bool = False,
            unit: str | ParamUnit | None = None,
            default: float | None = None,
            period: float | None = None,
            title: str | None = None
    ):
        for name, bound, exclusive in (
                ('min', min, min_exclusive), ('max', max, max_exclusive)):
            if exclusive and bound is None:
                raise ValueError(
                    f"a parameter with {name}_exclusive needs a {name}")
        valid = ValidRange(
            -np.inf if min is None else float(min),
            np.inf if max is None else float(max),
            bool(min_exclusive), bool(max_exclusive))
        if not (valid.min < valid.max or valid.min == valid.max and not (
                valid.min_exclusive or valid.max_exclusive)):
            raise ValueError(
                f"the valid range of a parameter, {valid}, is empty")
        if default is not None and not (
                np.isfinite(default) and valid.contains(default)):
            raise ValueError(
                f"the default of a parameter ({default}) must be finite and "
                f"in its valid range {valid}")
        if period is not None and not (np.isfinite(period) and period > 0):
            raise ValueError(
                f"the period of a parameter ({period}) must be positive")
        self._size = int(size)
        self._title = title
        self._desc = desc
        self._unit = make_param_unit(unit)
        self._valid = valid
        self._default = None if default is None else float(default)
        self._period = None if period is None else float(period)

    @abc.abstractmethod
    def is_scalar(self) -> bool:
        """Check whether it is a scalar (a vector has elements)."""
        pass

    def size(self) -> int:
        """Return the number of its values (1 for a scalar)."""
        return self._size

    def title(self) -> str | None:
        """Return its name for people, if given."""
        return self._title

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
        ValueError
            If the unit is not a unit.
        """
        pdesc = copy.copy(self)
        pdesc._unit = make_param_unit(unit)
        return pdesc

    def valid_range(self) -> ValidRange:
        """Return its valid range (see ParamDesc)."""
        return self._valid

    def default(self) -> float | None:
        """Return its typical value, if given (see ParamDesc)."""
        return self._default

    def period(self) -> float | None:
        """Return the period of its values, if they repeat."""
        return self._period


class ParamScalarDesc(ParamDesc):
    """
    The description of a scalar parameter.

    Parameters
    ----------
    desc : str, optional
        What it is (see ParamDesc).
    min, max : float, optional
        Its lowest and highest valid values (see ParamDesc).
    min_exclusive, max_exclusive : bool, optional
        Whether min and max are out of its valid range (see ParamDesc).
    unit : str or UnitBase, optional
        The unit of its values (see ParamDesc).
    default : float, optional
        Its typical value (see ParamDesc).
    period : float, optional
        The period of its values, if they repeat (see ParamDesc).
    title : str, optional
        Its name for people (see ParamDesc).

    Raises
    ------
    ValueError
        See ParamDesc.
    """

    @staticmethod
    def type() -> str:
        return 'scalar'

    @staticmethod
    def type_title() -> str:
        return 'Scalar'

    def __init__(
            self,
            desc: str | None = None,
            min: float | None = None,
            max: float | None = None,
            min_exclusive: bool = False,
            max_exclusive: bool = False,
            unit: str | ParamUnit | None = None,
            default: float | None = None,
            period: float | None = None,
            title: str | None = None
    ):
        check_options(ParamScalarDesc.__init__, locals())
        super().__init__(
            1, desc, min, max, min_exclusive, max_exclusive, unit, default,
            period, title)

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
        The lowest and highest valid values of each element (see
        ParamDesc).
    min_exclusive, max_exclusive : bool, optional
        Whether min and max are out of the valid range (see ParamDesc).
    unit : str or UnitBase, optional
        The unit of the values of its elements (see ParamDesc).
    default : float, optional
        The typical value of each element (see ParamDesc).
    period : float, optional
        The period of the values of its elements, if they repeat (see
        ParamDesc).
    title : str, optional
        Its name for people (see ParamDesc).

    Raises
    ------
    ValueError
        See ParamDesc.
    """

    @staticmethod
    def type() -> str:
        return 'vector'

    @staticmethod
    def type_title() -> str:
        return 'Vector'

    def __init__(
            self,
            size: Annotated[int, Ge(1)],
            desc: str | None = None,
            min: float | None = None,
            max: float | None = None,
            min_exclusive: bool = False,
            max_exclusive: bool = False,
            unit: str | ParamUnit | None = None,
            default: float | None = None,
            period: float | None = None,
            title: str | None = None
    ):
        check_options(ParamVectorDesc.__init__, locals())
        super().__init__(
            size, desc, min, max, min_exclusive, max_exclusive, unit,
            default, period, title)
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


pdesc_parser = TypedParser(ParamDesc, [
    ParamScalarDesc,
    ParamVectorDesc])


def default_values(
        pdescs: Mapping[str, ParamDesc]
) -> dict[str, float | np.ndarray]:
    """
    Return the typical values of parameters (see ParamDesc), by name: of
    those that have one; an array of it for a vector. With them, models
    can be evaluated before their values are chosen (e.g. the parameters
    of a configuration, see gbkfit.setup.describe and
    ObservationGroup.pdescs).

    Parameters
    ----------
    pdescs : Mapping
        The descriptions of the parameters, by name.

    Returns
    -------
    dict
        The values, by name.
    """
    return {
        name: pdesc.default() if pdesc.is_scalar()
        else np.full(pdesc.size(), pdesc.default())
        for name, pdesc in pdescs.items() if pdesc.default() is not None}


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
    ValueError, TypeError
        If an object it makes refuses its options.
    """
    return {
        name: load_option(
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
    ValueError
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
        if isinstance(unit, str):
            return gbkfit.units.parse_unit(unit)
        return astropy.units.Unit(unit)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"the unit of a parameter must be a unit (e.g. 'km/s'); it is "
            f"{unit!r}") from e
