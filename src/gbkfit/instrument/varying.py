"""
Options of PSFs and LSFs that vary along the spectral axis.

An option that varies is a table of its values at points of the spectral
axis: wavelengths, frequencies or velocities (SpectralTable). Between the
points the values are interpolated linearly, and beyond them they are
those of the nearest point. A PSF or an LSF whose options vary is made by
the varying of its class, e.g.

    alpha = SpectralTable([4800, 7000, 9300] * u.AA, [0.75, 0.65, 0.58])
    psf = PSFMoffat.varying(alpha=alpha, beta=2.5)
    psf.at(6500 * u.AA)  # the PSFMoffat at 6500 Angstrom

In a configuration, a table is given inline (see SpectralTable), e.g.

    alpha:
      points: {value: [4800, 7000, 9300], unit: Angstrom}
      values: [0.75, 0.65, 0.58]

or as two columns of a table file (FITS, ECSV or CSV), those of the
points and of the values (see SpectralTable.from_file), e.g.

    alpha: {file: psf.ecsv, points: wavelength, column: alpha}

Values without a unit are in the unit of the option (e.g. arcsec for
the widths of PSFs); the widths of LSFs, Quantities, have a unit of their
own (e.g. Angstrom), which tables give. Velocities are in the convention
of the spectral axis (see VelocityConvention): optical or log for a rest
wavelength, radio for a rest frequency.
"""

import typing
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import astropy.constants
import astropy.units as u
import numpy as np

import gbkfit.units
from gbkfit.config import (
    DataFile, Serializable, check_options, config_constructor, config_path,
    dump_quantity, option_types, parse_options_for_callable, split_data_file,
    union_members, unit_of, varying_options)
from gbkfit.grid import VelocityConvention
from gbkfit.io import read_column, read_table


__all__ = [
    'SpectralTable'
]


# The velocity unit of the model
_KMS = u.km / u.s
_C = astropy.constants.c.to_value(_KMS)

# The things given at points of the spectral axis (see per_point)
_Thing = TypeVar('_Thing')


class _NoConvention(ValueError):
    """
    The error of a conversion between velocities and wavelengths or
    frequencies without the convention of the velocities of the spectral
    axis (see _equivalency).
    """


def _convert(
        points: u.Quantity,
        unit: u.UnitBase,
        convention: VelocityConvention | None
) -> np.ndarray:
    """
    Points of the spectral axis in the unit of another kind of point (e.g.
    velocities as wavelengths): only between velocities and the others
    is the convention needed.
    """
    if points.unit.is_equivalent(unit, u.spectral()):
        return points.to_value(unit, u.spectral())
    return points.to_value(unit, _equivalency(convention))


def _equivalency(
        convention: VelocityConvention | None
) -> u.Equivalency:
    """
    Return the equivalency of the velocities of the spectral axis of a
    convention and its wavelengths or frequencies.

    Raises
    ------
    ValueError
        If there is no convention, or it has no rest.
    """
    if convention is None:
        raise _NoConvention(
            "velocities and wavelengths or frequencies are converted with "
            "the convention of the velocities of the spectral axis, which "
            "is not given")
    return convention.equivalency()


def as_points(points: u.Quantity) -> u.Quantity:
    """
    Return points of the spectral axis as a Quantity.

    Parameters
    ----------
    points : Quantity
        Wavelengths, frequencies or velocities: one, or a sequence.

    Returns
    -------
    Quantity
        The points.

    Raises
    ------
    TypeError
        If they are not wavelengths, frequencies or velocities.
    """
    points = u.Quantity(points, dtype=float)
    if not any(points.unit.is_equivalent(unit, u.spectral())
               for unit in (u.m, _KMS)):
        raise TypeError(
            f"points of the spectral axis must be wavelengths, frequencies "
            f"or velocities; their unit is '{points.unit}'")
    return points


def per_point(
        points: u.Quantity,
        make: Callable[[u.Quantity], list[_Thing]]
) -> _Thing | list[_Thing]:
    """
    Return the things that make gives for points of the spectral axis:
    a list, or one thing for one point.

    Parameters
    ----------
    points : Quantity
        One point, or a sequence (see as_points).
    make : Callable
        Gives a list of things for a 1D Quantity of points.

    Returns
    -------
    object or list
        The list, or the only thing for one point.
    """
    points = as_points(points)
    things = make(np.atleast_1d(points))
    return things[0] if points.ndim == 0 else things


def to_velocities(
        points: u.Quantity,
        convention: VelocityConvention | None = None
) -> np.ndarray:
    """
    Return points of the spectral axis as velocities.

    Parameters
    ----------
    points : Quantity
        Wavelengths, frequencies or velocities.
    convention : VelocityConvention, optional
        The velocity convention of the spectral axis; needed for
        wavelengths and frequencies.

    Returns
    -------
    ndarray
        The velocities (km/s), in the convention of the spectral axis.

    Raises
    ------
    ValueError
        If the convention is needed but not given, or has no rest.
    """
    return _convert(as_points(points), _KMS, convention)


def to_frequencies(
        points: u.Quantity,
        convention: VelocityConvention | None = None
) -> np.ndarray:
    """
    Return points of the spectral axis as frequencies.

    Parameters
    ----------
    points : Quantity
        Wavelengths, frequencies or velocities.
    convention : VelocityConvention, optional
        The velocity convention of the spectral axis; needed for
        velocities.

    Returns
    -------
    ndarray
        The frequencies (Hz).

    Raises
    ------
    ValueError
        If the convention is needed but not given, or has no rest.
    """
    return _convert(as_points(points), u.Hz, convention)


def blend(
        points: u.Quantity,
        at: u.Quantity,
        convention: VelocityConvention | None = None
) -> list[tuple[int, int, float]]:
    """
    Return how to blend things given at points of the spectral axis (e.g.
    images) at other points.

    The things are blended linearly along the axis of their points.

    Parameters
    ----------
    points : Quantity
        The points of the things: wavelengths, frequencies or velocities.
    at : Quantity
        The points to blend them at, of any kind (1D).
    convention : VelocityConvention, optional
        The velocity convention of the spectral axis; needed
        between velocities and the other kinds of points.

    Returns
    -------
    list of tuple
        For each point, (i, j, t): it takes 1 - t of the thing of the
        point i and t of that of the point j, the two nearest points;
        beyond the points, all of the nearest: of i (t = 0) below the
        first, and of j (t = 1) above the last.
    """
    order = np.argsort(points.value)
    positions = np.interp(
        _convert(at, points.unit, convention), points.value[order],
        np.arange(len(order)))
    result = []
    for position in positions:
        k = min(int(position), len(order) - 2) if len(order) > 1 else 0
        t = float(position - k)
        result.append((
            int(order[k]), int(order[k + 1] if t > 0 else order[k]), t))
    return result


class SpectralTable(Serializable):
    """
    The values of an option at points of the spectral axis.

    Between the points the values are interpolated linearly along the axis
    of the points (e.g. in wavelength), and beyond them they are those of
    the nearest point.

    Parameters
    ----------
    points : Quantity
        The points: wavelengths, frequencies or velocities, each once.
    values : array_like
        The value at each point.
    unit : str or Unit, optional
        The unit of the values; by default, the units of the option.

    Raises
    ------
    TypeError
        If the points are not wavelengths, frequencies or velocities.
    ValueError
        If the points are repeated, or there is not a value for each.
    """

    config_constructors = ('__init__', 'from_file')

    def __init__(
            self,
            points: u.Quantity,
            values: Sequence[float] | np.ndarray,
            unit: str | u.UnitBase | None = None
    ):
        check_options(SpectralTable.__init__, locals())
        points = as_points(points)
        values = np.asarray(values, dtype=float)
        if (points.ndim != 1 or points.size == 0
                or values.shape != points.shape):
            raise ValueError(
                f"a table needs a value at each of its points; it has "
                f"{points.size} points and {values.size} values")
        if not (np.all(np.isfinite(points)) and np.all(np.isfinite(values))):
            raise ValueError(
                "the points and values of a table must be finite")
        if np.unique(points).size != points.size:
            raise ValueError("the points of a table must be distinct")
        self._points = points
        self._values = values
        if isinstance(unit, str):
            unit = gbkfit.units.parse_unit(unit)
        self._unit = None if unit is None else u.Unit(unit)

    @classmethod
    def from_file(
            cls,
            file: DataFile,
            points: str,
            column: str,
            units: Mapping[str, str] | None = None
    ) -> 'SpectralTable':
        """
        Read a table from two columns of a table file (FITS, ECSV or
        CSV): those of its points and of its values.

        The unit of the points says whether they are wavelengths,
        frequencies or velocities.

        Parameters
        ----------
        file : DataFile
            The file, and of a FITS file the HDU of the table (see
            read_table).
        points : str
            The column of the points.
        column : str
            The column of the values.
        units : Mapping, optional
            The units of the columns without units (e.g. those of a CSV
            file), by column.

        Returns
        -------
        SpectralTable
            The table.

        Raises
        ------
        ValueError
            If the file has no such columns, or the points have no unit.
        """
        check_options(SpectralTable.from_file, locals())
        filename = split_data_file(file)[0]
        table = read_table(file)
        units = units or {}
        for name in (points, column):
            if name not in table.colnames:
                raise ValueError(
                    f"{filename}: there is no column {name!r}; the columns "
                    f"are {table.colnames}")
        # Masked values are NaN, which the table refuses
        point_values = read_column(table, points, units.get(points))
        if point_values.unit == u.dimensionless_unscaled:
            raise ValueError(
                f"{filename}: the column of the points, {points!r}, has no "
                f"unit; give it in units")
        values = read_column(table, column, units.get(column))
        unit = None if values.unit == u.dimensionless_unscaled \
            else values.unit
        return cls(point_values, values.value, unit)

    def dump(self) -> dict[str, Any]:
        """
        Dump the table to its configuration (with its points and values).

        Returns
        -------
        dict
            The configuration.
        """
        unit = {} if self._unit is None \
            else dict(unit=gbkfit.units.format_unit(self._unit))
        return dict(
            points=dump_quantity(self._points),
            values=self._values.tolist()) | unit

    def points(self) -> u.Quantity:
        """
        Return the points.

        Returns
        -------
        Quantity
            The points.
        """
        return self._points.copy()

    def values(self) -> np.ndarray:
        """
        Return the values, in their unit.

        Returns
        -------
        ndarray
            The values.
        """
        return self._values.copy()

    def unit(self) -> u.Unit | None:
        """
        Return the unit of the values.

        Returns
        -------
        Unit or None
            The unit, or None for the units of the option.
        """
        return self._unit

    def at(
            self,
            points: u.Quantity,
            convention: VelocityConvention | None = None
    ) -> np.ndarray | u.Quantity:
        """
        Return the values at points of the spectral axis.

        Parameters
        ----------
        points : Quantity
            The points, of any kind (see as_points).
        convention : VelocityConvention, optional
            The velocity convention of the spectral axis; needed
            between velocities and the other kinds of points.

        Returns
        -------
        ndarray or Quantity
            The values at the points: in their unit, if they have one.

        Raises
        ------
        ValueError
            If the convention is needed but not given, or has no rest.
        """
        x = _convert(as_points(points), self._points.unit, convention)
        order = np.argsort(self._points.value)
        values = np.interp(x, self._points.value[order], self._values[order])
        return values if self._unit is None else values * self._unit

    def velocity_range(
            self, convention: VelocityConvention | None = None
    ) -> tuple[float, float]:
        """
        Return the range of the points as velocities of a spectral axis.

        Parameters
        ----------
        convention : VelocityConvention, optional
            The velocity convention of the spectral axis; needed
            for points in wavelength or frequency.

        Returns
        -------
        tuple of float
            The lowest and the highest velocity (km/s).
        """
        velocities = to_velocities(self._points, convention)
        return float(velocities.min()), float(velocities.max())

def load_tables(cls: type, info: dict[str, Any]) -> dict[str, Any]:
    """
    Return the options of a configuration of an object of a class (e.g. a
    PSF) with those that can vary (see _VaryingOptions) and are given as
    tables loaded (see SpectralTable).

    The options that can vary are those of the constructor of the class
    that the configuration is of (see config_constructor). A table is a
    dict of points (inline, or from a file); a Quantity, of a value and a
    unit, is not one.

    Parameters
    ----------
    cls : type
        The class.
    info : dict
        The options.

    Returns
    -------
    dict
        The options.

    Raises
    ------
    ValueError
        If a table is invalid.
    """
    options = dict(info)
    for name in varying_options(_constructor(cls, info)[0]):
        if isinstance(info.get(name), Mapping) and 'points' in info[name]:
            with config_path(name):
                options[name] = SpectralTable.load(info[name])
    return options


def _constructor(
        cls: type, options: Mapping[str, Any]
) -> tuple[Callable, Callable]:
    """
    Return the constructor of a class that options are of (see
    config_constructor), as a function to read its parameters from, and
    as a callable that makes the object.
    """
    name = config_constructor(cls, options)
    if name == '__init__':
        return cls.__init__, cls
    return getattr(cls, name), getattr(cls, name)


class _VaryingOptions:
    """
    The options of an object of a class (e.g. a PSF), some of which are
    tables of values along the spectral axis (see PSF.varying).

    The options are those of a constructor of the class (see
    config_constructor; e.g. the fwhm of LSFGauss.from_fwhm). Those that
    can vary are those whose annotations are marked so (see Varying).
    Values of an option that is a Quantity (e.g. the width of an LSF) are
    in the unit of their table, which must have one; values of others are
    in the unit of the annotation (see Unit; none for none), and are
    converted to it from that of their table.

    Parameters
    ----------
    cls : type
        The class.
    options : dict
        The options; those that vary are tables (SpectralTable).

    Raises
    ------
    ValueError
        If an option that cannot vary is a table, or a table of a
        Quantity has no unit, or the options are invalid, or so are the
        values of the tables at their points (see _check).
    """

    def __init__(self, cls: type, options: dict[str, Any]):
        signature, self._make = _constructor(cls, options)
        tables = {
            name: value for name, value in options.items()
            if isinstance(value, SpectralTable)}
        types = option_types(signature)
        for name, table in tables.items():
            if name not in varying_options(signature):
                raise ValueError(
                    f"option '{name}' of {cls.__qualname__} cannot vary "
                    f"along the spectral axis")
            if table.unit() is None and _is_quantity(types[name]):
                raise ValueError(
                    f"option '{name}' of {cls.__qualname__} has a unit: give "
                    f"that of the values of its table (unit)")
        self._cls = cls
        self._units = {
            name: None if _is_quantity(types[name])
            else u.Unit(unit_of(types[name]) or '')
            for name in tables}
        self._constants = parse_options_for_callable(
            {name: value for name, value in options.items()
             if name not in tables},
            signature, ignore_params=set(tables))
        self._tables = tables
        self._check()

    def _check(self) -> None:
        """
        Build the objects at the points of all the tables, from the values
        of all of them there: values valid each on its own may not be
        together (e.g. a bmin above bmaj).

        Velocities are put among wavelengths or frequencies by the
        convention of the velocities of the spectral axis, which tables
        do not know: tables that need it are checked where the objects
        are built, at the channels of the data (see at).
        """
        tables = list(self._tables.values())
        unit = tables[0].points().unit
        try:
            points = np.unique(np.concatenate([
                _convert(table.points(), unit, None) for table in tables]))
        except _NoConvention:
            return
        self.at(points * unit)

    def cls(self) -> type:
        """Return the class."""
        return self._cls

    def dump(self) -> dict[str, Any]:
        """Return the configuration of the options, with the tables."""
        constants = {
            name: gbkfit.units.format_quantity(value)
            if isinstance(value, u.Quantity) else value
            for name, value in self._constants.items()}
        return dict(type=self._cls.type()) | constants | {
            name: table.dump() for name, table in self._tables.items()}

    def at(
            self,
            points: u.Quantity,
            convention: VelocityConvention | None = None
    ) -> list[Any]:
        """
        Return an object of the class at each of the points (1D).

        Raise ValueError if the convention is needed but not given, or
        the options are invalid at a point (e.g. a bmin above bmaj).
        """
        values = {
            name: self._values_at(name, points, convention)
            for name in self._tables}
        objects = []
        for i, point in enumerate(points):
            options = {name: value[i] for name, value in values.items()}
            try:
                objects.append(self._make(**self._constants | options))
            except ValueError as e:
                raise ValueError(
                    f"at {gbkfit.units.format_quantity(point)}: {e}") from None
        return objects

    def _values_at(
            self,
            name: str,
            points: u.Quantity,
            convention: VelocityConvention | None
    ) -> list[float] | u.Quantity:
        """
        The values of the table of an option at points (1D): Quantities in
        the unit of the table, or floats in the unit of the option.
        """
        values = self._tables[name].at(points, convention)
        unit = self._units[name]
        if unit is None:
            return values
        if isinstance(values, u.Quantity):
            try:
                values = values.to_value(unit)
            except u.UnitConversionError:
                raise ValueError(
                    f"the values of the table of option '{name}' are in "
                    f"'{gbkfit.units.format_unit(values.unit)}', which "
                    f"cannot be converted to "
                    f"'{gbkfit.units.format_unit(unit)}'") from None
        return [float(x) for x in values]

    def velocity_range(
            self, convention: VelocityConvention | None = None
    ) -> tuple[float, float]:
        """Return the range of velocities that all the tables cover."""
        ranges = [
            table.velocity_range(convention)
            for table in self._tables.values()]
        return (max(low for low, _ in ranges), min(high for _, high in ranges))


def _is_quantity(type_: Any) -> bool:
    """Return whether an option of an annotation is a Quantity."""
    return any(
        x is u.Quantity or typing.get_args(x)[:1] == (u.Quantity,)
        for x in union_members(type_))

