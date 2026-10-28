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

or as a column of a table file (FITS, ECSV or CSV), which has a column of
the points named wavelength, frequency or velocity (see
SpectralTable.from_file), e.g.

    alpha: {file: psf.ecsv, column: alpha}

Values without units are in the units of the option. Velocities are in
the convention of the spectral axis (see gridutils.VelocityConvention):
optical or log for a rest wavelength, radio for a rest frequency.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import astropy.table
import astropy.units as u
import numpy as np

from gbkfit.utils import gridutils, parseutils, typeutils
from gbkfit.utils.parseutils import ConfigError, check_options
from gbkfit.utils.typeutils import FilePath


__all__ = [
    'SpectralTable'
]


# The kinds of points of the spectral axis, and the velocity unit of the
# model
_AXES = ('wavelength', 'frequency', 'velocity')
_KMS = u.km / u.s

# The things given at points of the spectral axis (see per_point)
_Thing = TypeVar('_Thing')


def _convert(
        points: u.Quantity,
        unit: u.UnitBase,
        convention: gridutils.VelocityConvention | None
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
        convention: gridutils.VelocityConvention | None
) -> u.Equivalency:
    """
    Return the equivalency of the velocities of the spectral axis of a
    convention and its wavelengths or frequencies.

    Raises
    ------
    ConfigError
        If there is no convention, or it has no rest.
    """
    if convention is None:
        raise ConfigError(
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
        convention: gridutils.VelocityConvention | None = None
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
    ConfigError
        If the convention is needed but not given, or has no rest.
    """
    return _convert(as_points(points), _KMS, convention)


def to_frequencies(
        points: u.Quantity,
        convention: gridutils.VelocityConvention | None = None
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
    ConfigError
        If the convention is needed but not given, or has no rest.
    """
    return _convert(as_points(points), u.Hz, convention)


def blend(
        points: u.Quantity,
        at: u.Quantity,
        convention: gridutils.VelocityConvention | None = None
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
        beyond the points, all of the nearest (t = 0).
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


class SpectralTable(parseutils.Serializable):
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
    ConfigError
        If the points are not wavelengths, frequencies or velocities, are
        repeated, or there is not a value for each.
    """

    config_constructors = ('__init__', 'from_file')

    def __init__(
            self,
            points: u.Quantity,
            values: Sequence[float] | np.ndarray,
            unit: str | u.UnitBase | None = None
    ):
        check_options(SpectralTable.__init__, locals())
        try:
            points = as_points(points)
        except TypeError as e:
            raise ConfigError(str(e)) from e
        values = np.asarray(values, dtype=float)
        if (points.ndim != 1 or points.size == 0
                or values.shape != points.shape):
            raise ConfigError(
                f"a table needs a value at each of its points; it has "
                f"{points.size} points and {values.size} values")
        if not (np.all(np.isfinite(points)) and np.all(np.isfinite(values))):
            raise ConfigError(
                "the points and values of a table must be finite")
        if np.unique(points).size != points.size:
            raise ConfigError("the points of a table must be distinct")
        self._points = points
        self._values = values
        self._unit = None if unit is None else u.Unit(unit)

    @classmethod
    def from_file(
            cls,
            file: FilePath,
            column: str,
            hdu: int | str | None = None,
            units: Mapping[str, str] | None = None
    ) -> 'SpectralTable':
        """
        Read a table from a column of a table file (FITS, ECSV or CSV).

        The file has a column of the points, named wavelength, frequency
        or velocity.

        Parameters
        ----------
        file : str
            The file.
        column : str
            The column of the values.
        hdu : int or str, optional
            The HDU of a FITS file.
        units : Mapping, optional
            The units of the columns without units (e.g. those of a CSV
            file), by column.

        Returns
        -------
        SpectralTable
            The table.

        Raises
        ------
        ConfigError
            If the file has no column of points or values, or its points
            have no unit.
        """
        check_options(SpectralTable.from_file, locals())
        options = {} if hdu is None else dict(hdu=hdu)
        table = astropy.table.Table.read(file, **options)
        units = units or {}
        axes = [axis for axis in _AXES if axis in table.colnames]
        if len(axes) != 1:
            raise ConfigError(
                f"{file}: a table needs a column of its points, named "
                f"one of {list(_AXES)}; its columns are {table.colnames}")
        if column not in table.colnames:
            raise ConfigError(
                f"{file}: there is no column '{column}'; the columns "
                f"are {table.colnames}")

        def unit(name):
            return units.get(name, table[name].unit)
        axis = axes[0]
        points_unit = unit(axis) or (_KMS if axis == 'velocity' else None)
        if points_unit is None:
            raise ConfigError(
                f"{file}: the column '{axis}' has no unit; give it in "
                f"units")
        return cls(
            np.asarray(table[axis], dtype=float) * u.Unit(points_unit),
            np.asarray(table[column], dtype=float), unit(column))

    def dump(self) -> dict[str, Any]:
        """
        Dump the table to its configuration (with its points and values).

        Returns
        -------
        dict
            The configuration.
        """
        points = dict(
            value=self._points.value.tolist(), unit=str(self._points.unit))
        unit = {} if self._unit is None else dict(unit=str(self._unit))
        return dict(points=points, values=self._values.tolist()) | unit

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
            self, points: u.Quantity, unit: str | u.UnitBase,
            convention: gridutils.VelocityConvention | None = None
    ) -> np.ndarray:
        """
        Return the values at points of the spectral axis.

        Parameters
        ----------
        points : Quantity
            The points, of any kind (see as_points).
        unit : str or Unit
            The unit of the option. Values with other units are converted
            to it; for a velocity, a width in wavelength or frequency is
            converted at each point of the table.
        convention : VelocityConvention, optional
            The velocity convention of the spectral axis; needed
            between velocities and the other kinds of points.

        Returns
        -------
        ndarray
            The values at the points, in the unit of the option.

        Raises
        ------
        ConfigError
            If the convention is needed but not given or has no convention,
            or the values cannot be
            converted to the unit of the option.
        """
        x = _convert(as_points(points), self._points.unit, convention)
        values = self._values_in(u.Unit(unit), convention)
        order = np.argsort(self._points.value)
        return np.interp(x, self._points.value[order], values[order])

    def velocity_range(
            self, convention: gridutils.VelocityConvention | None = None
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

    def _values_in(
            self,
            unit: u.UnitBase,
            convention: gridutils.VelocityConvention | None
    ) -> np.ndarray:
        """The values in a unit."""
        if self._unit is None:
            return self._values
        values = self._values * self._unit
        if values.unit.is_equivalent(unit):
            return values.to_value(unit)
        if unit.is_equivalent(_KMS) and values.unit.is_equivalent(
                u.m, u.spectral()):
            # A width around each point, in wavelength or frequency
            doppler = _equivalency(convention)
            centres = self._points.to(values.unit, u.spectral() + doppler)
            low = (centres - values / 2).to_value(_KMS, doppler)
            high = (centres + values / 2).to_value(_KMS, doppler)
            return np.abs(high - low)
        raise ConfigError(
            f"the values of a table in '{self._unit}' cannot be converted "
            f"to '{unit}'")


def load_tables(cls: type, info: dict[str, Any]) -> dict[str, Any]:
    """
    Return the options of a configuration of an object of a class (e.g. a
    PSF) with those that can vary (see _VaryingOptions) and are given as
    tables (dicts) loaded (see SpectralTable).

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
    ConfigError
        If a table is invalid.
    """
    options = dict(info)
    for name in getattr(cls, 'VARYING_OPTIONS', ()):
        if isinstance(info.get(name), Mapping):
            with parseutils.config_path(name):
                options[name] = SpectralTable.load(info[name])
    return options


class _VaryingOptions:
    """
    The options of an object of a class (e.g. a PSF), some of which are
    tables of values along the spectral axis (see PSF.varying).

    The class lists the names of its options that can vary in
    VARYING_OPTIONS (e.g. ('sigma', 'ratio')); their units are those of
    the annotations of its __init__ (see typeutils.Unit; none for none).

    Parameters
    ----------
    cls : type
        The class.
    options : dict
        The options; those that vary are tables (SpectralTable).

    Raises
    ------
    ConfigError
        If an option that cannot vary is a table, or the options, or the
        values of the tables, are invalid.
    """

    def __init__(self, cls: type, options: dict[str, Any]):
        tables = {
            name: value for name, value in options.items()
            if isinstance(value, SpectralTable)}
        for name in tables:
            if name not in getattr(cls, 'VARYING_OPTIONS', ()):
                raise ConfigError(
                    f"option '{name}' of {cls.__qualname__} cannot vary "
                    f"along the spectral axis")
        # Check the options with the first value of each table, and the
        # values of each table with the first values of the others
        first = {name: table.values()[0] for name, table in tables.items()}
        checked = parseutils.parse_options_for_callable(
            options | first, cls.__init__)
        for name, table in tables.items():
            with parseutils.config_path(name):
                for value in table.values():
                    cls(**checked | {name: value})
        types = typeutils.option_types(cls.__init__)
        self._cls = cls
        self._units = {
            name: typeutils.unit_of(types[name]) or '' for name in tables}
        self._constants = {
            name: value for name, value in checked.items()
            if name not in tables}
        self._tables = tables

    def cls(self) -> type:
        """Return the class."""
        return self._cls

    def dump(self) -> dict[str, Any]:
        """Return the configuration of the options, with the tables."""
        return dict(type=self._cls.type()) | self._constants | {
            name: table.dump() for name, table in self._tables.items()}

    def at(
            self,
            points: u.Quantity,
            convention: gridutils.VelocityConvention | None = None
    ) -> list[Any]:
        """Return an object of the class at each of the points (1D)."""
        values = {
            name: table.at(points, self._units[name], convention)
            for name, table in self._tables.items()}
        return [
            self._cls(**self._constants | {
                name: float(value[i]) for name, value in values.items()})
            for i in range(len(points))]

    def velocity_range(
            self, convention: gridutils.VelocityConvention | None = None
    ) -> tuple[float, float]:
        """Return the range of velocities that all the tables cover."""
        ranges = [
            table.velocity_range(convention)
            for table in self._tables.values()]
        return (max(low for low, _ in ranges), min(high for _, high in ranges))
