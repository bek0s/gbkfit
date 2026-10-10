"""
Options of PSFs and LSFs that vary along the spectral axis.

An option that varies is a table of its values at points of the spectral
axis: wavelengths, frequencies or velocities. Between the points the
values are interpolated linearly, and beyond them they are those of the
nearest point. The table is given in the configuration, e.g.

    alpha:
      wavelength: {values: [4800, 7000, 9300], unit: Angstrom}
      values: [0.75, 0.65, 0.58]

or as a column of a table file (FITS, ECSV or CSV), which has a column of
the points named wavelength, frequency or velocity, e.g.

    alpha: {table: psf.ecsv, column: alpha}

Values without units are in the units of the option. Velocities are in
the convention of the spectral axis: optical for a rest wavelength, and
radio for a rest frequency (see gridutils.Coords).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import astropy.table
import astropy.units as u
import numpy as np

from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'SpectralTable',
    'Varying'
]


# The kinds of points of a table, and the velocity unit of the model
_AXES = ('wavelength', 'frequency', 'velocity')
_KMS = u.km / u.s


def _doppler(rest):
    """The velocity convention of a spectral axis of the given rest."""
    if rest is None:
        raise ConfigError(
            "a table in wavelength or frequency needs the rest of the "
            "spectral axis of the data (rest)")
    rest = u.Quantity(rest)
    return (u.doppler_optical(rest) if rest.unit.is_equivalent(u.m)
            else u.doppler_radio(rest))


def _quantity(x, unit):
    """
    A Quantity from an option: a list of values, in the given unit (None
    for none), or a dict with the values and their unit.
    """
    if isinstance(x, Mapping):
        x = parseutils.parse_options(x, required={'values', 'unit'})
        return np.asarray(x['values'], dtype=float), u.Unit(x['unit'])
    return np.asarray(x, dtype=float), unit


class SpectralTable:
    """
    The values of an option at points of the spectral axis.

    Parameters
    ----------
    points : Quantity
        The points: wavelengths, frequencies or velocities, each once.
    values : array_like
        The value at each point.
    unit : str or Unit, optional
        The unit of the values; by default, the units of the option.
    """

    def __init__(
            self,
            points: u.Quantity,
            values: Sequence[float] | np.ndarray,
            unit: Any = None
    ):
        points = u.Quantity(points, dtype=float)
        values = np.asarray(values, dtype=float)
        if (points.ndim != 1 or points.size == 0
                or values.shape != points.shape):
            raise ConfigError(
                f"a table needs a value at each of its points; it has "
                f"{points.size} points and {values.size} values")
        if not any(points.unit.is_equivalent(unit_, u.spectral())
                   for unit_ in (u.m, _KMS)):
            raise ConfigError(
                f"the points of a table must be wavelengths, frequencies or "
                f"velocities; their unit is '{points.unit}'")
        if not (np.all(np.isfinite(points)) and np.all(np.isfinite(values))):
            raise ConfigError(
                "the points and values of a table must be finite")
        if np.unique(points).size != points.size:
            raise ConfigError("the points of a table must be distinct")
        self._points = points
        self._values = values
        self._unit = None if unit is None else u.Unit(unit)

    @classmethod
    def load(cls, info: Mapping[str, Any]) -> 'SpectralTable':
        """
        Load a table from its configuration (see the module).

        Parameters
        ----------
        info : Mapping
            The table: its points (wavelength, frequency or velocity) and
            values, or a file (table), its column, and optionally the HDU
            of a FITS file and the units of columns without units (units,
            a dict of units by column).

        Returns
        -------
        SpectralTable
            The table.

        Raises
        ------
        ConfigError
            If the configuration or the table is invalid.
        """
        if 'table' in info:
            return cls._load_file(info)
        info = parseutils.parse_options(
            info, required={'values'}, optional=set(_AXES))
        axes = [axis for axis in _AXES if axis in info]
        if len(axes) != 1:
            raise ConfigError(
                f"a table needs its points as one of {list(_AXES)}")
        axis = axes[0]
        points, points_unit = _quantity(
            info[axis], _KMS if axis == 'velocity' else None)
        if points_unit is None:
            raise ConfigError(f"the {axis} of a table needs a unit")
        values, unit = _quantity(info['values'], None)
        return cls(points * points_unit, values, unit)

    @classmethod
    def _load_file(cls, info):
        """Load a table from a column of a table file."""
        info = parseutils.parse_options(
            info, required={'table', 'column'}, optional={'hdu', 'units'})
        options = {} if info.get('hdu') is None else dict(hdu=info['hdu'])
        with parseutils.config_path('table'):
            table = astropy.table.Table.read(info['table'], **options)
        units = info.get('units') or {}
        axes = [axis for axis in _AXES if axis in table.colnames]
        if len(axes) != 1:
            raise ConfigError(
                f"{info['table']}: a table needs a column of its points, "
                f"named one of {list(_AXES)}; its columns are "
                f"{table.colnames}")
        column = info['column']
        if column not in table.colnames:
            raise ConfigError(
                f"{info['table']}: there is no column '{column}'; the "
                f"columns are {table.colnames}")

        def unit(name):
            return units.get(name, table[name].unit)
        axis = axes[0]
        points_unit = unit(axis) or (_KMS if axis == 'velocity' else None)
        if points_unit is None:
            raise ConfigError(
                f"{info['table']}: the column '{axis}' has no unit; give it "
                f"in units")
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
        points = self._points
        axis = ('velocity' if points.unit.is_equivalent(_KMS) else
                'wavelength' if points.unit.is_equivalent(u.m) else
                'frequency')
        values = self._values.tolist()
        return {
            axis: dict(values=points.value.tolist(), unit=str(points.unit)),
            'values': values if self._unit is None
            else dict(values=values, unit=str(self._unit))}

    def values(self) -> np.ndarray:
        """
        Return the values, in their unit.

        Returns
        -------
        ndarray
            The values.
        """
        return self._values.copy()

    def velocities(self, rest: Any = None) -> np.ndarray:
        """
        Return the points as velocities of a spectral axis.

        Parameters
        ----------
        rest : Quantity, optional
            The rest of the spectral axis (see gridutils.Coords); needed
            for points in wavelength or frequency.

        Returns
        -------
        ndarray
            The velocities (km/s).

        Raises
        ------
        ConfigError
            If the rest is needed but not given.
        """
        if self._points.unit.is_equivalent(_KMS):
            return self._points.to_value(_KMS)
        return self._points.to_value(_KMS, _doppler(rest))

    def at(
            self, velocities: Sequence[float] | np.ndarray, unit: Any,
            rest: Any = None
    ) -> np.ndarray:
        """
        Return the values at velocities of a spectral axis.

        Parameters
        ----------
        velocities : array_like
            The velocities (km/s).
        unit : str or Unit
            The unit of the option. Values with other units are converted
            to it; for a velocity, a width in wavelength or frequency is
            converted at each point.
        rest : Quantity, optional
            The rest of the spectral axis (see gridutils.Coords); needed
            for points or widths in wavelength or frequency.

        Returns
        -------
        ndarray
            The values at the velocities, in the unit of the option.

        Raises
        ------
        ConfigError
            If the rest is needed but not given, or the values cannot be
            converted to the unit of the option.
        """
        points = self.velocities(rest)
        values = self._values_in(u.Unit(unit), points, rest)
        order = np.argsort(points)
        return np.interp(velocities, points[order], values[order])

    def _values_in(self, unit, points, rest):
        """The values in a unit, at the points (velocities)."""
        if self._unit is None:
            return self._values
        values = self._values * self._unit
        if values.unit.is_equivalent(unit):
            return values.to_value(unit)
        if unit.is_equivalent(_KMS) and values.unit.is_equivalent(
                u.m, u.spectral()):
            # A width around each point, in wavelength or frequency
            doppler = _doppler(rest)
            centres = (points * _KMS).to(values.unit, doppler)
            low = (centres - values / 2).to_value(_KMS, doppler)
            high = (centres + values / 2).to_value(_KMS, doppler)
            return np.abs(high - low)
        raise ConfigError(
            f"the values of a table in '{self._unit}' cannot be converted "
            f"to '{unit}'")


class Varying:
    """
    The options of an object of a class (e.g. a PSF), some of which are
    tables of values along the spectral axis.

    The class lists its options that can vary, with their units, in
    VARYING_OPTIONS (e.g. dict(sigma='arcsec', ratio='')).

    Parameters
    ----------
    cls : type
        The class.
    info : dict
        The options that do not vary.
    tables : dict
        The tables of the options that vary (see SpectralTable).

    Raises
    ------
    ConfigError
        If the options, or the values of the tables, are invalid.
    """

    def __init__(
            self,
            cls: type,
            info: dict[str, Any],
            tables: dict[str, SpectralTable]
    ):
        # Check the options with the first value of each table, and the
        # values of each table with the first values of the others
        first = {name: table.values()[0] for name, table in tables.items()}
        options = parseutils.parse_options_for_callable(
            info | first, cls.__init__)
        for name, table in tables.items():
            with parseutils.config_path(name):
                for value in table.values():
                    cls(**options | {name: value})
        self._cls = cls
        self._constants = {
            name: value for name, value in options.items()
            if name not in tables}
        self._tables = tables

    @staticmethod
    def load_tables(
            cls: type, info: dict[str, Any]
    ) -> dict[str, SpectralTable]:
        """
        Remove the options of a configuration that are tables (dicts) and
        load them.

        Parameters
        ----------
        cls : type
            The class whose options they are (see VARYING_OPTIONS).
        info : dict
            The options; the tables are removed from it.

        Returns
        -------
        dict
            The tables, by option.

        Raises
        ------
        ConfigError
            If an option that cannot vary is a table.
        """
        tables = {}
        for name in [name for name, value in info.items()
                     if isinstance(value, Mapping)]:
            with parseutils.config_path(name):
                if name not in getattr(cls, 'VARYING_OPTIONS', {}):
                    raise ConfigError(
                        f"option '{name}' cannot vary along the spectral "
                        f"axis")
                tables[name] = SpectralTable.load(info.pop(name))
        return tables

    def cls(self) -> type:
        """
        Return the class.

        Returns
        -------
        type
            The class.
        """
        return self._cls

    def dump(self) -> dict[str, Any]:
        """
        Dump the options to their configuration.

        Returns
        -------
        dict
            The options, with the tables (see SpectralTable.dump).
        """
        return dict(type=self._cls.type()) | self._constants | {
            name: table.dump() for name, table in self._tables.items()}

    def at_velocities(
            self, velocities: Sequence[float] | np.ndarray, rest: Any = None
    ) -> list[Any]:
        """
        Return the objects at velocities of a spectral axis.

        Parameters
        ----------
        velocities : array_like
            The velocities (km/s).
        rest : Quantity, optional
            The rest of the spectral axis (see gridutils.Coords).

        Returns
        -------
        list
            An object of the class at each velocity.
        """
        values = {
            name: table.at(velocities, self._cls.VARYING_OPTIONS[name], rest)
            for name, table in self._tables.items()}
        return [
            self._cls(**self._constants | {
                name: float(value[i]) for name, value in values.items()})
            for i in range(len(velocities))]

    def velocity_range(self, rest: Any = None) -> tuple[float, float]:
        """
        Return the range of velocities that all the tables cover.

        Parameters
        ----------
        rest : Quantity, optional
            The rest of the spectral axis (see gridutils.Coords).

        Returns
        -------
        tuple of float
            The lowest and highest velocity (km/s).
        """
        ranges = [table.velocities(rest) for table in self._tables.values()]
        return (max(float(r.min()) for r in ranges),
                min(float(r.max()) for r in ranges))
