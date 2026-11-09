"""
Options of PSFs and LSFs that vary along the spectral axis, and the
points of the spectral axis that things are given at.

An option that varies is a table of its values at points of the spectral
axis: wavelengths, frequencies or velocities (SpectralTable). Between the
points the values are interpolated linearly, and beyond them they are
those of the nearest point (see SpectralPoints). A PSF or an LSF whose
options vary is made by the varying of its class, e.g.

    alpha = SpectralTable([4800, 7000, 9300] * u.AA, [0.75, 0.65, 0.58])
    psf = PSFMoffat.varying(alpha=alpha, beta=2.5)
    psf.at([6500] * u.AA)  # [the PSFMoffat at 6500 Angstrom]

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

from collections.abc import Mapping, Sequence
from typing import Any

import astropy.units as u
import numpy as np

import gbkfit.units
from gbkfit.config import (
    DataFile, Serializable, check_options, dump_quantity, split_data_file)
from gbkfit.grid import VelocityConvention
from gbkfit.io import read_column, read_table


__all__ = [
    'SpectralTable'
]


# The velocity unit of the model
_KMS = u.km / u.s


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
    if convention is None:
        raise ValueError(
            "velocities and wavelengths or frequencies are converted with "
            "the convention of the velocities of the spectral axis, which "
            "is not given")
    return points.to_value(unit, convention.equivalency())


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


def merged_points(points: Sequence[u.Quantity]) -> u.Quantity | None:
    """
    Return the points of the spectral axis of several sets (e.g. of
    tables), each once, in the unit of the first set; or None if some are
    velocities and others wavelengths or frequencies, which only the
    convention of the velocities of the spectral axis puts together.

    Parameters
    ----------
    points : Sequence of Quantity
        The sets of points, at least one.

    Returns
    -------
    Quantity or None
        The points.
    """
    unit = points[0].unit
    if not all(x.unit.is_equivalent(unit, u.spectral()) for x in points):
        return None
    return np.unique(np.concatenate([
        x.to_value(unit, u.spectral()) for x in points])) * unit


class SpectralPoints:
    """
    The points of the spectral axis that things are given at (e.g. the
    values of a table, or images): wavelengths, frequencies or
    velocities, each once. Between the points things are blended linearly
    along the axis of the points (e.g. in wavelength), and beyond them
    they are those of the nearest point (see interpolate and blend).

    Parameters
    ----------
    points : Quantity
        The points (1D).
    desc : str
        What they are the points of, in messages (e.g. 'a table').

    Raises
    ------
    TypeError
        If they are not wavelengths, frequencies or velocities.
    ValueError
        If there are none, or they are not 1D, finite and distinct.
    """

    def __init__(self, points: u.Quantity, desc: str):
        points = as_points(points)
        if points.ndim != 1 or points.size == 0:
            raise ValueError(
                f"the points of {desc} are a sequence of one or more; their "
                f"shape is {points.shape}")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"the points of {desc} must be finite")
        if np.unique(points).size != points.size:
            raise ValueError(f"the points of {desc} must be distinct")
        self._points = points
        # (the order of the points along their axis, and they in it)
        self._order = np.argsort(points.value)
        self._sorted = points.value[self._order]

    def __len__(self) -> int:
        return self._points.size

    def quantity(self) -> u.Quantity:
        """
        Return the points.

        Returns
        -------
        Quantity
            The points.
        """
        return self._points.copy()

    def interpolate(
            self,
            values: np.ndarray,
            at: u.Quantity,
            convention: VelocityConvention | None = None
    ) -> np.ndarray:
        """
        Return values given at the points at other points.

        Parameters
        ----------
        values : ndarray
            The value at each point.
        at : Quantity
            The other points, of any kind (1D).
        convention : VelocityConvention, optional
            The velocity convention of the spectral axis; needed
            between velocities and the other kinds of points.

        Returns
        -------
        ndarray
            The values at the other points.

        Raises
        ------
        ValueError
            If the convention is needed but not given, or has no rest.
        """
        return np.interp(
            self._converted(at, convention), self._sorted,
            values[self._order])

    def blend(
            self,
            at: u.Quantity,
            convention: VelocityConvention | None = None
    ) -> list[tuple[int, int, float]]:
        """
        Return how to blend things given at the points (e.g. images) at
        other points.

        Parameters
        ----------
        at : Quantity
            The other points, of any kind (1D).
        convention : VelocityConvention, optional
            The velocity convention of the spectral axis; needed
            between velocities and the other kinds of points.

        Returns
        -------
        list of tuple
            For each other point, (i, j, t): it takes 1 - t of the thing
            of the point i and t of that of the point j, the two nearest
            points; beyond the points, all of the nearest: of i (t = 0)
            below the first, and of j (t = 1) above the last.

        Raises
        ------
        ValueError
            If the convention is needed but not given, or has no rest.
        """
        order = self._order
        positions = np.interp(
            self._converted(at, convention), self._sorted,
            np.arange(len(order)))
        result = []
        for position in positions:
            k = min(int(position), len(order) - 2) if len(order) > 1 else 0
            t = float(position - k)
            result.append((
                int(order[k]), int(order[k + 1] if t > 0 else order[k]), t))
        return result

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

    def _converted(
            self, at: u.Quantity, convention: VelocityConvention | None
    ) -> np.ndarray:
        """Other points, in the unit of the points."""
        return _convert(as_points(at), self._points.unit, convention)


class SpectralTable(Serializable):
    """
    The values of an option at points of the spectral axis.

    Between the points the values are interpolated linearly along the axis
    of the points (e.g. in wavelength), and beyond them they are those of
    the nearest point (see SpectralPoints).

    Parameters
    ----------
    points : Quantity
        The points: wavelengths, frequencies or velocities, each once.
    values : array_like or Quantity
        The value at each point, finite; of a Quantity, in its unit.
    unit : str or Unit, optional
        The unit of the values (of values that are not a Quantity); by
        default, the units of the option.

    Raises
    ------
    TypeError
        If the points are not wavelengths, frequencies or velocities.
    ValueError
        If the points are invalid (see SpectralPoints), there is not a
        finite value for each, or the values are a Quantity of another
        unit than unit.
    """

    config_constructors = ('__init__', 'from_file')

    @staticmethod
    def is_config(value: Any) -> bool:
        """
        Check whether the configuration of an option is a table (see
        load): a dict of points, inline or of a file. A Quantity, of a
        value and a unit, is not one.

        Parameters
        ----------
        value : Any
            The configuration of the option.

        Returns
        -------
        bool
            Whether it is a table.
        """
        return isinstance(value, Mapping) and 'points' in value

    def __init__(
            self,
            points: u.Quantity,
            values: Sequence[float] | np.ndarray | u.Quantity,
            unit: str | u.UnitBase | None = None
    ):
        check_options(SpectralTable.__init__, locals())
        self._points = SpectralPoints(points, "a table")
        if isinstance(unit, str):
            unit = gbkfit.units.parse_unit(unit)
        if isinstance(values, u.Quantity):
            if unit is not None and values.unit != unit:
                raise ValueError(
                    f"the values of a table are in "
                    f"'{gbkfit.units.format_unit(values.unit)}', and its unit "
                    f"is '{gbkfit.units.format_unit(unit)}'")
            # (dimensionless values are in the units of the option)
            unit = (None if values.unit == u.dimensionless_unscaled
                    else values.unit)
            values = values.value
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self._points),):
            raise ValueError(
                f"a table needs a value at each of its points; it has "
                f"{len(self._points)} points and {values.size} values")
        if not np.all(np.isfinite(values)):
            raise ValueError("the values of a table must be finite")
        self._values = values
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
        return cls(point_values, read_column(table, column, units.get(column)))

    def dump(self) -> dict[str, Any]:
        """
        Dump the table to its configuration (with its points and values).

        Returns
        -------
        dict
            The configuration.
        """
        unit = ({} if self._unit is None
                else dict(unit=gbkfit.units.format_unit(self._unit)))
        return dict(
            points=dump_quantity(self._points.quantity()),
            values=self._values.tolist()) | unit

    def points(self) -> u.Quantity:
        """
        Return the points.

        Returns
        -------
        Quantity
            The points.
        """
        return self._points.quantity()

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
        values = self._points.interpolate(self._values, points, convention)
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
        return self._points.velocity_range(convention)
