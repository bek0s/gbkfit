"""
What the responses of an instrument share (PSFs, LSFs and primary beams;
see Response): how they vary along the spectral axis. And what the PSFs
and the LSFs share (see SpreadFunction): their configurations, the kinds
of those that vary (tables of options, images at points, and sums and
convolutions of terms that vary), and helpers to draw them.
"""

import abc
import copy
import os
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Self, TypeAlias

import astropy.units as u
import numpy as np
import scipy.ndimage

import gbkfit.units
from gbkfit.config import (
    TypedSerializable, Unit, config_path, constructor_of, dump_quantity,
    is_quantity, option_types, parse_options_for_callable, unit_of,
    varying_options)
from gbkfit.grid import VelocityConvention
from gbkfit.io import write_image
from gbkfit.utils import numutils
from .varying import SpectralPoints, SpectralTable, as_points, merged_points


# The analytic PSFs and LSFs are drawn out to at least MIN_EXTENT scale
# lengths (e.g. sigma), and further for profiles with wide wings: until
# the wings beyond hold at most WING_FLUX of their flux. Beyond that
# they are 0, and the arrays are normalised to sum to 1.
MIN_EXTENT = 8
WING_FLUX = 0.01


# The size of the pixels of an image on the sky along x and y (arcsec);
# of one number, square pixels
Step: TypeAlias = Annotated[float | Sequence[float], Unit('arcsec')]

# Why a response that varies along the spectral axis is not drawn
NOT_DRAWN = (
    "a PSF, an LSF or a primary beam that varies along the spectral axis "
    "is drawn as it is at each point of it (see at)")


def odd_size(kernel: 'SpreadFunction', *size: float) -> tuple[int, ...]:
    """
    Return the size of the array of a PSF or LSF of the given size before
    rounding: each rounded up to odd.

    Raise RuntimeError unless it is finite: profiles whose wings are too
    heavy (e.g. a Moffat of beta close to its limit) put WING_FLUX of
    their flux at no finite distance.
    """
    if not np.all(np.isfinite(size)):
        raise RuntimeError(
            f"{type(kernel).__name__}: its wings are too heavy for an array "
            f"to hold {1 - WING_FLUX:.0%} of its flux")
    return tuple(int(numutils.roundu_odd(x)) for x in size)


def check_odd_centre(
        size: Sequence[int], offset: Sequence[int], desc: str
) -> None:
    """
    Raise RuntimeError unless each size + offset of an array of a PSF or
    LSF (desc) is odd: its centre, at size // 2 + offset, is then at the
    centre of the array, as the FFT convolution takes it.
    """
    if any(numutils.is_even(n + o) for n, o in zip(size, offset)):
        sums = ', '.join(f"{n} + {o}" for n, o in zip(size, offset))
        raise RuntimeError(
            f"invalid {desc} size: each size + offset must be odd; they are "
            f"{sums}")


def embed(
        kernel: np.ndarray, size: Sequence[int], offset: Sequence[int]
) -> np.ndarray:
    """
    Put a kernel of odd shape (numpy order), normalised to sum to 1, in an
    array of the given size (FITS order), with its centre at size // 2 +
    offset on each axis, as the analytic PSFs and LSFs put theirs. Like
    theirs, a kernel larger than the array is cut to it, and normalised
    again.
    """
    shape = tuple(size)[::-1]
    offset = tuple(offset)[::-1]
    data = np.zeros(shape)
    into, of = [], []
    for n, k, o in zip(shape, kernel.shape, offset):
        start = n // 2 + o - k // 2
        low, high = max(start, 0), min(start + k, n)
        into.append(slice(low, high))
        of.append(slice(low - start, high - start))
    data[tuple(into)] = kernel[tuple(of)]
    cut = any(x.stop - x.start < k for x, k in zip(of, kernel.shape))
    return data / np.sum(data) if cut else data


def resample(
        image: np.ndarray,
        step_ratio: Sequence[float],
        size: Sequence[int],
        offset: Sequence[int]
) -> np.ndarray:
    """
    Resample a measured PSF or LSF (an image, numpy order) to an array of
    the given size (FITS order), of pixels step_ratio times its own along
    each axis (FITS order): its centre at size // 2 + offset on each axis,
    as the analytic PSFs and LSFs put theirs, with splines of order 5, and
    normalised to sum to 1.
    """
    shape = tuple(size)[::-1]
    step_ratio = tuple(step_ratio)[::-1]
    offset = tuple(offset)[::-1]
    # The pixels of the array in those of the image, from centre to centre
    pixels = np.meshgrid(*(
        (np.arange(n) - (n // 2 + o)) * r + (k / 2 - 0.5)
        for n, r, o, k in zip(shape, step_ratio, offset, image.shape)),
        indexing='ij')
    data = scipy.ndimage.map_coordinates(
        image, pixels, order=5, mode='grid-constant')
    return data / np.sum(data)


def dump_image(
        kernel: 'SpreadFunction',
        data: np.ndarray,
        name: str,
        prefix: str,
        dump_path: bool,
        overwrite: bool
) -> dict[str, Any]:
    """
    Write the data of a measured PSF or LSF to the file {prefix}{name}.fits,
    and return its configuration: its type and the file (see dump).
    """
    filename = f'{prefix}{name}.fits'
    write_image(filename, data, overwrite=overwrite)
    return dict(
        type=kernel.type(),
        file=filename if dump_path else os.path.basename(filename))


def _dump_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Options as configurations have them: Quantities as their text."""
    return {
        name: gbkfit.units.format_quantity(value)
        if isinstance(value, u.Quantity) else value
        for name, value in options.items()}


def _common_range(
        ranges: Sequence[tuple[float, float]]
) -> tuple[float, float]:
    """The range of velocities where all of several ranges are."""
    return max(low for low, _ in ranges), min(high for _, high in ranges)


class Response(TypedSerializable, abc.ABC):
    """
    A response of an instrument: a PSF, an LSF or a primary beam. It is
    the same along the spectral axis, or varies along it: at each point
    of it, it is one that does not (see at).
    """

    @abc.abstractmethod
    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Dump the response to its configuration.

        Parameters
        ----------
        prefix : str, optional
            The start of the names of the files of its data (e.g. an
            image), if any.
        dump_path : bool, optional
            Whether the configuration has the paths of the files, or only
            their names.
        overwrite : bool, optional
            Whether to overwrite existing files.

        Returns
        -------
        dict
            The options.
        """

    def varies(self) -> bool:
        """
        Check whether the response varies along the spectral axis.

        Returns
        -------
        bool
            Whether it varies.
        """
        return False

    def at(
            self,
            points: u.Quantity,
            convention: VelocityConvention | None = None
    ) -> list['Response']:
        """
        Return the response at points of the spectral axis: one that does
        not vary at each.

        Parameters
        ----------
        points : Quantity
            The points (1D): wavelengths, frequencies or velocities.
        convention : VelocityConvention, optional
            The velocity convention of the spectral axis; needed between
            velocities and the points of one that varies, if they are of
            another kind.

        Returns
        -------
        list
            The response at each point: itself, unless it varies.

        Raises
        ------
        TypeError
            If the points are not wavelengths, frequencies or velocities.
        ValueError
            If the points are not 1D, or the convention is needed but not
            given.
        """
        points = as_points(points)
        if points.ndim != 1:
            raise ValueError(
                f"the points of the spectral axis to give a response at are "
                f"a sequence (1D); their shape is {points.shape}")
        if not self.varies():
            return [self] * len(points)
        return self._at_impl(points, convention)

    def _at_impl(
            self,
            points: u.Quantity,
            convention: VelocityConvention | None
    ) -> list['Response']:
        """
        The response at each of the points (1D) of one that varies (see
        at): those that vary override it.
        """
        raise NotImplementedError(f"{type(self).__name__}._at_impl")

    def _check_not_varying(self) -> None:
        """
        Raise RuntimeError if it varies along the spectral axis: then it is
        drawn (or applied) as it is at each point of it (see at), not
        itself.
        """
        if self.varies():
            raise RuntimeError(NOT_DRAWN)


class SpreadFunction(Response):
    """
    A PSF or an LSF: what they share beyond what responses do. Their
    analytic ones can vary by tables of their options (see
    AnalyticSpreadFunction), and their sums and convolutions vary where
    their terms do (see SpreadFunctionTerms, and _TERMS_OPTION). Those
    that vary are known between the points of the spectral axis they are
    given at (see velocity_range and check_covers).
    """

    # The option of the terms of the sums and the convolutions of the
    # family (e.g. psfs)
    _TERMS_OPTION: str

    def velocity_range(
            self, convention: VelocityConvention | None = None
    ) -> tuple[float, float]:
        """
        Return the range of velocities of the spectral axis that the PSF
        or the LSF is known at.

        Parameters
        ----------
        convention : VelocityConvention, optional
            The velocity convention of the spectral axis.

        Returns
        -------
        tuple of float
            The lowest and highest velocity (km/s); infinite unless it
            varies.
        """
        return -np.inf, np.inf

    def check_covers(
            self,
            velocities: np.ndarray,
            convention: VelocityConvention | None,
            desc: str
    ) -> None:
        """
        Check that the PSF or the LSF is known at velocities of the
        spectral axis (e.g. of the channels of data), to a millionth of
        their spacing (of a km/s, for one): beyond the points it is given
        at, it would be that of the nearest point.

        Parameters
        ----------
        velocities : ndarray
            The velocities (km/s), evenly spaced.
        convention : VelocityConvention, optional
            The velocity convention of the spectral axis.
        desc : str
            What the PSF or the LSF is, in messages (e.g. 'the PSF').

        Raises
        ------
        ValueError
            If it is not known at a velocity.
        """
        low, high = self.velocity_range(convention)
        first, last = np.min(velocities), np.max(velocities)
        spacing = (last - first) / (len(velocities) - 1) \
            if len(velocities) > 1 else 1.0
        margin = 1e-6 * spacing
        if first < low - margin or last > high + margin:
            raise ValueError(
                f"{desc} is given from {low:g} to {high:g} km/s, but the "
                f"channels of the data go from {first:g} to {last:g} km/s")


class AnalyticSpreadFunction(SpreadFunction):
    """
    A PSF or an LSF of an analytic profile, of a scale (e.g. its sigma),
    drawn out to its extent (see MIN_EXTENT and _wings).

    Configurations give its options in one of several forms (its
    config_constructors: e.g. its scale, or its FWHM), and dumps write
    those it was given (see _given_as). Its options marked Varying can be
    tables of their values along the spectral axis, which make one that
    varies (see varying).

    Parameters
    ----------
    scale : float or Quantity
        The scale.
    """

    config_constructors = ('__init__', 'from_fwhm')

    @classmethod
    def load(cls, info: dict[str, Any]) -> SpreadFunction:
        """
        Load a PSF or an LSF from its configuration.

        Options that can vary (see Varying) given as tables make one that
        varies along the spectral axis (see varying): a table is a dict
        of points, inline or from a file (see SpectralTable); a Quantity,
        of a value and a unit, is not one.

        Parameters
        ----------
        info : dict
            The options.

        Returns
        -------
        SpreadFunction
            The PSF or the LSF.
        """
        tables = {}
        for name in varying_options(constructor_of(cls, info)[0]):
            if SpectralTable.is_config(info.get(name)):
                with config_path(name):
                    tables[name] = SpectralTable.load(info[name])
        if tables:
            return cls.varying(**(info | tables))
        return super().load(info)

    @classmethod
    def varying(cls, **options: Any) -> 'SpreadFunctionTables':
        """
        Make a PSF or an LSF of this class whose options vary along the
        spectral axis.

        Parameters
        ----------
        **options
            The options of a constructor of the class (see
            config_constructor); those that vary (see the options of the
            class marked Varying) are tables (SpectralTable).

        Returns
        -------
        SpreadFunctionTables
            The PSF or the LSF (see PSFTables and LSFTables).

        Raises
        ------
        ValueError
            If no option is a table, an option that cannot vary is one,
            or the options are invalid.
        """
        return cls._tables_class()(cls, options)

    def __init__(self, scale: float | u.Quantity):
        self._scale = scale

    @staticmethod
    @abc.abstractmethod
    def _tables_class() -> type['SpreadFunctionTables']:
        """The class of those of its family whose options vary."""

    def _given_as(self, **given: Any) -> Self:
        """Itself, with the options it was given, which dump writes."""
        self._given = given
        return self

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(type=self.type()) | _dump_options(self._given)

    @abc.abstractmethod
    def _profile(self, x: np.ndarray, scale: float) -> np.ndarray:
        """
        Its profile at the distances x from its centre (of a PSF, the
        elliptical radii), to any factor, for its scale (in the unit of x).
        """

    @abc.abstractmethod
    def _wings(self) -> float:
        """
        How many scale lengths from its centre hold all but WING_FLUX of
        its flux.
        """

    def _extent(self, scale: float) -> float:
        """
        How far from its centre it is drawn (see MIN_EXTENT), for its
        scale in the unit of the axes it is drawn on.
        """
        return scale * max(MIN_EXTENT, self._wings())

    def _drawn(self, x: np.ndarray, scale: float) -> np.ndarray:
        """
        Its profile at the distances x from its centre (see _profile), 0
        beyond its extent, normalised to sum to 1.
        """
        data = self._profile(x, scale)
        data[np.abs(x) > self._extent(scale)] = 0
        return data / np.sum(data)


class SpreadFunctionTerms(SpreadFunction):
    """
    A PSF or an LSF of terms of its family (a sum or a convolution, of
    _terms): it varies where any of them does, and at each point of the
    spectral axis it is that of their terms there.
    """

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        # (the files of each term are named with its own prefix: term0_,
        # term1_, ...)
        terms = [
            term.dump(f'{prefix}term{i}_', dump_path, overwrite)
            for i, term in enumerate(self._terms)]
        return dict(type=self.type()) | {self._TERMS_OPTION: terms}

    def __init__(self, terms: Sequence[SpreadFunction], desc: str):
        if not terms:
            raise ValueError(f"{desc} needs at least one term")
        self._terms = tuple(terms)

    def varies(self) -> bool:
        return any(term.varies() for term in self._terms)

    def _at_impl(
            self,
            points: u.Quantity,
            convention: VelocityConvention | None
    ) -> list[SpreadFunction]:
        per_term = [term.at(points, convention) for term in self._terms]
        return [self._with_terms(terms) for terms in zip(*per_term)]

    def _with_terms(self, terms: Sequence[SpreadFunction]) -> Self:
        """The same, of other terms (e.g. those at a point; see at)."""
        other = copy.copy(self)
        other._terms = tuple(terms)
        return other

    def velocity_range(
            self, convention: VelocityConvention | None = None
    ) -> tuple[float, float]:
        return _common_range(
            [term.velocity_range(convention) for term in self._terms])


class SpreadFunctionSum(SpreadFunctionTerms):
    """
    A sum of PSFs or of LSFs, each with its fraction of the light.

    Parameters
    ----------
    terms : Sequence of SpreadFunction
        The terms.
    weights : Sequence of float
        The weight of each term, positive; they are normalised to sum to 1.
    desc : str
        What the sum is, in messages (e.g. 'a sum of PSFs').
    """

    @staticmethod
    def type() -> str:
        return 'sum'

    @staticmethod
    def type_title() -> str:
        return 'Sum'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return super().dump(prefix, dump_path, overwrite) | dict(
            weights=self._weights.tolist())

    def __init__(
            self,
            terms: Sequence[SpreadFunction],
            weights: Sequence[float],
            desc: str
    ):
        super().__init__(terms, desc)
        weights = np.asarray(weights, dtype=float)
        n = len(self._terms)
        if weights.shape != (n,) or not np.all(weights > 0):
            raise ValueError(
                f"{desc} needs a positive weight for each of its {n} terms; "
                f"its weights are {weights.tolist()}")
        self._weights = weights / weights.sum()


class SpreadFunctionConvolution(SpreadFunctionTerms):
    """
    A convolution of PSFs or of LSFs.
    """

    @staticmethod
    def type() -> str:
        return 'convolution'

    @staticmethod
    def type_title() -> str:
        return 'Convolution'


class VaryingSpreadFunction(SpreadFunction):
    """
    A PSF or an LSF that always varies along the spectral axis (by tables
    of its options, or images at points of it): it is drawn as it is at
    each point (see at), not itself.
    """

    def varies(self) -> bool:
        return True

    def _size_impl(self, *args: Any) -> Any:
        raise RuntimeError(NOT_DRAWN)

    def _asarray_impl(self, *args: Any) -> Any:
        raise RuntimeError(NOT_DRAWN)


class SpreadFunctionTables(VaryingSpreadFunction):
    """
    A PSF or an LSF of an analytic class whose options vary along the
    spectral axis, as tables of their values at points of it (see
    SpectralTable): at each point, the one of the class of their values
    there (see at). It is made by the varying of the class (e.g.
    PSFMoffat.varying).

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
        The options; those that vary are tables (SpectralTable), at least
        one.

    Raises
    ------
    ValueError
        If no option is a table, an option that cannot vary is one, a
        table of a Quantity has no unit, or the options are invalid, or
        so are the values of the tables at their points (see _check).
    """

    def type(self) -> str:
        return self._cls.type()

    def type_title(self) -> str:
        return self._cls.type_title()

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(type=self.type()) | _dump_options(self._constants) | {
            name: table.dump() for name, table in self._tables.items()}

    def __init__(self, cls: type, options: dict[str, Any]):
        func, self._make = constructor_of(cls, options)
        tables = {
            name: value for name, value in options.items()
            if isinstance(value, SpectralTable)}
        if not tables:
            raise ValueError(
                f"none of the options of {cls.__qualname__} is a table of "
                f"values along the spectral axis")
        types = option_types(func)
        # The options that are Quantities, whose values are in the units
        # of their tables, and the unit of the values of the others (that
        # of their annotations)
        self._quantities = set()
        self._units = {}
        for name, table in tables.items():
            if name not in varying_options(func):
                raise ValueError(
                    f"option '{name}' of {cls.__qualname__} cannot vary "
                    f"along the spectral axis")
            given = table.unit()
            if is_quantity(types[name]):
                if given is None:
                    raise ValueError(
                        f"option '{name}' of {cls.__qualname__} has a unit: "
                        f"give that of the values of its table (unit, of an "
                        f"inline table; of a table file, its column in "
                        f"units, e.g. units: {{<column>: Angstrom}})")
                self._quantities.add(name)
                continue
            unit = u.Unit(unit_of(types[name]) or '')
            if given is not None and not given.is_equivalent(unit):
                raise ValueError(
                    f"the values of the table of option '{name}' are in "
                    f"'{gbkfit.units.format_unit(given)}', which cannot be "
                    f"converted to '{gbkfit.units.format_unit(unit)}'")
            self._units[name] = unit
        self._cls = cls
        self._constants = parse_options_for_callable(
            {name: value for name, value in options.items()
             if name not in tables},
            func, ignore_params=set(tables))
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
        points = merged_points(
            [table.points() for table in self._tables.values()])
        if points is not None:
            self._at_impl(points, None)

    def _at_impl(
            self,
            points: u.Quantity,
            convention: VelocityConvention | None
    ) -> list[SpreadFunction]:
        # (one of the class at each point, of the values of the tables
        # there)
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
        if name in self._quantities:
            return values
        if isinstance(values, u.Quantity):
            values = values.to_value(self._units[name])
        return [float(x) for x in values]

    def velocity_range(
            self, convention: VelocityConvention | None = None
    ) -> tuple[float, float]:
        # (where all the tables are)
        return _common_range([
            table.velocity_range(convention)
            for table in self._tables.values()])


class SpreadFunctionImages(VaryingSpreadFunction):
    """
    A measured PSF or LSF that changes along the spectral axis, given as
    images (of a PSF) or profiles (of an LSF) at points of it. At each
    point it is the image of the images of the two nearest points,
    interpolated linearly; beyond the points, that of the nearest image
    (see SpectralPoints.blend).

    Parameters
    ----------
    data : ndarray
        The images, along the first axis, finite; each holds light.
    points : Quantity
        The point of each image: wavelengths, frequencies or velocities.
    desc : str
        What the images are, in messages (e.g. 'PSF images').
    """

    def __init__(self, data: np.ndarray, points: u.Quantity, desc: str):
        self._points = SpectralPoints(points, desc)
        if len(self._points) != len(data):
            raise ValueError(
                f"{desc} need a point each; there are {len(data)}, and "
                f"{len(self._points)} points")
        light = data.sum(axis=tuple(range(1, data.ndim)), keepdims=True)
        if not np.all(light > 0):
            point = self._points.quantity()[np.argmin(light.ravel() > 0)]
            raise ValueError(
                f"{desc} must each hold light; that at "
                f"{gbkfit.units.format_quantity(point)} holds none")
        # (each image holds all the light)
        self._data = data / light

    @abc.abstractmethod
    def _of_image(self, data: np.ndarray) -> SpreadFunction:
        """The PSF or the LSF of one image (see PSFImage and LSFProfile)."""

    def _dump_points(self) -> dict[str, Any]:
        """The configuration of the points."""
        return dump_quantity(self._points.quantity())

    def _at_impl(
            self,
            points: u.Quantity,
            convention: VelocityConvention | None
    ) -> list[SpreadFunction]:
        # (one image for each blend)
        blends = self._points.blend(points, convention)
        images = {
            (i, j, t): self._of_image(
                (1 - t) * self._data[i] + t * self._data[j])
            for i, j, t in set(blends)}
        return [images[blend] for blend in blends]

    def velocity_range(
            self, convention: VelocityConvention | None = None
    ) -> tuple[float, float]:
        return self._points.velocity_range(convention)
