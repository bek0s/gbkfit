"""
What the PSFs and the LSFs share: their configurations, how they vary
along the spectral axis (see SpreadFunction), and helpers to draw them.
"""

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self, TypeVar

import astropy.units as u
import astropy.wcs
import numpy as np

import gbkfit.units
from gbkfit.config import TypedParser, TypedSerializable, dump_quantity
from gbkfit.grid import VelocityConvention
from . import varying

if TYPE_CHECKING:
    from .lsfs import LSF
    from .psfs import PSF

# The terms of a sum or a convolution: all PSFs, or all LSFs
_Term = TypeVar('_Term', 'PSF', 'LSF')


# The analytic PSFs and LSFs are drawn out to at least MIN_EXTENT scale
# lengths (e.g. sigma), and further for profiles with wide wings: until
# the wings beyond hold at most WING_FLUX of their flux. Beyond that
# they are 0, and the arrays are normalised to sum to 1.
MIN_EXTENT = 8
WING_FLUX = 0.01


def check_finite_size(kernel: 'PSF | LSF', *size: float) -> None:
    """
    Raise RuntimeError unless the size of the array of a PSF or LSF is
    finite: profiles whose wings are too heavy (e.g. a Moffat of beta
    close to its limit) put WING_FLUX of their flux at no finite distance.
    """
    if not np.all(np.isfinite(size)):
        raise RuntimeError(
            f"{kernel.__class__.__name__} {kernel.dump()}: its wings are too "
            f"heavy for an array to hold {1 - WING_FLUX:.0%} of its flux")


def embed(
        kernel: np.ndarray, size: Sequence[int], offset: Sequence[int]
) -> np.ndarray:
    """
    Put a kernel of odd shape (numpy order) in an array of the given size
    (FITS order), with its centre at size // 2 + offset on each axis, as
    the analytic PSFs and LSFs put theirs. It must fit.
    """
    shape = tuple(size)[::-1]
    offset = tuple(offset)[::-1]
    data = np.zeros(shape)
    slices = []
    for n, k, o in zip(shape, kernel.shape, offset):
        start = n // 2 + o - k // 2
        if start < 0 or start + k > n:
            raise RuntimeError(
                f"a kernel of size {k} does not fit in an array of size {n}")
        slices.append(slice(start, start + k))
    data[tuple(slices)] = kernel
    return data


def spectral_points(wcs: astropy.wcs.WCS, n: int) -> u.Quantity | None:
    """
    Return the points of the first n pixels of the spectral axis of a WCS,
    or None if it has no spectral axis.
    """
    if wcs.wcs.spec < 0:
        return None
    spectral = wcs.sub([wcs.wcs.spec + 1])
    return u.Quantity(
        spectral.pixel_to_world_values(np.arange(n)), spectral.wcs.cunit[0])


def sum_weights(
        weights: Sequence[float], n: int, desc: str
) -> np.ndarray:
    """
    Return the weights of the terms of a sum (one for each of n terms),
    each positive, normalised to sum to 1; desc names the sum in messages.
    """
    weights = np.asarray(weights, dtype=float)
    if n < 1:
        raise ValueError(f"{desc} needs at least one term")
    if weights.shape != (n,) or not np.all(weights > 0):
        raise ValueError(
            f"{desc} needs a positive weight for each of its {n} terms; "
            f"its weights are {weights.tolist()}")
    return weights / weights.sum()


def dump_terms(
        parser: TypedParser,
        terms: Sequence[_Term],
        prefix: str,
        dump_path: bool,
        overwrite: bool
) -> list[dict[str, Any]]:
    """
    Return the options of the terms of a sum or a convolution, with the
    files of each named with its own prefix (term0_, term1_, ...).
    """
    return [
        parser.dump(
            term, prefix=f'{prefix}term{i}_', dump_path=dump_path,
            overwrite=overwrite)
        for i, term in enumerate(terms)]


class SpreadFunction(TypedSerializable, abc.ABC):
    """
    A PSF or an LSF: what they share, which is how they are configured
    and how they vary along the spectral axis (see PSF and LSF for how
    they are drawn).
    """

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'SpreadFunction':
        """
        Load a PSF or an LSF from its configuration.

        Options that can vary (see Varying) given as tables make one that
        varies along the spectral axis (see varying, and the varying of
        its class).

        Parameters
        ----------
        info : dict
            The options.

        Returns
        -------
        SpreadFunction
            The PSF or the LSF.
        """
        options = varying.load_tables(cls, info)
        if any(isinstance(x, varying.SpectralTable)
               for x in options.values()):
            return cls.varying(**options)
        return super().load(info)

    @classmethod
    def varying(cls, **options: Any) -> 'SpreadFunction':
        """
        Make a PSF or an LSF of this class whose options vary along the
        spectral axis.

        Parameters
        ----------
        **options
            The options of the class; those that vary (see the options of
            the class marked Varying) are tables (SpectralTable).

        Returns
        -------
        SpreadFunction
            The PSF or the LSF (see PSFVarying and LSFVarying).

        Raises
        ------
        ValueError
            If an option that cannot vary is a table, or the options are
            invalid.
        """
        return cls._varying_class()(varying._VaryingOptions(cls, options))

    @staticmethod
    @abc.abstractmethod
    def _varying_class() -> type['VaryingSpreadFunction']:
        """The class of those of the family whose options vary."""

    @abc.abstractmethod
    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Dump the PSF or the LSF to its configuration.

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
        Check whether the PSF or the LSF varies along the spectral axis.

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
    ) -> 'SpreadFunction | list[SpreadFunction]':
        """
        Return the PSF or the LSF at points of the spectral axis.

        Parameters
        ----------
        points : Quantity
            Wavelengths, frequencies or velocities: one, or a sequence.
        convention : VelocityConvention, optional
            The velocity convention of the spectral axis; needed between
            velocities and the points of one that varies, if they are of
            another kind.

        Returns
        -------
        SpreadFunction or list of SpreadFunction
            The PSF or the LSF at the point, or at each point: itself,
            unless it varies.
        """
        return varying.per_point(points, lambda points_: [self] * len(points_))

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


class GivenOptions(SpreadFunction):
    """
    A PSF or an LSF of several forms of options (its config_constructors,
    e.g. its scale or its FWHM), which dumps those it was given in (see
    _given_as).
    """

    def _given_as(self, **given: Any) -> Self:
        """Itself, with the options it was given, which dump writes."""
        self._given = given
        return self

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(type=self.type()) | {
            name: gbkfit.units.format_quantity(value)
            if isinstance(value, u.Quantity) else value
            for name, value in self._given.items()}


class SpreadFunctionTerms(SpreadFunction):
    """
    A PSF or an LSF of terms of its family (a sum or a convolution, of
    _terms): it varies where any of them does, and at each point of the
    spectral axis it is that of their terms there.
    """

    _terms: tuple[SpreadFunction, ...]

    @abc.abstractmethod
    def _of_terms(self, terms: Sequence[SpreadFunction]) -> SpreadFunction:
        """The same of other terms (e.g. those at a point; see at)."""

    def varies(self) -> bool:
        return any(term.varies() for term in self._terms)

    def at(
            self,
            points: u.Quantity,
            convention: VelocityConvention | None = None
    ) -> 'SpreadFunction | list[SpreadFunction]':
        if not self.varies():
            return super().at(points, convention)

        def at_points(points_: u.Quantity) -> list[SpreadFunction]:
            per_term = [term.at(points_, convention) for term in self._terms]
            return [self._of_terms(terms) for terms in zip(*per_term)]
        return varying.per_point(points, at_points)

    def velocity_range(
            self, convention: VelocityConvention | None = None
    ) -> tuple[float, float]:
        # (where all of the terms are known)
        ranges = [term.velocity_range(convention) for term in self._terms]
        return max(r[0] for r in ranges), min(r[1] for r in ranges)


class VaryingSpreadFunction(SpreadFunction):
    """
    A PSF or an LSF of a type whose options vary along the spectral axis:
    one of the type at each channel (see at). It is not drawn itself. It
    is made by the varying of the class of the type (e.g.
    PSFMoffat.varying).
    """

    def type(self) -> str:
        return self._options.cls().type()

    def type_title(self) -> str:
        return self._options.cls().type_title()

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return self._options.dump()

    def __init__(self, options: 'varying._VaryingOptions'):
        self._options = options

    def varies(self) -> bool:
        return True

    def at(
            self,
            points: u.Quantity,
            convention: VelocityConvention | None = None
    ) -> 'SpreadFunction | list[SpreadFunction]':
        return varying.per_point(
            points, lambda points_: self._options.at(points_, convention))

    def velocity_range(
            self, convention: VelocityConvention | None = None
    ) -> tuple[float, float]:
        return self._options.velocity_range(convention)

    def _size_impl(self, *args: Any) -> Any:
        raise RuntimeError(
            "a PSF or an LSF that varies along the spectral axis has a size "
            "at each channel (see at)")

    def _asarray_impl(self, *args: Any) -> np.ndarray:
        raise RuntimeError(
            "a PSF or an LSF that varies along the spectral axis is drawn at "
            "each channel (see at)")


class SpreadFunctionImages(SpreadFunction):
    """
    A measured PSF or LSF that changes along the spectral axis, given as
    images (of a PSF) or profiles (of an LSF) at points of it. At each
    channel it is the image of the images of the two nearest points,
    interpolated linearly; beyond the points, that of the nearest image
    (see varying.blend).

    Parameters
    ----------
    data : ndarray
        The images, along the first axis, finite.
    points : Quantity
        The point of each image: wavelengths, frequencies or velocities.
    desc : str
        What the images are, in messages (e.g. 'PSF images').
    """

    def __init__(self, data: np.ndarray, points: u.Quantity, desc: str):
        points = u.Quantity(points, dtype=float)
        if points.shape != data.shape[:1]:
            raise ValueError(
                f"{desc} need a point each; there are {len(data)}, and "
                f"{points.size} points")
        if np.unique(points).size != points.size:
            raise ValueError(f"the points of {desc} must be distinct")
        # (each image holds all the light)
        self._data = data / data.sum(
            axis=tuple(range(1, data.ndim)), keepdims=True)
        self._points = points

    @abc.abstractmethod
    def _image(self, data: np.ndarray) -> SpreadFunction:
        """The PSF or the LSF of one image (see PSFImage and LSFImage)."""

    def _dump_points(self) -> dict[str, Any]:
        """The configuration of the points."""
        return dump_quantity(self._points)

    def varies(self) -> bool:
        return True

    def at(
            self,
            points: u.Quantity,
            convention: VelocityConvention | None = None
    ) -> 'SpreadFunction | list[SpreadFunction]':
        return varying.per_point(
            points, lambda points_: self._at(points_, convention))

    def _at(
            self,
            points: u.Quantity,
            convention: VelocityConvention | None
    ) -> list[SpreadFunction]:
        """The image of each of the points (1D), one for each blend."""
        blends = varying.blend(self._points, points, convention)
        images = {
            (i, j, t): self._image(
                (1 - t) * self._data[i] + t * self._data[j])
            for i, j, t in set(blends)}
        return [images[blend] for blend in blends]

    def velocity_range(
            self, convention: VelocityConvention | None = None
    ) -> tuple[float, float]:
        points = varying.to_velocities(self._points, convention)
        return float(points.min()), float(points.max())

    def _size_impl(self, *args: Any) -> Any:
        raise RuntimeError(
            "images of a PSF or an LSF give an image at each channel (see at)")

    def _asarray_impl(self, *args: Any) -> np.ndarray:
        raise RuntimeError(
            "images of a PSF or an LSF give an image at each channel (see at)")
