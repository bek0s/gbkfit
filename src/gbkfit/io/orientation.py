"""
The orientation of the model, and how an array of a file and its world
coordinates are put in it: the celestial axes first, longitude (e.g. RA)
then latitude, and the spectral axis last; east to the left of north;
and velocities increasing along the spectral axis (see model_header).
"""

from collections.abc import Sequence
from typing import NamedTuple

import astropy.io.fits
import astropy.wcs
import numpy as np

from gbkfit.config import warn
from gbkfit.grid import Rest
from .headers import (
    _centre_missing_crpix, _with_shape, _with_wcs, has_wcs, parse_wcs)
from .spectral import _axis_of_rest, _read_as_velocities, _stated_header


__all__ = [
    'Orientation',
    'model_axis_order',
    'model_header',
    'to_model_axes'
]


class Orientation(NamedTuple):
    """
    How an array of a file is put in the orientation of the model (see
    model_header): its axes of length 1 that are not its own are dropped
    (see _own_axes), and the others are reordered and reversed.

    Attributes
    ----------
    file_shape : tuple of int
        The shape of the array in the file (numpy order).
    shape : tuple of int
        Its shape once those axes of length 1 are dropped (numpy order).
    order : tuple of int
        Those axes (FITS order, from 0) in the order of the model (see
        _reorder_data).
    flips : tuple of int
        The axes then reversed (FITS order, from 0; see _flip_data).
    """
    file_shape: tuple[int, ...]
    shape: tuple[int, ...]
    order: tuple[int, ...]
    flips: tuple[int, ...]

    def apply(self, array: np.ndarray) -> np.ndarray:
        """
        Return an array of the file (of file_shape) in the orientation of
        the model, contiguous.

        Raises
        ------
        ValueError
            If the array is not of file_shape.
        """
        if array.shape != self.file_shape:
            raise ValueError(
                f"the array is of shape {array.shape}, not {self.file_shape}")
        array = array.reshape(self.shape)
        if self.order != tuple(range(array.ndim)):
            array = _reorder_data(array, self.order)
        for axis in self.flips:
            array = _flip_data(array, axis)
        return np.ascontiguousarray(array)

    def sky(self) -> 'Orientation':
        """
        Return how an image of the sky of the array, in the axes of its
        file, is put in the orientation of the model, as the array is:
        an image of the two axes that become the first two of the model
        (the celestial axes, which the model puts first).

        Raises
        ------
        ValueError
            If the array has fewer than two axes in the model.
        """
        if len(self.order) < 2:
            raise ValueError("the array has no sky of two axes")
        sky = sorted(self.order[:2])
        sizes = self.shape[::-1]
        shape = (sizes[sky[1]], sizes[sky[0]])
        return Orientation(
            shape, shape, tuple(sky.index(axis) for axis in self.order[:2]),
            tuple(axis for axis in self.flips if axis < 2))


def model_header(
        source: str,
        header: astropy.io.fits.Header,
        shape: tuple[int, ...],
        rest: Rest = None,
        velocity: str | None = None,
        spectral_type: str | None = None,
        medium: str | None = None,
        spectral_axis: int | None = None
) -> tuple[astropy.io.fits.Header, Orientation]:
    """
    Return the header of an array of a file in the orientation of the
    model, and how the array is put in it (see Orientation).

    An axis whose type FITS does not define (e.g. CTYPE3 = 'WAVELENGTH',
    as CALIFA writes) is read as the type given (see _with_spectral_type),
    and is refused without one: nothing else says what its values are.

    The orientation of the model has the celestial axes first, longitude
    (e.g. RA) then latitude, and the spectral axis last (see
    model_axis_order); east to the left of north, and velocities (or
    logarithmic wavelengths, see coords_from_header) increasing along the
    spectral axis (see _model_axis_flips). Axes of length 1 that are not
    the array's own are dropped (see _own_axes); world axes beyond those
    of the array (e.g. the spectral axis of a moment map written with the
    header of its cube) are taken as axes of length 1. A missing
    reference pixel of an axis without a world position is at the centre
    of its axis (see _centre_missing_crpix).
    Distortions of the world coordinates (e.g. SIP) are left out, with a
    warning: the model reads them as linear.

    A rest turns the spectral axis into velocities of it first (before
    the orientation, which then reverses velocities that decrease),
    exactly, as all are linear in the pixels: a linear wavelength axis
    (WAVE, or AWAV with a rest in air too) into the optical velocities c
    (w / rest - 1), a linear frequency axis (FREQ) into the radio
    velocities c (1 - f / rest), and optical or radio velocities (VOPT or
    VRAD) of another rest into those of this one (an axis without a rest
    takes it). The rest is in the medium of the wavelengths of the axis
    (for optical velocities, that of AIRORVAC, or the given medium; see
    _stated_medium). The header gets the rest (RESTWAV, in vacuum, or
    RESTFRQ), and for wavelengths their medium (AIRORVAC). An axis of
    logarithmic wavelengths (WAVE-LOG) gets the rest only (see
    coords_from_header), and so does a spectral axis of one pixel, which
    has no velocities to read (e.g. that of a moment map).
    The rest of data without a spectral axis in their header is left to
    their grid (see coords_from_header).

    Parameters
    ----------
    source : str
        The name of the file, for messages.
    header : astropy.io.fits.Header
        The header of the array.
    shape : tuple of int
        The shape of the array (numpy order).
    rest : str or Quantity, optional
        The rest of the spectral axis (see make_rest).
    velocity : str, optional
        The convention of the velocities of a spectral axis that the
        header leaves ambiguous: 'radio' or 'optical' (see
        _with_velocity_convention); it is not used for others.
    spectral_type : str, optional
        The type of an axis whose header gives one that FITS does not
        define (see _with_spectral_type); it is not used for others.
    medium : str, optional
        The medium of the wavelengths of an axis of optical velocities and
        of their rest, if AIRORVAC does not state it: 'vacuum' (by
        default) or 'air'.
    spectral_axis : int, optional
        The index (FITS order, from 0) that the spectral axis of the array
        (if any) has in the orientation of the model: the array has an
        axis of its own before it for each, of length 1 too (e.g. the
        region of spectra of one region; see _own_axes).

    Returns
    -------
    astropy.io.fits.Header
        The header in the orientation of the model (its other keywords as
        they are).
    Orientation
        How the array is put in the orientation of the model.

    Raises
    ------
    ValueError
        If the world coordinates are invalid, or of fewer axes than the
        array, or the header does not say whether its velocities are radio
        or optical and velocity is not given; or the rest is given for a
        spectral axis that does not take one (see above); or an axis has a
        type that FITS does not define and spectral_type is not given; or
        the medium is not valid or not the one AIRORVAC states.
    """
    header = _stated_header(source, header, shape, spectral_type, velocity)
    ndim = None if spectral_axis is None else spectral_axis + 1
    return _oriented_header(source, header, shape, rest, medium, ndim)


def _oriented_header(
        source: str,
        header: astropy.io.fits.Header,
        shape: tuple[int, ...],
        rest: Rest = None,
        medium: str | None = None,
        ndim: int | None = None
) -> tuple[astropy.io.fits.Header, Orientation]:
    """
    Return a header in the orientation of the model, with the spectral
    axis in velocities of a rest, if given, and how its array is put in it:
    model_header once the header states the type of its spectral axis and
    the convention of its velocities (see read_oriented, which looks at
    arrays whose headers need not).
    """
    shape = tuple(shape)
    if not has_wcs(header):
        kept = tuple(shape[axis] for axis in _own_axes(shape, ndim))
        return (
            _with_shape(header, kept),
            Orientation(shape, kept, tuple(range(len(kept))), ()))
    wcs, full = _file_wcs(source, header, shape)
    if wcs.has_distortion:
        warn(
            f"{source}: its world coordinates have distortions (e.g. SIP), "
            f"which the model leaves out: it reads them as linear")
    wcs, cards = _axis_of_rest(source, header, wcs, full, rest, medium)
    stokes = [
        len(full) - 1 - axis for axis, ctype in enumerate(wcs.wcs.ctype)
        if ctype.strip().upper() == 'STOKES']
    own = _own_axes(full, ndim, stokes)
    kept = tuple(full[axis] for axis in own)
    wcs = _squeezed_wcs(wcs, [len(full) - 1 - axis for axis in own])
    wcs, order, flips = _model_wcs(wcs, kept[::-1])
    sizes = [kept[len(kept) - 1 - axis] for axis in order]
    header = _with_shape(_with_wcs(header, wcs), tuple(sizes[::-1]))
    header.update(cards)
    return header, Orientation(shape, kept, order, flips)


def model_axis_order(wcs: astropy.wcs.WCS) -> tuple[int, ...]:
    """
    Return the axes of a WCS in the order of the model.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        World coordinates.

    Returns
    -------
    tuple[int, ...]
        Its axes (FITS order, from 0): the celestial axes first,
        longitude (e.g. RA) then latitude, the spectral axis last, and
        the other axes between them, in their order.
    """
    lng, lat, spec = wcs.wcs.lng, wcs.wcs.lat, wcs.wcs.spec
    first = (lng, lat) if lng >= 0 else ()
    last = (spec,) if spec >= 0 else ()
    middle = tuple(
        axis for axis in range(wcs.naxis) if axis not in first + last)
    return first + middle + last


def _model_axis_flips(wcs: astropy.wcs.WCS) -> tuple[int, ...]:
    """
    Return the axes (FITS order, from 0) of a WCS with its axes in the
    order of the model (see model_axis_order) that the model needs
    reversed: the longitude axis of a mirrored image (east to the right
    of north), and a spectral axis of velocities, or of logarithmic
    wavelengths (read as log velocities, see coords_from_header), that
    decrease along it.
    """
    lng, lat, spec = wcs.wcs.lng, wcs.wcs.lat, wcs.wcs.spec
    linear = wcs.pixel_scale_matrix
    flips = []
    if lng >= 0 and np.linalg.det(linear[np.ix_([lng, lat], [lng, lat])]) > 0:
        flips.append(lng)
    is_velocity = spec >= 0 and _read_as_velocities(wcs.wcs.ctype[spec])
    if is_velocity and linear[spec, spec] < 0:
        flips.append(spec)
    return tuple(flips)


def _reorder_data(data: np.ndarray, order: Sequence[int]) -> np.ndarray:
    """
    Return data, as a view, with their axes (FITS order, from 0) in the
    given order (see model_axis_order).
    """
    n = data.ndim
    # numpy axes are in the reverse of the FITS order
    return np.transpose(data, [n - 1 - order[n - 1 - i] for i in range(n)])


def _reorder_wcs(
        wcs: astropy.wcs.WCS, order: Sequence[int]
) -> astropy.wcs.WCS:
    """
    Return world coordinates with their axes (FITS order, from 0) in the
    given order (see model_axis_order).
    """
    return wcs.sub([axis + 1 for axis in order])


def _flip_data(data: np.ndarray, axis: int) -> np.ndarray:
    """
    Return data, as a view, reversed along an axis (FITS order, from 0).
    """
    return np.flip(data, data.ndim - 1 - axis)


def _flip_wcs(
        wcs: astropy.wcs.WCS, axis: int, size: int
) -> astropy.wcs.WCS:
    """
    Return world coordinates with a pixel axis (FITS order, from 0; of
    size pixels) reversed: each pixel keeps its world coordinates when
    its data are reversed along the axis (see _flip_data).
    """
    wcs = wcs.deepcopy()
    # Reversing a pixel axis negates its column of the linear
    # transformation (CD, or CDELT times the rows of PC). For PC, the sign
    # goes to CDELT, and the diagonal of PC stays (e.g. 1): CDELT and the
    # row of PC are negated, which leaves the rest of the row as it was,
    # and so is the column of PC, which negates the rest of the column.
    if wcs.wcs.has_cd():
        cd = wcs.wcs.cd.copy()
        cd[:, axis] *= -1
        wcs.wcs.cd = cd
    else:
        pc = wcs.wcs.get_pc().copy()
        pc[axis, :] *= -1
        pc[:, axis] *= -1
        wcs.wcs.pc = pc
        wcs.wcs.cdelt[axis] *= -1
    wcs.wcs.crpix[axis] = size + 1 - wcs.wcs.crpix[axis]
    wcs.wcs.set()
    return wcs


def to_model_axes(
        data: np.ndarray,
        wcs: astropy.wcs.WCS
) -> tuple[np.ndarray, astropy.wcs.WCS]:
    """
    Return data and their world coordinates in the orientation of the
    model.

    The axes are reordered (see model_axis_order), and those that the
    model needs reversed are reversed (see _model_axis_flips).

    Parameters
    ----------
    data : np.ndarray
        The data.
    wcs : astropy.wcs.WCS
        Their world coordinates.

    Returns
    -------
    np.ndarray
        The data.
    astropy.wcs.WCS
        Their world coordinates.
    """
    wcs, order, flips = _model_wcs(wcs, data.shape[::-1])
    return Orientation(data.shape, data.shape, order, flips).apply(data), wcs


def _model_wcs(
        wcs: astropy.wcs.WCS, sizes: Sequence[int]
) -> tuple[astropy.wcs.WCS, tuple[int, ...], tuple[int, ...]]:
    """
    Return world coordinates in the orientation of the model, the order
    of their axes in it and the axes then reversed (see Orientation), of
    an array of the given sizes of its axes (FITS order).
    """
    order = model_axis_order(wcs)
    wcs = _reorder_wcs(wcs, order)
    flips = _model_axis_flips(wcs)
    for axis in flips:
        wcs = _flip_wcs(wcs, axis, sizes[order[axis]])
    return wcs, order, flips


def _squeezed_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Return the shape of data (numpy order) without its axes of length 1,
    or as it is if all are.
    """
    return tuple(shape[axis] for axis in _own_axes(shape))


def _own_axes(
        shape: tuple[int, ...],
        ndim: int | None = None,
        others: Sequence[int] = ()
) -> list[int]:
    """
    Return the axes (numpy order) of data of a shape that are their own:
    those not of length 1, or all if all are; and, if the data have more
    (ndim, if given), as many of length 1 as they lack, the first in FITS
    order (e.g. the axis of the one region of spectra of regions), but
    for the axes given as others (e.g. one of Stokes parameters, which
    may come before).
    """
    own = [axis for axis, size in enumerate(shape) if size > 1]
    if not own:
        return list(range(len(shape)))
    # (from the first in FITS order: the last in numpy order)
    flat = [
        axis for axis in reversed(range(len(shape)))
        if axis not in own and axis not in others]
    missing = 0 if ndim is None else max(ndim - len(own), 0)
    return sorted(own + flat[:missing])


def _squeezed_wcs(
        wcs: astropy.wcs.WCS, kept: Sequence[int]
) -> astropy.wcs.WCS:
    """
    Return world coordinates of only the given axes (FITS order, from 0)
    of data whose other axes are of length 1: at the one pixel of each
    axis left out.

    Raises
    ------
    ValueError
        If an axis left out is coupled to the axes kept in a way that they
        alone cannot represent.
    """
    kept = sorted(kept)
    if len(kept) == wcs.naxis:
        return wcs
    # The world coordinates of the axes kept alone are at the reference
    # pixel of those left out; the data are at their one pixel (0): an
    # axis left out that the linear transformation couples to the axes
    # kept moves them, by as much as a shift of their reference pixel
    flat = [axis for axis in range(wcs.naxis) if axis not in kept]
    linear = wcs.pixel_scale_matrix
    moved = linear[np.ix_(kept, flat)] @ (0 - (wcs.wcs.crpix[flat] - 1))
    if np.any(moved != 0):
        try:
            shift = np.linalg.solve(linear[np.ix_(kept, kept)], moved)
        except np.linalg.LinAlgError:
            raise ValueError(
                "the world coordinates couple an axis of length 1 to the "
                "others in a way that they alone cannot represent") from None
        wcs = wcs.deepcopy()
        crpix = wcs.wcs.crpix.copy()
        crpix[kept] -= shift
        wcs.wcs.crpix = crpix
        # (the coupling, now in the reference pixel, left out: wcslib
        # takes no axes out of a matrix that couples them to others)
        matrix = (
            wcs.wcs.cd.copy() if wcs.wcs.has_cd()
            else wcs.wcs.get_pc().copy())
        matrix[np.ix_(kept, flat)] = 0
        matrix[np.ix_(flat, kept)] = 0
        if wcs.wcs.has_cd():
            wcs.wcs.cd = matrix
        else:
            wcs.wcs.pc = matrix
        wcs.wcs.set()
    return wcs.sub([axis + 1 for axis in kept])


def _file_wcs(
        source: str,
        header: astropy.io.fits.Header,
        shape: tuple[int, ...]
) -> tuple[astropy.wcs.WCS, tuple[int, ...]]:
    """
    Return the world coordinates of the header of an array of the given
    shape (numpy order), in the axes of the file, with a missing reference
    pixel of an axis without a world position at its centre (see
    _centre_missing_crpix); and the shape of the array with an axis of
    length 1 for each world axis beyond its axes.

    Raises
    ------
    ValueError
        If the world coordinates are invalid, or of fewer axes than the
        array.
    """
    naxis = parse_wcs(source, header).naxis
    if naxis < len(shape):
        raise ValueError(
            f"{source}: the header has world coordinates for {naxis} axes, "
            f"but the data have {len(shape)}")
    full = (1,) * (naxis - len(shape)) + tuple(shape)
    return parse_wcs(source, _centre_missing_crpix(header, full)), full
