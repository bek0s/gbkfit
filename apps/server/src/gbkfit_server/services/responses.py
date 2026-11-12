"""
The responses of instruments, to look at them as they are made: a PSF
drawn on pixels, and an LSF on channels (see gbkfit.instrument), of their
options as a piece of a project gives them (their files relative to it),
with what they amount to: the solid angle of the beam of a PSF, and the
full widths at half maximum of both.
"""

import dataclasses
import math
import os
from collections.abc import Iterator, Mapping
from typing import Any

import astropy.units as u
import numpy as np

import gbkfit.io
import gbkfit.units
from gbkfit.config import ConfigError, TypedParser, paths_relative_to
from gbkfit.grid.spectral import VelocityConvention, make_rest
from gbkfit.instrument.lsfs import Channel, ChannelLSF, lsf_parser
from gbkfit.instrument.psfs import psf_parser

from .errors import Invalid
from .workspace import Workspace


__all__ = [
    'LSFView',
    'PSFView',
    'Responses'
]


# The unit of an axis of velocities
_KMS = u.km / u.s

# The pixels (channels) across a response drawn at its own scale, and the
# most drawn at a scale given (e.g. that of the data)
_ACROSS = 129
_MOST = 1025

# The most pixels across a PSF drawn at its own scale, of its core and as
# much of its wings as they hold (a view, sent again as its options change)
_OWN_MOST = 257

# The fewest pixels across the core of a PSF (its region above half its
# peak) drawn at its own scale; the pixels across the window about its
# centre that finds them, how many times finer a core of one pixel is
# drawn next, and the most tries (see _own_step)
_CORE = 12
_PROBE = 65
_ZOOM = 16
_TRIES = 30


@dataclasses.dataclass
class PSFView:
    """
    A PSF drawn on pixels: its values (ny, nx; they sum to 1), the size
    of the pixels (x, y; arcsec), the solid angle of its beam (arcsec^2)
    and the full width at half maximum of the circular Gaussian of that
    beam (arcsec; none for a point), and its full widths at half maximum
    along its major and minor axes (arcsec) and the position angle of the
    major axis (degrees, from north through east; none if it is round),
    as measured (see _axes; none for a point).
    """
    values: np.ndarray
    step: tuple[float, float]
    beam_area: float | None
    fwhm: float | None
    axes: tuple[float, float, float | None] | None


@dataclasses.dataclass
class LSFView:
    """
    An LSF drawn on channels: its values (they sum to 1), the width of
    the channels, its full width at half maximum (none for a point, or
    one narrower than a channel), and the unit of both (e.g. 'km/s').
    """
    values: np.ndarray
    step: float
    fwhm: float | None
    unit: str


class Responses:
    """
    The responses of the instruments of the projects of a workspace.

    Parameters
    ----------
    workspace : Workspace
        The workspace.
    """

    def __init__(self, workspace: Workspace):
        self._workspace = workspace

    def psf(
            self,
            project: str,
            options: Mapping[str, Any],
            step: tuple[float, float] | None = None
    ) -> PSFView:
        """
        Draw a PSF of the options of a piece of a project: on pixels of a
        size (e.g. those of the data; arcsec), or of its own scale (see
        _own_step).

        Raises
        ------
        Invalid
            If the options are not those of a PSF, or one that varies
            along the spectral axis, or it needs too many pixels.
        """
        psf = self._load(psf_parser, project, options, 'PSF')
        try:
            own = _own_step(psf)
            if step is None:
                step = (own, own)
                size = _size_at_most(psf, step, _OWN_MOST)
            else:
                if min(step) <= 0:
                    raise Invalid(
                        f"the size of the pixels must be positive: {step}")
                size = psf.size(step)
                if max(size) > _MOST:
                    raise Invalid(
                        f"the PSF needs {size[0]} x {size[1]} pixels of "
                        f"{step[0]:g} x {step[1]:g} arcsec; at most {_MOST}")
            values = psf.asarray(step, size)
        except (RuntimeError, OverflowError) as e:
            # (an overflow: the extent of wings far too heavy, e.g. of a
            # Moffat of a beta very close to 1)
            raise Invalid(f"cannot draw the PSF: {e}") from e
        try:
            beam_area = psf.beam_area()
        # A point has none
        except ValueError:
            beam_area = None
        # (that of the round Gaussian beam of the solid angle)
        fwhm = None if beam_area is None \
            else float(np.sqrt(beam_area / gbkfit.io.beam_area(1, 1)))
        return PSFView(
            np.asarray(values, dtype=np.float32), tuple(step), beam_area,
            fwhm, None if beam_area is None else _axes(psf, own))

    def lsf(
            self,
            project: str,
            options: Mapping[str, Any],
            step: float | None = None,
            rest: str | None = None,
            unit: str | None = None,
            channel: float | None = None
    ) -> LSFView:
        """
        Draw an LSF of the options of a piece of a project: on channels
        of a width (e.g. those of the data), or of its own scale (about
        _ACROSS channels across), of an axis of velocities (km/s), at the
        rest of a line if given (e.g. '6562.8 Angstrom'; that of the data),
        which its widths in a unit of wavelength or frequency need; or of
        an axis of another unit, if given (e.g. 'Angstrom', or 'pix': of
        channels of the data, each of width 1), which draws it as its
        widths give it. Its widths in pixels and its Hanning smoothing are
        of the channels of the data: of a width, if given (in the unit of
        the axis; e.g. those of a dataset, it drawn on finer ones), or else
        those it is drawn on.

        Raises
        ------
        Invalid
            If the options are not those of an LSF, or one that varies
            along the spectral axis, or it needs too many channels or a
            rest that is not given, or the rest or the unit is not one, or
            both are given, or a width of the channels is not positive or
            is given of an axis of pixels.
        """
        lsf = self._load(lsf_parser, project, options, 'LSF')
        for width in (step, channel):
            if width is not None and width <= 0:
                raise Invalid(
                    f"the width of the channels must be positive: {width}")
        try:
            axis = _KMS if unit is None else gbkfit.units.parse_unit(unit)
            convention = VelocityConvention(make_rest(rest))
        except ValueError as e:
            raise Invalid(str(e)) from e
        if axis != _KMS and rest is not None:
            raise Invalid(
                f"a rest is of an axis of velocities, not of '{unit}'")
        if axis == u.pix and channel is not None:
            raise Invalid("the channels of an axis of pixels are of width 1")
        unit = unit or 'km/s'
        # (drawn at 0 km/s, the velocity of the rest, on channels of the
        # data that it is drawn on: its widths in km/s of the light span
        # as many km/s there, and those in a unit of wavelength or
        # frequency span km/s by the rest; or on an axis of their unit, as
        # much as they are, or of channels of the data, each of width 1)
        data = 1.0 if axis == u.pix else channel
        def on(width: float) -> ChannelLSF:
            return lsf.on(Channel(
                0.0, width if data is None else data, convention, axis))
        try:
            # (of its own scale, the first width below about 1/_ACROSS of
            # it that it can be drawn on: round, or a whole part of the
            # channels of the data, if given (e.g. a Hanning smoothing, only
            # on channels of which its width is a multiple)
            if step is not None:
                steps = [step]
            elif channel is not None:
                parts = math.ceil(channel / (on(1.0).size(1.0) / _ACROSS))
                steps = [channel / k for k in range(parts, parts + 4)]
            else:
                steps = _round_below(on(1.0).size(1.0) / _ACROSS)
        except (RuntimeError, ValueError) as e:
            raise Invalid(f"cannot draw the LSF: {e}") from e
        for i, width in enumerate(steps):
            try:
                size = on(width).size(width)
                if size > _MOST:
                    raise Invalid(
                        f"the LSF needs {size} channels of {width:g} {unit}; "
                        f"at most {_MOST}")
                values = np.asarray(
                    on(width).asarray(width, size), dtype=np.float32)
            except RuntimeError as e:
                if i == len(steps) - 1:
                    raise Invalid(f"cannot draw the LSF: {e}") from e
                continue
            return LSFView(values, width, _fwhm(values, width), unit)
        raise AssertionError("unreachable: there is a width")

    def _load(
            self,
            parser: TypedParser,
            project: str,
            options: Mapping[str, Any],
            what: str
    ) -> Any:
        """
        Load a response of the options of a piece of a project, its files
        relative to the project (and in the workspace).

        Raises
        ------
        Invalid
            If the options are invalid, a file is out of the workspace, or
            it varies along the spectral axis.
        """
        folder = self._workspace.path(project)
        for file in _files(options):
            self._workspace.path(os.path.join(project, file))
        try:
            with paths_relative_to(folder):
                response = parser.load(dict(options))
        except (ConfigError, OSError, ValueError, TypeError) as e:
            raise Invalid(str(e)) from e
        if response.varies():
            raise Invalid(
                f"this {what} varies along the spectral axis; the Studio "
                f"cannot draw one yet")
        return response


def _files(options: Any) -> Iterator[str]:
    """Yield the files that options name (their options 'file')."""
    if isinstance(options, Mapping):
        for key, value in options.items():
            if key == 'file':
                if isinstance(value, str):
                    yield value
                elif isinstance(value, Mapping) \
                        and isinstance(value.get('file'), str):
                    yield value['file']
                elif isinstance(value, list) and value \
                        and isinstance(value[0], str):
                    yield value[0]
            else:
                yield from _files(value)
    elif isinstance(options, list):
        for item in options:
            yield from _files(item)


# How many times finer than its own scale a PSF is drawn to measure its
# widths (see _axes)
_FINER = 3


def _size_at_most(
        psf: Any, step: tuple[float, float], most: int = _MOST
) -> tuple[int, int]:
    """
    Return the size of the array of a PSF on pixels of a size, cut to the
    most pixels (odd, as its sizes are) along each axis: the heavy wings
    of some (e.g. a Moffat of a small beta) reach far beyond its core.
    """
    nx, ny = psf.size(step)
    return min(nx, most), min(ny, most)


def _core(values: np.ndarray) -> int:
    """
    Return the pixels across the core of a PSF drawn (its region above
    half its peak), along x or y, the wider.
    """
    y, x = np.nonzero(values >= values.max() / 2)
    return int(max(np.ptp(x), np.ptp(y))) + 1


def _own_step(psf: Any) -> float:
    """
    Return the size of the pixels of a PSF at its own scale (arcsec):
    those of the whole of it (its extent) across _ACROSS pixels, or finer,
    so that its core is at least _CORE pixels across (e.g. a Moffat of a
    small beta, whose wings reach far beyond its core: the whole of it on
    _ACROSS pixels would put its core on one, or on a tiny part of one),
    drawn then out to _OWN_MOST pixels. The core is measured on a window
    about its centre, _ZOOM times finer while it is on one pixel.
    """
    step = max(psf.size((1.0, 1.0))) / _ACROSS
    for _ in range(_TRIES):
        core = _core(psf.asarray((step, step), (_PROBE, _PROBE)))
        if core >= _CORE:
            break
        step *= 1 / _ZOOM if core == 1 else core / _CORE
    return step


def _axes(psf: Any, own: float) -> tuple[float, float, float | None]:
    """
    Return the full widths at half maximum of a PSF along its major and
    minor axes (arcsec), and the position angle of its major axis
    (degrees, from north through east, in [0, 180); none if it is round
    to 1%): those of the ellipse of the second moments of the region
    above half its peak, drawn on pixels _FINER times finer than those
    of its own scale (own; see _own_step) (for a filled ellipse, the
    variance along an axis is a quarter of the square of its semi-axis).
    """
    step = own / _FINER
    values = psf.asarray((step, step), _size_at_most(psf, (step, step)))
    y, x = np.mgrid[:values.shape[0], :values.shape[1]]
    half = values >= values.max() / 2
    xs, ys = x[half] * step, y[half] * step
    variances, vectors = np.linalg.eigh(np.cov(np.vstack([xs, ys]), bias=True))
    major, minor = 4 * np.sqrt(np.maximum(variances[::-1], 0))
    # (the angle of the major axis on the array, from x, is that of the
    # position angle from north plus 90)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    posa = None if major - minor < 0.01 * major else float((angle - 90) % 180)
    return float(major), float(minor), posa


def _round_below(value: float) -> list[float]:
    """
    Return the round numbers (1, 2, 2.5 and 5 times a power of 10) at or
    below a value, from it down, over three powers of 10.
    """
    power = 10.0 ** np.floor(np.log10(value))
    rounds = [
        m * power * 10.0 ** k for k in (0, -1, -2, -3)
        for m in (5, 2.5, 2, 1)]
    return [float(f'{r:.6g}') for r in rounds if r <= value]


def _fwhm(values: np.ndarray, step: float) -> float | None:
    """
    Return the full width at half maximum of a profile of one peak on
    channels of a width, between its crossings of half its peak (linear
    between channels); none if it does not cross it on both sides.
    """
    peak = int(np.argmax(values))
    half = values[peak] / 2
    below = np.flatnonzero(values[:peak] < half)
    above = np.flatnonzero(values[peak:] < half)
    if below.size == 0 or above.size == 0:
        return None
    i, j = below[-1], peak + above[0]
    left = i + (half - values[i]) / (values[i + 1] - values[i])
    right = j - 1 + (values[j - 1] - half) / (values[j - 1] - values[j])
    return float((right - left) * step)
