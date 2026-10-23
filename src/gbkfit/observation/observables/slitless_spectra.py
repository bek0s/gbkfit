from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import astropy.constants
import astropy.units
import numpy as np
import numpy.typing as npt
import scipy.sparse

from gbkfit.dataset import Data, Dataset, DatasetSlitlessSpectra
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument
from gbkfit.model import LineEmission
from gbkfit.model.components._units import DataUnit
from gbkfit.utils import gridutils, iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _dcube, _detail, _moments
from ._region_sums import RegionSumsPlan
from .apertures import Aperture, aperture_parser
from .base import HostModelData, ModelData, Observable
from .traces import Trace, trace_parser

if TYPE_CHECKING:
    from .._scene import Scene
    from ..foreground import Foreground
    from ..noise import Noise
    from ..weights import Weights


__all__ = [
    'Exposure',
    'SlitlessSpectra',
    'exposure_parser',
    'pixel_footprint',
    'pixel_footprints'
]


_C = astropy.constants.c.to_value('km/s')

# The segments of a chunk of pixel_footprints (a bound of its memory)
_FOOTPRINT_CHUNK = 20000


def _overlap(s: np.ndarray, width: np.ndarray) -> np.ndarray:
    """
    Return the overlap of an interval of a width and a unit interval whose
    centres are s apart: linear in s but at s = +-(width - 1) / 2 and
    +-(width + 1) / 2.
    """
    return np.maximum(
        0.0, np.minimum(width / 2, s + 0.5) - np.maximum(-width / 2, s - 0.5))


def pixel_footprint(
        start: tuple[float, float],
        end: tuple[float, float],
        size: tuple[float, float] = (1, 1)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return where the light of a box (a pixel, or a part of it) lands when
    it is moved uniformly along a straight segment: the light of the box
    centred at 0, spread evenly over the copies of the box centred on the
    segment from start to end (in pixels), in each pixel (p, q) of the
    grid.

    The fraction in a pixel is the mean over the segment of the overlap of
    the moved box with it, over the area of the box: the integral over t
    in [0, 1] of o_x(p - x(t)) o_y(q - y(t)) / (w_x w_y), where (x, y)
    moves from start to end and o_x(s) is the overlap of an interval of
    the width w_x of the box with a unit interval s apart (likewise o_y).
    Between the values of t where either factor has a kink, the product is
    a quadratic, so Simpson's rule on each such interval is exact. The
    fractions add up to 1.

    Parameters
    ----------
    start, end : tuple of float
        The ends of the segment (x, y).
    size : tuple of float, optional
        The widths of the box (pixels); a pixel by default.

    Returns
    -------
    p, q : np.ndarray
        The pixels that get light (integers).
    weights : np.ndarray
        The fraction of the light in each.
    """
    _, p, q, weights = pixel_footprints([start], [end], [size])
    return p, q, weights


def pixel_footprints(
        starts: npt.ArrayLike,
        ends: npt.ArrayLike,
        sizes: npt.ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the footprints of boxes moved along segments (see
    pixel_footprint), from starts to ends (of shape (n, 2), pixels), of
    the given sizes (of shape (n, 2); pixels by default): the segment of
    each fraction, its pixel (p, q) and the fraction.
    """
    starts = np.asarray(starts, dtype=float).reshape(-1, 2)
    ends = np.asarray(ends, dtype=float).reshape(-1, 2)
    sizes = np.ones_like(starts) if sizes is None \
        else np.asarray(sizes, dtype=float).reshape(-1, 2)
    out: list[list[np.ndarray]] = [[], [], [], []]
    for first in range(0, len(starts), _FOOTPRINT_CHUNK):
        chunk = slice(first, first + _FOOTPRINT_CHUNK)
        for k, part in enumerate(_footprints(
                starts[chunk], ends[chunk], sizes[chunk], first)):
            out[k].append(part)
    index, p, q, weights = (np.concatenate(parts) for parts in out)
    return index, p.astype(int), q.astype(int), weights


def _footprints(
        starts: np.ndarray, ends: np.ndarray, sizes: np.ndarray, first: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the footprints of a chunk of boxes and segments (see
    pixel_footprints), the first of which is of the index first.
    """
    ax, ay = starts[:, 0:1], starts[:, 1:2]
    lx, ly = ends[:, 0:1] - ax, ends[:, 1:2] - ay
    wx, wy = sizes[:, 0:1], sizes[:, 1:2]
    # (the pixels within (w + 1) / 2 of the segment)
    reach = (sizes + 1) / 2
    corner = np.floor(np.minimum(starts, ends) - reach)
    span = (np.ceil(np.maximum(starts, ends) + reach) + 1 - corner).max(
        axis=0)
    ox, oy = (o.ravel() for o in np.meshgrid(
        np.arange(span[0]), np.arange(span[1])))
    p, q = corner[:, 0:1] + ox, corner[:, 1:2] + oy
    # (the kinks: p - x(t) or q - y(t) at +-(w - 1) / 2 or +-(w + 1) / 2)
    breaks = [np.zeros_like(p), np.ones_like(p)]
    for pixel, origin, length, width in (
            (p, ax, lx, wx), (q, ay, ly, wy)):
        for kink in (-(width + 1) / 2, -(width - 1) / 2, (width - 1) / 2,
                     (width + 1) / 2):
            breaks.append(np.divide(
                pixel - kink - origin, length, out=np.zeros_like(pixel),
                where=length != 0))
    t = np.sort(np.clip(np.stack(breaks, axis=-1), 0, 1), axis=-1)

    def overlap(t_: np.ndarray) -> np.ndarray:
        return (_overlap(p[..., None] - ax[..., None] - t_ * lx[..., None],
                         wx[..., None])
                * _overlap(q[..., None] - ay[..., None] - t_ * ly[..., None],
                           wy[..., None]))
    # (Simpson's rule on each interval, of the values at the breaks shared
    # by the intervals on either side)
    at_breaks = overlap(t)
    weights = np.sum(
        (t[..., 1:] - t[..., :-1]) / 6
        * (at_breaks[..., :-1] + 4 * overlap((t[..., :-1] + t[..., 1:]) / 2)
           + at_breaks[..., 1:]), axis=-1) / (wx * wy)
    keep = weights > 0
    index = np.broadcast_to(
        first + np.arange(len(starts))[:, None], p.shape)
    return index[keep], p[keep], q[keep], weights[keep]


def _unique_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the distinct rows of an array and the index of the distinct row
    of each row (as np.unique with axis=0, by a faster lexical sort).
    """
    order = np.lexsort(rows.T[::-1])
    ordered = rows[order]
    new = np.ones(len(rows), dtype=bool)
    new[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    inverse = np.empty(len(rows), dtype=int)
    inverse[order] = np.cumsum(new) - 1
    return ordered[new], inverse


class Exposure(parseutils.Serializable):
    """
    An exposure combined into the image of slitless spectra (see
    SlitlessSpectra): of the sky at an offset from the instrument (e.g.
    after a nod of the telescope), with its image moved by an offset (e.g.
    back by the nod, when the reduction aligns the exposures on the sky)
    and weighted. Spectra whose background is subtracted with the
    exposures of the other nods (e.g. A - (B + C) / 2) are the exposures
    of the nods of weights 1, -1/2 and -1/2.

    Parameters
    ----------
    sky_offset : Sequence of float, optional
        The offset of the sky (x, y; arcsec along the axes of the image):
        the light of each place of the sky passes the aperture and lands
        on the detector as that of the place plus the offset.
    image_offset : Sequence of float, optional
        The offset of its image (x, y; arcsec).
    weight : float, optional
        Its weight.

    Raises
    ------
    ConfigError
        If an offset is not two numbers, or the weight is not finite.
    """

    def dump(self) -> dict[str, Any]:
        return dict(
            sky_offset=list(self._sky_offset),
            image_offset=list(self._image_offset), weight=self._weight)

    def __init__(
            self,
            sky_offset: Sequence[float] = (0, 0),
            image_offset: Sequence[float] = (0, 0),
            weight: float = 1
    ):
        for name, offset in (
                ('sky_offset', sky_offset), ('image_offset', image_offset)):
            if len(offset) != 2:
                raise ConfigError(
                    f"{name} must be two offsets (x, y); it is {offset}")
        if not np.isfinite(weight):
            raise ConfigError(
                f"the weight of an exposure must be finite; it is {weight}")
        self._sky_offset = tuple(float(value) for value in sky_offset)
        self._image_offset = tuple(float(value) for value in image_offset)
        self._weight = float(weight)

    def sky_offset(self) -> tuple[float, float]:
        """Return the offset of the sky (arcsec)."""
        return self._sky_offset

    def image_offset(self) -> tuple[float, float]:
        """Return the offset of its image (arcsec)."""
        return self._image_offset

    def weight(self) -> float:
        """Return its weight."""
        return self._weight


exposure_parser = parseutils.BasicParser(Exposure)


class SlitlessSpectra(Observable):
    """
    Slitless spectra: the image of a grism (or prism) exposure, where the
    light of each wavelength of an emission line lands on the detector
    away from its place on the sky by the trace of the disperser (e.g.
    JWST NIRCam or NIRISS wide-field slitless spectroscopy).

    The model is a spectral cube of the sky around the image, seen
    through the instrument (its PSF, which can vary with wavelength), of
    spec_size channels of spec_step centred on spec_rval (km/s, whose
    wavelengths are spec_rest (1 + v / c)). Each of its pixels and
    channels lands on the detector moved by the trace (dx, dy) of the
    wavelengths of the channel: its light is spread evenly from the
    position of the start of the channel to that of its end (see
    pixel_footprint). The image is the light (integrated over velocity,
    as moment 0) that lands on each pixel.

    The grid of the image is that of the detector, with the world
    coordinates of the undispersed light (see DatasetSlitlessSpectra);
    the cube of the model covers the sky whose light of the channels can
    land on the image (the reach of the trace at the corners of the image,
    then at those of that sky: the trace changes slowly across the
    detector). The trace (see Trace) gives the offsets of each wavelength
    from the place of its undispersed light: the same at every place (see
    TracePolynomial), or varying across the detector (see
    TraceGrismConf). The spectral axis covers the lines; the continuum is
    not modelled.

    An aperture in front of the disperser (e.g. the shutters of JWST
    NIRSpec; see Aperture) lets only the light of its open parts through:
    the light of each pixel of the sky, seen through the PSF, is clipped
    to them, and the open parts are dispersed by the trace where they are.
    The light from across a shutter thus lands on the detector moved along
    the dispersion, as that of other wavelengths. The spectrograph after
    the aperture then blurs the spectra by its own LSF (the instrument's,
    with an aperture), along the trace.

    The light within each pixel of the sky is spread evenly over it; the
    sky can be sampled by sky_sampling pixels for each pixel of the image
    along each axis, to place the light within a pixel (e.g. across
    shutters of a few pixels). The image can combine several exposures,
    of the sky at offsets (e.g. nods), with weights (e.g. to subtract the
    background as the reduction did; see Exposure).

    Its configuration has the options below, but those of the grid when
    it has data, which give them (see from_data).

    Parameters
    ----------
    size : Sequence of int
        The number of pixels of the image (nx, ny).
    spec_step : float
        The width of the channels of the model (km/s); a channel narrower
        than a pixel of the detector along the trace is needed to resolve
        the lines.
    spec_rest : str or Quantity
        The rest wavelength of the velocities of the spectral axis (see
        gridutils.make_rest).
    trace : Trace
        The trace of the disperser.
    step, rpix, rval : Sequence of float, optional
        The world coordinates of the image (see gridutils.Coords); rpix
        is the centre by default.
    rota : float, optional
        The rotation of the image on the sky.
    spec_size : int, optional
        The number of channels of the model; by default, they span 1000
        km/s.
    spec_rval : float, optional
        The centre of the spectral axis (km/s).
    spec_medium : str, optional
        The medium of the wavelengths: 'vacuum' (the default) or 'air'
        (see gridutils.make_medium).
    spec_convention : str, optional
        The convention of the velocities: 'optical' (the default) or 'log'
        (see gridutils.make_convention).
    aperture : Aperture, optional
        The aperture in front of the disperser, if any.
    exposures : Exposure or Sequence of Exposure, optional
        The exposures combined into the image; one, of the sky as it is,
        by default.
    sky_sampling : int, optional
        The pixels of the sky for each pixel of the image, along each axis.

    Raises
    ------
    ConfigError
        If spec_step is not positive, spec_size is less than 1, spec_rest
        is not a wavelength, there are no exposures, or sky_sampling is
        not an integer of at least 1.
    """

    dataset_class = DatasetSlitlessSpectra

    emission = LineEmission

    # The image has no spectral axis: wavelength and position share the
    # axes of the detector
    spectral_axis = None

    @staticmethod
    def type() -> str:
        return 'slitless_spectra'

    @classmethod
    def load(
            cls, info: dict[str, Any],
            dataset: DatasetSlitlessSpectra | None = None
    ) -> 'SlitlessSpectra':
        info = dict(info)
        parseutils.load_option_and_update_info(
            trace_parser, info, 'trace', required=True)
        parseutils.load_option_and_update_info(
            aperture_parser, info, 'aperture')
        parseutils.load_option_and_update_info(
            exposure_parser, info, 'exposures')
        return _detail.load_observable(cls, info, dataset, 2)

    @classmethod
    def from_data(
            cls,
            dataset: DatasetSlitlessSpectra,
            spec_step: float,
            spec_rest: str | astropy.units.Quantity,
            trace: Trace,
            spec_size: int | None = None,
            spec_rval: float = 0,
            spec_medium: str | None = None,
            spec_convention: str | None = None,
            aperture: Aperture | None = None,
            exposures: Exposure | Sequence[Exposure] | None = None,
            sky_sampling: int = 1
    ) -> 'SlitlessSpectra':
        """
        Make the observable of slitless spectra, on their grid.

        Parameters
        ----------
        dataset : DatasetSlitlessSpectra
            The image.
        spec_step, spec_rest, trace :
            As those of SlitlessSpectra.
        spec_size, spec_rval, spec_medium, spec_convention, aperture,
        exposures, sky_sampling : optional
            As those of SlitlessSpectra.

        Returns
        -------
        SlitlessSpectra
            The observable.

        Raises
        ------
        ConfigError
            If the dataset is not of slitless spectra, or the options are
            not valid.
        """
        cls._require_dataset_class(dataset)
        return cls(
            **_detail.grid_options(dataset.grid()), spec_step=spec_step,
            spec_rest=spec_rest, trace=trace, spec_size=spec_size,
            spec_rval=spec_rval, spec_medium=spec_medium,
            spec_convention=spec_convention, aperture=aperture,
            exposures=exposures, sky_sampling=sky_sampling)

    def dump(
            self,
            data: DatasetSlitlessSpectra | None = None,
            prefix: str = '',
            dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _detail.without_options_from_data(self, dict(
            type=self.type(),
            size=self.size(),
            step=self.step(),
            rpix=self.rpix(),
            rval=self.rval(),
            rota=self.rota(),
            spec_size=self._spec_size,
            spec_step=self._spec_step,
            spec_rval=self._spec_rval,
            spec_rest=_detail.dump_rest(self._spec_rest),
            spec_medium=self._spec_medium,
            spec_convention=self._spec_convention,
            trace=trace_parser.dump(self._trace),
            aperture=aperture_parser.dump(self._aperture),
            exposures=exposure_parser.dump(list(self._exposures)),
            sky_sampling=self._sky_sampling), data)

    def __init__(
            self,
            size: Sequence[int],
            spec_step: float,
            spec_rest: str | astropy.units.Quantity,
            trace: Trace,
            step: Sequence[float] = (1, 1),
            rpix: Sequence[float] | None = None,
            rval: Sequence[float] = (0, 0),
            rota: float = 0,
            spec_size: int | None = None,
            spec_rval: float = 0,
            spec_medium: str | None = None,
            spec_convention: str | None = None,
            aperture: Aperture | None = None,
            exposures: Exposure | Sequence[Exposure] | None = None,
            sky_sampling: int = 1
    ):
        super().__init__(size, step, rpix, rval, rota)
        _detail.check_positive('spec_step', spec_step)
        if spec_size is None:
            spec_size = _moments.default_spec_size(spec_step)
        if spec_size < 1:
            raise ConfigError(f"spec_size must be at least 1; it is {spec_size}")
        rest = gridutils.make_rest(spec_rest)
        if rest is None or rest.unit != astropy.units.m:
            raise ConfigError(
                f"the spectral axis of slitless spectra needs a rest "
                f"wavelength (spec_rest); it is {spec_rest!r}")
        self._spec_size = spec_size
        self._spec_step = spec_step
        self._spec_rval = spec_rval
        self._spec_rest = rest
        self._spec_medium = gridutils.make_medium(spec_medium, rest)
        self._spec_convention = gridutils.make_convention(
            spec_convention, rest)
        exposures = (Exposure(),) if exposures is None \
            else iterutils.tuplify(exposures)
        if not exposures:
            raise ConfigError("slitless spectra need at least one exposure")
        if isinstance(sky_sampling, bool) \
                or not isinstance(sky_sampling, int) or sky_sampling < 1:
            raise ConfigError(
                f"sky_sampling must be an integer of at least 1; it is "
                f"{sky_sampling!r}")
        self._trace = trace
        self._aperture = aperture
        self._exposures = exposures
        self._sky_sampling = sky_sampling

    def spec_size(self) -> int:
        """Return the number of channels of the model."""
        return self._spec_size

    def spec_step(self) -> float:
        """Return the width of the channels of the model (km/s)."""
        return self._spec_step

    def spec_rval(self) -> float:
        """Return the centre of the spectral axis (km/s)."""
        return self._spec_rval

    def spec_rest(self) -> astropy.units.Quantity:
        """Return the rest wavelength of the spectral axis."""
        return self._spec_rest

    def spec_medium(self) -> str:
        """Return the medium of the wavelengths."""
        return self._spec_medium

    def spec_convention(self) -> str:
        """Return the convention of the velocities of the rest."""
        return self._spec_convention

    def trace(self) -> Trace:
        """Return the trace of the disperser."""
        return self._trace

    def aperture(self) -> Aperture | None:
        """Return the aperture in front of the disperser, if any."""
        return self._aperture

    def exposures(self) -> tuple[Exposure, ...]:
        """Return the exposures combined into the image."""
        return self._exposures

    def sky_sampling(self) -> int:
        """Return the pixels of the sky for each pixel of the image."""
        return self._sky_sampling

    def keys(self) -> tuple[str, ...]:
        return ('spectra',)

    def spectra_item(self) -> str | None:
        return 'spectra'

    def spectra_axis(self) -> int:
        # (each line of the image along the dispersion, that over which the
        # trace moves the most across the channels at the centre of the
        # image: x, the last axis of the image, or y)
        edges = self._channel_edges()[[0, -1], [0, 1]]
        centre = (np.array(self.size()) - 1) / 2
        dx, dy = self._trace.offsets(edges, *centre)
        return 1 if abs(dx[1] - dx[0]) >= abs(dy[1] - dy[0]) else 0

    def dataset(self, items: Mapping[str, Data]) -> DatasetSlitlessSpectra:
        return DatasetSlitlessSpectra(
            items['spectra'], **_detail.dataset_grid_options(self.grid()))

    def data_unit(self, dataset: Dataset | None) -> DataUnit | None:
        # (the light of a pixel is integrated over the lines, as moment 0)
        return _detail.data_unit(
            dataset, 'spectra', abs(self.step()[0] * self.step()[1]), True)

    def model_can_lack_values(self) -> bool:
        return False

    def check_instrument(self, instrument: Instrument) -> None:
        if instrument.lsf() is not None and self._aperture is None:
            raise ConfigError(
                "slitless spectra without an aperture have no LSF: the PSF "
                "along the trace sets their resolution; remove the LSF of "
                "the instrument, or give the aperture")

    def plan(
            self,
            driver: Driver,
            scene: 'Scene',
            foreground: 'Foreground',
            instrument: Instrument,
            scale: Sequence[int],
            dtype: np.dtype,
            fft_dtype: np.dtype,
            weights: 'Weights | None',
            calibration: str | None
    ) -> 'SlitlessSpectraPlan':
        if weights is not None:
            raise ConfigError("slitless spectra do not support weights yet")
        edges = self._channel_edges()
        offset, size = self._sky(edges)
        rpix, step, k = self.rpix(), self.step(), self._sky_sampling
        # (the centre of sub-pixel i of the sky is at offset - 1/2 + (i +
        # 1/2) / k pixels of the image)
        dcube = _dcube.DCube(
            size=(size[0] * k, size[1] * k, self._spec_size),
            step=(step[0] / k, step[1] / k, self._spec_step),
            rpix=((rpix[0] - offset[0] + 0.5) * k - 0.5,
                  (rpix[1] - offset[1] + 0.5) * k - 0.5,
                  self._spec_size / 2 - 0.5),
            rval=self.rval() + (self._spec_rval,),
            rota=self.rota(),
            rest=self._spec_rest,
            medium=self._spec_medium,
            convention=self._spec_convention,
            scale=tuple(scale) + (1,),
            primary_beam=instrument.primary_beam(),
            psf=instrument.psf(),
            lsf=instrument.lsf(),
            smooth_weights=False,
            mask_cutoff=None,
            mask_apply=False,
            dtype=dtype,
            fft_dtype=fft_dtype)
        dispersion = self._dispersion(edges, size, offset)
        return SlitlessSpectraPlan(
            dcube, dispersion, self.grid(), driver, scene, foreground, dtype,
            calibration)

    def _channel_edges(self) -> np.ndarray:
        """
        Return the wavelengths (m) of the edges of each channel of the
        model (one row each) at their velocities: spec_rest (1 + v / c) of
        optical velocities, spec_rest exp(v / c) of log ones.
        """
        step, n = self._spec_step, self._spec_size
        centres = self._spec_rval + step * (np.arange(n) - (n - 1) / 2)
        velocities = centres[:, None] + np.array([-0.5, 0.5]) * step
        if self._spec_convention == 'log':
            return self._spec_rest.value * np.exp(velocities / _C)
        return self._spec_rest.value * (1 + velocities / _C)

    def _pixels(self, offset: Sequence[float]) -> np.ndarray:
        """Return an offset (arcsec along the axes of the image) in pixels."""
        return np.asarray(offset, dtype=float) / np.asarray(self.step())

    def _sky(
            self, edges: np.ndarray
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Return the sky of the cube, the pixels whose light of some channel
        (of the edges of its wavelengths) can land on the image in some
        exposure, through the aperture if any: its offset from the image
        (pixels) and its size, and a pixel more each way (of the
        footprints of the pixels).
        """
        los, his = [], []
        for exposure in self._exposures:
            lo, hi = self._reach(edges, self._pixels(exposure.image_offset()))
            if self._aperture is not None:
                # (the pixels that overlap its rectangles)
                rectangles = self._aperture.rectangles()
                x = self.rpix()[0] + rectangles[:, :2] / self.step()[0]
                y = self.rpix()[1] + rectangles[:, 2:] / self.step()[1]
                lo = np.maximum(lo, [x.min() - 0.5, y.min() - 0.5])
                hi = np.minimum(hi, [x.max() + 0.5, y.max() + 0.5])
            if np.all(lo <= hi):
                sky_offset = self._pixels(exposure.sky_offset())
                los.append(lo - sky_offset)
                his.append(hi - sky_offset)
        if not los:
            raise ConfigError(
                "no light of the sky can land on the image of the slitless "
                "spectra; check the aperture and the exposures")
        lo = np.floor(np.min(los, axis=0)) - 1
        hi = np.ceil(np.max(his, axis=0)) + 1
        offset = (int(lo[0]), int(lo[1]))
        return offset, (int(hi[0] - lo[0]) + 1, int(hi[1] - lo[1]) + 1)

    def _reach(
            self, edges: np.ndarray, image_offset: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the box (its lower and upper corners, pixels) of the places
        on the detector whose light of some channel can land on the image
        moved by image_offset (pixels): of the reach of the trace at the
        corners and the edges of the image, then of that box (the trace
        changes slowly across the detector).
        """
        width, height = self.size()
        lo = -image_offset
        hi = np.array([width - 1, height - 1]) - image_offset
        for _ in range(2):
            x, y = (a.ravel() for a in np.meshgrid(
                np.linspace(lo[0], hi[0], 5), np.linspace(lo[1], hi[1], 5)))
            dx, dy = self._trace.offsets(edges.ravel()[:, None], x, y)
            lo = np.array([-dx.max(), -dy.max()]) - image_offset
            hi = np.array([width - 1 - dx.min(), height - 1 - dy.min()]) \
                - image_offset
        return lo, hi

    def _dispersion(
            self,
            edges: np.ndarray,
            size: tuple[int, int],
            offset: tuple[int, int]
    ) -> scipy.sparse.csr_array:
        """
        Return the matrix that disperses the cube of the model (flattened,
        of shape (nz, ny, nx), of sky_sampling pixels for each pixel of the
        sky of the given size, offset from the image) into the image
        (flattened). For each exposure, the footprint (see
        pixel_footprints) of each pixel of the cube (of each of its parts
        open in the aperture, if any) at its place in the exposure, moved
        by the trace there from the start of each channel (of the edges of
        its wavelengths) to its end, and by the offset of the image of the
        exposure; times the weight of the exposure, the area of the part
        (pixels of the image: each pixel of the cube holds the light of a
        pixel of the image over its area) and the width of the channel
        (km/s), so that the image is the light integrated over velocity.
        """
        k = self._sky_sampling
        nx, ny = size[0] * k, size[1] * k
        # (the pixel of the image of each pixel of the cube, and the centre
        # of the latter from that of the former, both in pixels of the
        # image: the footprints of the same centres and segments are the
        # same in every pixel)
        j, i = (a.ravel() for a in np.mgrid[:ny, :nx])
        base = np.column_stack([offset[0] + i // k, offset[1] + j // k])
        centres = np.column_stack(
            [(i % k + 0.5) / k - 0.5, (j % k + 0.5) / k - 0.5])
        sizes = np.full(centres.shape, 1 / k)
        matrix = scipy.sparse.csr_array(
            (self.size()[0] * self.size()[1], len(edges) * nx * ny))
        for exposure in self._exposures:
            index, parts, part_sizes = self._open_parts(
                base, centres + self._pixels(exposure.sky_offset()), sizes)
            matrix = matrix + self._footprints_matrix(
                edges, nx * ny, base[index], index, parts, part_sizes,
                self._pixels(exposure.image_offset()), exposure.weight())
        return matrix.tocsr()

    def _open_parts(
            self, base: np.ndarray, centres: np.ndarray, sizes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the parts of boxes (of the pixels base, the centres from
        them and the sizes, in pixels of the image) open in the aperture:
        the box of each part, its centre (from its pixel) and its sizes;
        the whole boxes without an aperture.
        """
        if self._aperture is None:
            return np.arange(len(base)), centres, sizes
        rpix, step = np.asarray(self.rpix()), np.asarray(self.step())
        index, parts, part_sizes = self._aperture.clip(
            (base + centres - rpix) * step, sizes * step, step)
        return index, parts / step + rpix - base[index], part_sizes / step

    def _footprints_matrix(
            self,
            edges: np.ndarray,
            npixels: int,
            base: np.ndarray,
            index: np.ndarray,
            centres: np.ndarray,
            sizes: np.ndarray,
            image_offset: np.ndarray,
            weight: float
    ) -> scipy.sparse.coo_array:
        """
        Return the matrix of the footprints of boxes of the cube (see
        _dispersion): of the pixels base (of the image), the centres from
        them and the sizes (pixels of the image), of the pixels index of
        the cube (of npixels per channel), with the image moved by
        image_offset (pixels) and weighted.
        """
        width, height = self.size()
        nboxes = len(index)
        # (the segment of each box and channel from its pixel, of the trace
        # at its place, and its sizes)
        dx, dy = self._trace.offsets(edges[:, :, None], *(base + centres).T)
        x0 = centres[:, 0] + image_offset[0]
        y0 = centres[:, 1] + image_offset[1]
        rows = np.concatenate([
            np.stack([x0 + dx[:, 0], y0 + dy[:, 0], x0 + dx[:, 1],
                      y0 + dy[:, 1]], axis=-1).reshape(-1, 4),
            np.tile(sizes, (len(edges), 1))], axis=1)
        # (the footprints of the distinct rows, moved to each pixel: one
        # per channel and centre of a trace the same at every place)
        distinct, inverse = _unique_rows(rows)
        segment, p, q, weights = pixel_footprints(
            distinct[:, 0:2], distinct[:, 2:4], distinct[:, 4:6])
        counts = np.bincount(segment, minlength=len(distinct))
        first = np.cumsum(counts) - counts
        counts, first = counts[inverse], first[inverse]
        row = np.repeat(np.arange(len(inverse)), counts)
        entry = np.repeat(first - (np.cumsum(counts) - counts), counts) \
            + np.arange(counts.sum())
        channel, box = np.divmod(row, nboxes)
        p = p[entry] + base[box, 0]
        q = q[entry] + base[box, 1]
        weights = weights[entry] * np.prod(sizes[box], axis=-1) \
            * abs(self._spec_step) * weight
        inside = (p >= 0) & (p < width) & (q >= 0) & (q < height)
        voxel = channel * npixels + index[box]
        return scipy.sparse.coo_array(
            (weights[inside], (q[inside] * width + p[inside], voxel[inside])),
            shape=(width * height, len(edges) * npixels))


class SlitlessSpectraPlan(_detail.DCubePlanBase):
    """
    The evaluation of SlitlessSpectra: the cube of the sky (see
    DCubePlanBase), then its dispersion into the image of the grid of the
    detector, the sums of its voxels with the weights of a sparse matrix
    (see RegionSumsPlan). The detector records the image, so the noise of
    its mock data goes in there.
    """

    def __init__(
            self,
            dcube: _dcube.DCube,
            dispersion: scipy.sparse.csr_array,
            grid: gridutils.Grid,
            driver: Driver,
            scene: 'Scene',
            foreground: 'Foreground',
            dtype: np.dtype,
            calibration: str | None
    ):
        super().__init__(
            dcube, driver, scene, foreground, dtype, None, calibration)
        self._dispersion = RegionSumsPlan(dispersion, driver, dtype)
        size = grid.size
        self._image = driver.mem_alloc_d((1, size[0] * size[1]), dtype)
        self._shape = size[::-1]
        self._grid = grid

    def has_exact_errors(self) -> bool:
        return True

    def _exact_errors(
            self, noise: 'Noise', recorded: np.ndarray
    ) -> dict[str, np.ndarray]:
        # (the noise is in the image of the recorded cube)
        return dict(spectra=noise.sigma(self._grid, self._image_of(recorded)))

    def _measure(self) -> ModelData:
        cube = self._dcube_plan.dcube()
        self._dispersion.evaluate(cube.reshape(1, 1, cube.size), self._image)
        return dict(spectra=dict(
            d=self._image.reshape(self._shape), m=None, w=None))

    def _mock_once(
            self, recorded: np.ndarray, noise: 'Noise', rng: np.random.Generator
    ) -> HostModelData:
        data = dict(spectra=dict(d=self._image_of(recorded), m=None))
        data['spectra']['d'] += noise.draw(
            rng, self._grid, self._dcube.psf(), data['spectra']['d'].copy())
        return data

    def _image_of(self, recorded: np.ndarray) -> np.ndarray:
        """Return the image (on the host) of the recorded cube."""
        self._driver.mem_copy_h2d(recorded, self._dcube_plan.dcube())
        return self._to_host(self._measure())['spectra']['d']
