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
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _dcube, _detail, _moments
from ._region_sums import RegionSumsPlan
from .base import HostModelData, ModelData, Observable
from .traces import Trace, trace_parser

if TYPE_CHECKING:
    from .._scene import Scene
    from ..foreground import Foreground
    from ..noise import Noise
    from ..weights import Weights


__all__ = [
    'SlitlessSpectra',
    'pixel_footprint',
    'pixel_footprints'
]


_C = astropy.constants.c.to_value('km/s')

# The segments of a chunk of pixel_footprints (a bound of its memory)
_FOOTPRINT_CHUNK = 20000


def _tri(s: np.ndarray) -> np.ndarray:
    """Return the overlap of two unit intervals whose centres are s apart."""
    return np.maximum(0.0, 1.0 - np.abs(s))


def pixel_footprint(
        start: tuple[float, float], end: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return where the light of a pixel lands when it is moved uniformly
    along a straight segment: the light of the pixel centred at 0, spread
    evenly over the copies of the pixel centred on the segment from start
    to end (in pixels), in each pixel (p, q) of the grid.

    The fraction in a pixel is the mean over the segment of the overlap of
    the moved pixel with it: the integral over t in [0, 1] of tri(p - x(t))
    tri(q - y(t)), where (x, y) moves from start to end and tri(s) = max(0,
    1 - |s|). Between the values of t where either factor has a kink, the
    product is a quadratic, so Simpson's rule on each such interval is
    exact. The fractions add up to 1.

    Parameters
    ----------
    start, end : tuple of float
        The ends of the segment (x, y).

    Returns
    -------
    p, q : np.ndarray
        The pixels that get light (integers).
    weights : np.ndarray
        The fraction of the light in each.
    """
    _, p, q, weights = pixel_footprints([start], [end])
    return p, q, weights


def pixel_footprints(
        starts: npt.ArrayLike, ends: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the footprints of pixels moved along segments (see
    pixel_footprint), from starts to ends (of shape (n, 2), pixels): the
    segment of each fraction, its pixel (p, q) and the fraction.
    """
    starts = np.asarray(starts, dtype=float).reshape(-1, 2)
    ends = np.asarray(ends, dtype=float).reshape(-1, 2)
    out: list[list[np.ndarray]] = [[], [], [], []]
    for first in range(0, len(starts), _FOOTPRINT_CHUNK):
        chunk = slice(first, first + _FOOTPRINT_CHUNK)
        for k, part in enumerate(
                _footprints(starts[chunk], ends[chunk], first)):
            out[k].append(part)
    index, p, q, weights = (np.concatenate(parts) for parts in out)
    return index, p.astype(int), q.astype(int), weights


def _footprints(
        starts: np.ndarray, ends: np.ndarray, first: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the footprints of a chunk of segments (see pixel_footprints),
    the first of which is of the index first.
    """
    ax, ay = starts[:, 0:1], starts[:, 1:2]
    lx, ly = ends[:, 0:1] - ax, ends[:, 1:2] - ay
    corner = np.floor(np.minimum(starts, ends)) - 1
    span = (np.ceil(np.maximum(starts, ends)) + 2 - corner).max(axis=0)
    ox, oy = (o.ravel() for o in np.meshgrid(
        np.arange(span[0]), np.arange(span[1])))
    p, q = corner[:, 0:1] + ox, corner[:, 1:2] + oy
    # (the kinks: p - x(t) or q - y(t) at -1, 0 or 1)
    breaks = [np.zeros_like(p), np.ones_like(p)]
    for c in (-1, 0, 1):
        for pixel, origin, length in ((p, ax, lx), (q, ay, ly)):
            breaks.append(np.divide(
                pixel - c - origin, length, out=np.zeros_like(pixel),
                where=length != 0))
    t = np.sort(np.clip(np.stack(breaks, axis=-1), 0, 1), axis=-1)
    t0, t1 = t[..., :-1], t[..., 1:]

    def overlap(t_: np.ndarray) -> np.ndarray:
        return (_tri(p[..., None] - ax[..., None] - t_ * lx[..., None])
                * _tri(q[..., None] - ay[..., None] - t_ * ly[..., None]))
    weights = np.sum(
        (t1 - t0) / 6 * (overlap(t0) + 4 * overlap((t0 + t1) / 2)
                         + overlap(t1)), axis=-1)
    keep = weights > 0
    index = np.broadcast_to(
        first + np.arange(len(starts))[:, None], p.shape)
    return index[keep], p[keep], q[keep], weights[keep]


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

    Raises
    ------
    ConfigError
        If spec_step is not positive, spec_size is less than 1, or
        spec_rest is not a wavelength.
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
            spec_medium: str | None = None
    ) -> 'SlitlessSpectra':
        """
        Make the observable of slitless spectra, on their grid.

        Parameters
        ----------
        dataset : DatasetSlitlessSpectra
            The image.
        spec_step, spec_rest, trace :
            As those of SlitlessSpectra.
        spec_size, spec_rval, spec_medium : optional
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
            spec_rval=spec_rval, spec_medium=spec_medium)

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
            trace=trace_parser.dump(self._trace)), data)

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
            spec_medium: str | None = None
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
        self._trace = trace

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

    def trace(self) -> Trace:
        """Return the trace of the disperser."""
        return self._trace

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
        if instrument.lsf() is not None:
            raise ConfigError(
                "slitless spectra have no LSF: the PSF along the trace sets "
                "their resolution; remove the LSF of the instrument")

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
        rpix = self.rpix()
        dcube = _dcube.DCube(
            size=size + (self._spec_size,),
            step=self.step() + (self._spec_step,),
            rpix=(rpix[0] - offset[0], rpix[1] - offset[1],
                  self._spec_size / 2 - 0.5),
            rval=self.rval() + (self._spec_rval,),
            rota=self.rota(),
            rest=self._spec_rest,
            medium=self._spec_medium,
            scale=tuple(scale) + (1,),
            primary_beam=instrument.primary_beam(),
            psf=instrument.psf(),
            lsf=None,
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
        model (one row each): spec_rest (1 + v / c) at their velocities.
        """
        step, n = self._spec_step, self._spec_size
        centres = self._spec_rval + step * (np.arange(n) - (n - 1) / 2)
        velocities = centres[:, None] + np.array([-0.5, 0.5]) * step
        return self._spec_rest.value * (1 + velocities / _C)

    def _sky(
            self, edges: np.ndarray
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Return the sky of the cube, the pixels whose light of some channel
        (of the edges of its wavelengths) can land on the image: its offset
        from the image (pixels) and its size. Of the reach of the trace at
        the corners and the edges of the image, then of that sky (the trace
        changes slowly across the detector), and a pixel more each way (of
        the footprints of the pixels).
        """
        width, height = self.size()
        lo, hi = np.zeros(2), np.array([width - 1, height - 1], dtype=float)
        for _ in range(2):
            x, y = (a.ravel() for a in np.meshgrid(
                np.linspace(lo[0], hi[0], 5), np.linspace(lo[1], hi[1], 5)))
            dx, dy = self._trace.offsets(edges.ravel()[:, None], x, y)
            lo = np.array([-dx.max(), -dy.max()])
            hi = np.array([width - 1 - dx.min(), height - 1 - dy.min()])
        lo, hi = np.floor(lo) - 1, np.ceil(hi) + 1
        offset = (int(lo[0]), int(lo[1]))
        return offset, (int(hi[0] - lo[0]) + 1, int(hi[1] - lo[1]) + 1)

    def _dispersion(
            self,
            edges: np.ndarray,
            size: tuple[int, int],
            offset: tuple[int, int]
    ) -> scipy.sparse.csr_array:
        """
        Return the matrix that disperses the cube of the model (flattened,
        of shape (nz, ny, nx) on the sky of the given size, offset from the
        image) into the image (flattened): the footprint (see
        pixel_footprint) of each pixel of the sky moved by the trace there,
        from the start of each channel (of the edges of its wavelengths) to
        its end, times the width of the channel (km/s), so that the image
        is the light integrated over velocity.
        """
        nx, ny = size
        width, height = self.size()
        j, i = np.mgrid[:ny, :nx]
        x = (i.ravel() + offset[0]).astype(float)
        y = (j.ravel() + offset[1]).astype(float)
        # (the voxels in the order of the cube: channel, then pixel; the
        # segment of each from its place, a whole pixel)
        dx, dy = self._trace.offsets(edges[:, :, None], x, y)
        segments = np.stack(
            [dx[:, 0], dy[:, 0], dx[:, 1], dy[:, 1]], axis=-1).reshape(-1, 4)
        # (the footprints of the distinct segments, moved to each place: one
        # per channel of a trace the same at every place)
        distinct, inverse = np.unique(
            segments, axis=0, return_inverse=True)
        segment, p, q, weights = pixel_footprints(
            distinct[:, :2], distinct[:, 2:])
        counts = np.bincount(segment, minlength=len(distinct))
        first = np.cumsum(counts) - counts
        counts, first = counts[inverse], first[inverse]
        voxel = np.repeat(np.arange(len(inverse)), counts)
        entry = np.repeat(first - (np.cumsum(counts) - counts), counts) \
            + np.arange(counts.sum())
        p = p[entry] + np.tile(x, len(edges))[voxel].astype(int)
        q = q[entry] + np.tile(y, len(edges))[voxel].astype(int)
        weights = weights[entry]
        inside = (p >= 0) & (p < width) & (q >= 0) & (q < height)
        matrix = scipy.sparse.coo_array(
            (weights[inside] * abs(self._spec_step),
             (q[inside] * width + p[inside], voxel[inside])),
            shape=(width * height, len(edges) * nx * ny))
        return matrix.tocsr()


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
