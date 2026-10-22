from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import astropy.constants
import astropy.units
import numpy as np
import numpy.polynomial.polynomial as npp
import scipy.sparse

from gbkfit.dataset import Data, Dataset, DatasetSlitlessSpectra
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument
from gbkfit.model import LineEmission
from gbkfit.model.components._units import DataUnit
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError
from . import _dcube, _detail, _moments
from ._region_sums import RegionSumsPlan
from .base import HostModelData, ModelData, Observable

if TYPE_CHECKING:
    from .._scene import Scene
    from ..foreground import Foreground
    from ..noise import Noise
    from ..weights import Weights


__all__ = [
    'SlitlessSpectra',
    'pixel_footprint'
]


_C = astropy.constants.c.to_value('km/s')


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
    (ax, ay), (bx, by) = start, end
    lx, ly = bx - ax, by - ay
    px = np.arange(np.floor(min(ax, bx)) - 1, np.ceil(max(ax, bx)) + 2)
    py = np.arange(np.floor(min(ay, by)) - 1, np.ceil(max(ay, by)) + 2)
    p, q = (a.ravel() for a in np.meshgrid(px, py))
    # (the kinks: p - x(t) or q - y(t) at -1, 0 or 1)
    breaks = [np.zeros_like(p), np.ones_like(p)]
    for c in (-1, 0, 1):
        if lx != 0:
            breaks.append((p - c - ax) / lx)
        if ly != 0:
            breaks.append((q - c - ay) / ly)
    t = np.sort(np.clip(np.stack(breaks, axis=1), 0, 1), axis=1)
    t0, t1 = t[:, :-1], t[:, 1:]

    def overlap(t_: np.ndarray) -> np.ndarray:
        return _tri(p[:, None] - ax - t_ * lx) * _tri(q[:, None] - ay - t_ * ly)
    weights = np.sum(
        (t1 - t0) / 6 * (overlap(t0) + 4 * overlap((t0 + t1) / 2)
                         + overlap(t1)), axis=1)
    keep = weights > 0
    return p[keep].astype(int), q[keep].astype(int), weights[keep]


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
    land on the image. The trace is

        dx(w) = sum_n trace_dx[n] ((w - trace_wave) / 1 um)^n

    (pixels of the detector) and likewise dy, for a wavelength w in the
    medium of the spectral axis. It is the same for the whole image (the
    trace changes slowly across the detector). The spectral axis covers
    the lines; the continuum is not modelled.

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
    trace_wave : str or Quantity
        The wavelength the trace is expanded about.
    trace_dx : Sequence of float
        The coefficients of the trace along x (pixels; see above).
    trace_dy : Sequence of float, optional
        The coefficients of the trace along y; 0 by default.
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
        If spec_step is not positive, spec_size is less than 1, spec_rest
        is not a wavelength, or trace_dx is empty.
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
        return _detail.load_observable(cls, info, dataset, 2)

    @classmethod
    def from_data(
            cls,
            dataset: DatasetSlitlessSpectra,
            spec_step: float,
            spec_rest: str | astropy.units.Quantity,
            trace_wave: str | astropy.units.Quantity,
            trace_dx: Sequence[float],
            trace_dy: Sequence[float] = (0.0,),
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
        spec_step, spec_rest, trace_wave, trace_dx :
            As those of SlitlessSpectra.
        trace_dy, spec_size, spec_rval, spec_medium : optional
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
            spec_rest=spec_rest, trace_wave=trace_wave, trace_dx=trace_dx,
            trace_dy=trace_dy, spec_size=spec_size, spec_rval=spec_rval,
            spec_medium=spec_medium)

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
            trace_wave=str(self._trace_wave),
            trace_dx=list(self._trace_dx),
            trace_dy=list(self._trace_dy)), data)

    def __init__(
            self,
            size: Sequence[int],
            spec_step: float,
            spec_rest: str | astropy.units.Quantity,
            trace_wave: str | astropy.units.Quantity,
            trace_dx: Sequence[float],
            trace_dy: Sequence[float] = (0.0,),
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
        wave = gridutils.make_rest(trace_wave)
        if wave is None or wave.unit != astropy.units.m:
            raise ConfigError(
                f"trace_wave must be a wavelength; it is {trace_wave!r}")
        if not len(trace_dx) or not len(trace_dy):
            raise ConfigError("trace_dx and trace_dy need coefficients")
        self._spec_size = spec_size
        self._spec_step = spec_step
        self._spec_rval = spec_rval
        self._spec_rest = rest
        self._spec_medium = gridutils.make_medium(spec_medium, rest)
        self._trace_wave = wave
        self._trace_dx = tuple(float(c) for c in trace_dx)
        self._trace_dy = tuple(float(c) for c in trace_dy)

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

    def trace(self, wavelength: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the trace (dx, dy; pixels) at wavelengths (m, in the medium
        of the spectral axis).
        """
        um = (np.asarray(wavelength) - self._trace_wave.value) * 1e6
        return npp.polyval(um, self._trace_dx), npp.polyval(um, self._trace_dy)

    def keys(self) -> tuple[str, ...]:
        return ('spectra',)

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
        channels = self._channel_edges()
        footprints = [
            pixel_footprint(*zip(*self.trace(edges))) for edges in channels]
        # The sky of the cube: the pixels whose light of some channel can
        # land on the image (offset from those of the image by -offset)
        reach = np.array([
            (p.min(), p.max(), q.min(), q.max()) for p, q, _ in footprints])
        offset = (-int(reach[:, 1].max()), -int(reach[:, 3].max()))
        size = (
            self.size()[0] + int(reach[:, 1].max() - reach[:, 0].min()),
            self.size()[1] + int(reach[:, 3].max() - reach[:, 2].min()))
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
        dispersion = self._dispersion(footprints, size, offset)
        return SlitlessSpectraPlan(
            dcube, dispersion, self.grid(), driver, scene, foreground, dtype,
            calibration)

    def _channel_edges(self) -> list[np.ndarray]:
        """
        Return the wavelengths (m) of the edges of each channel of the
        model: spec_rest (1 + v / c) at their velocities.
        """
        step, n = self._spec_step, self._spec_size
        centres = self._spec_rval + step * (np.arange(n) - (n - 1) / 2)
        rest = self._spec_rest.value
        return [rest * (1 + (v + np.array([-0.5, 0.5]) * step) / _C)
                for v in centres]

    def _dispersion(
            self,
            footprints: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
            size: tuple[int, int],
            offset: tuple[int, int]
    ) -> scipy.sparse.csr_array:
        """
        Return the matrix that disperses the cube of the model (flattened,
        of shape (nz, ny, nx) on the sky of the given size, offset from the
        image) into the image (flattened): the footprint of each channel
        (see pixel_footprint) at each pixel of the sky, times the width of
        the channel (km/s), so that the image is the light integrated over
        velocity.
        """
        nx, ny = size
        width, height = self.size()
        j, i = np.mgrid[:ny, :nx]
        rows, cols, data = [], [], []
        for k, (p, q, weights) in enumerate(footprints):
            x = (i.ravel()[:, None] + offset[0] + p).ravel()
            y = (j.ravel()[:, None] + offset[1] + q).ravel()
            voxel = np.repeat(k * nx * ny + np.arange(nx * ny), len(p))
            inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            rows.append(y[inside] * width + x[inside])
            cols.append(voxel[inside])
            data.append(np.tile(weights, nx * ny)[inside])
        matrix = scipy.sparse.coo_array(
            (np.concatenate(data) * abs(self._spec_step),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(width * height, len(footprints) * nx * ny))
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
