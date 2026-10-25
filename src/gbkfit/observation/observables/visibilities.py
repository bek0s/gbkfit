from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np

from gbkfit.dataset import Data, Dataset, DatasetVisibilities, UVCoverage
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument
from gbkfit.model import DataUnit, GalaxyState, LineEmission
from gbkfit.params import ParamValues
from gbkfit.utils import gridutils, parseutils, unitutils
from gbkfit.utils.parseutils import ConfigError
from . import _dcube, _detail
from ._uv_transforms import UVTransform, uv_transforms
from .base import HostModelData, ModelData, Observable

if TYPE_CHECKING:
    from gbkfit.model import Scene
    from ..line_of_sight import LineOfSight
    from ..foreground import Foreground
    from ..noise import Noise
    from ..weights import Weights


__all__ = [
    'Visibilities'
]


# The speed of light (km/s and m/s), and arcsec per radian
_C_KMS = 299792.458
_C = 299792458.0
_ARCSEC = np.degrees(1) * 3600

# The tolerance of the transforms of models of each type: about the
# rounding of their values (that FINUFFT reaches, for float64)
_EPS = {np.dtype(np.float32): 1e-7, np.dtype(np.float64): 1e-12}


class Visibilities(Observable):
    """
    Interferometric visibilities (see DatasetVisibilities): the Fourier
    transform of the cube of the model on the sky, seen through the
    instrument, at the baselines of each channel (see uv_transforms).

    The spatial pixels of the cube are the points of the sum of the
    transform, at their centres (not integrated over their area, which
    would damp the visibilities of long baselines: the observation must
    not oversample them), so their step must sample the model and the
    longest baselines (at most 1 / (2 u_max), the default), and the grid
    cover the emission times the primary beam. Its spectral axis is that
    of the data: the radio velocities of their channels about the rest
    frequency of their coverage, in the order of increasing velocity. The
    instrument has no PSF (the baselines sample the visibilities
    themselves); its LSF is the response of the correlator (e.g. Hanning
    smoothing), and its primary beam attenuates the cube.

    Its configuration has the options below, but the coverage when it has
    data, which give it; without data, the coverage is a file of its
    arrays (see UVCoverage.from_file).

    The spatial weights of data do not apply (the data are not pixels);
    a mosaic is one observation for each pointing, fitted jointly. Not
    supported yet: averaging the visibilities in time or frequency (an
    approximation, to be opt-in).

    Parameters
    ----------
    coverage : UVCoverage
        Where the visibilities are measured.
    size : Sequence of int
        The number of pixels of the sky (nx, ny).
    step : float or Sequence of float, optional
        The size of the pixels (arcsec); by default 1 / (2 u_max), of the
        longest baseline at the highest frequency.
    rpix : Sequence of float, optional
        The reference pixel; by default the centre.
    rval : Sequence of float, optional
        The RA and Dec of the reference pixel (degrees); by default the
        phase centre.
    rota : float, optional
        The rotation of the grid on the sky (see gridutils.Coords).
    transform : str, optional
        The transform (see uv_transforms): 'exact' (by default), 'sin'
        (the pixels of a SIN projection), or 'flat' (those, without the
        w-term).
    eps : float, optional
        The relative tolerance of the transform; by default about the
        rounding of the model: 1e-7 for float32, 1e-12 for float64.

    Raises
    ------
    ConfigError
        If the transform is not one of those, or eps is not positive.
    """

    # TODO: averaging the visibilities in time or frequency, as an opt-in
    # approximation (for now, the data are averaged by CASA, e.g. split
    # with timebin and width)

    dataset_class = DatasetVisibilities

    emission = LineEmission

    # The axes of the cube of the model: x, y and the spectral axis
    spectral_axis = 2

    @staticmethod
    def type() -> str:
        return 'visibilities'

    @classmethod
    def options_from_data(cls, dataset: Dataset) -> tuple[str, ...]:
        return ('coverage',)

    @classmethod
    def load(
            cls, info: dict[str, Any],
            dataset: DatasetVisibilities | None = None
    ) -> 'Visibilities':
        if dataset is None and 'coverage' in info:
            info['coverage'] = UVCoverage.from_file(
                parseutils.resolve_path(info['coverage']))
        return _detail.load_observable(cls, info, dataset, 2)

    @classmethod
    def from_data(
            cls,
            dataset: DatasetVisibilities,
            size: Sequence[int],
            step: float | Sequence[float] | None = None,
            rpix: Sequence[float] | None = None,
            rval: Sequence[float] | None = None,
            rota: float = 0,
            transform: str = 'exact',
            eps: float | None = None
    ) -> 'Visibilities':
        """
        Make the observable of visibilities, of their coverage.

        Parameters
        ----------
        dataset : DatasetVisibilities
            The visibilities.
        size, step, rpix, rval, rota, transform, eps : optional
            As those of Visibilities.

        Returns
        -------
        Visibilities
            The observable.

        Raises
        ------
        ConfigError
            If the dataset is not of visibilities, or the options are not
            valid.
        """
        cls._require_dataset_class(dataset)
        return cls(
            dataset.coverage(), size, step, rpix, rval, rota, transform, eps)

    def dump(
            self,
            data: DatasetVisibilities | None = None,
            prefix: str = '',
            dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        info = dict(type=self.type())
        if data is None:
            info.update(coverage=self._coverage.dump(
                f'{prefix}coverage.npz', dump_path, overwrite))
        return info | dict(
            size=self.size()[:2],
            step=self.step()[:2],
            rpix=self.rpix()[:2],
            rval=self.rval()[:2],
            rota=self.rota(),
            transform=self._transform,
            eps=self._eps)

    def __init__(
            self,
            coverage: UVCoverage,
            size: Sequence[int],
            step: float | Sequence[float] | None = None,
            rpix: Sequence[float] | None = None,
            rval: Sequence[float] | None = None,
            rota: float = 0,
            transform: str = 'exact',
            eps: float | None = None
    ):
        if transform not in uv_transforms:
            raise ConfigError(
                f"transform must be one of {list(uv_transforms)}; it is "
                f"{transform!r}")
        if eps is not None:
            _detail.check_positive('eps', eps)
        size = tuple(size)
        if step is None:
            step = _nyquist_step(coverage)
        step = (step,) * 2 if np.ndim(step) == 0 else tuple(step)
        if rpix is None:
            rpix = tuple((np.asarray(size, float) / 2 - 0.5).tolist())
        if rval is None:
            rval = coverage.phase_centre()
        # The spectral axis: the radio velocities of the channels, from the
        # highest frequency
        frequencies = coverage.frequencies()
        rest = coverage.rest().value
        channel = _C_KMS * abs(frequencies[1] - frequencies[0]) / rest
        velocity = _C_KMS * (1 - frequencies.max() / rest)
        super().__init__(
            size + (coverage.nchannels(),), step + (channel,),
            tuple(rpix) + (0,), tuple(rval) + (velocity,), rota,
            coverage.rest())
        self._coverage = coverage
        self._transform = transform
        self._eps = eps

    def coverage(self) -> UVCoverage:
        """Return where the visibilities are measured."""
        return self._coverage

    def transform(self) -> str:
        """Return the name of the transform (see uv_transforms)."""
        return self._transform

    def eps(self) -> float | None:
        """Return the tolerance of the transform, if given."""
        return self._eps

    def keys(self) -> tuple[str, ...]:
        return ('re', 'im')

    def dataset(self, items: Mapping[str, Data]) -> DatasetVisibilities:
        return DatasetVisibilities(items['re'], items['im'], self._coverage)

    def data_unit(self, dataset: Dataset | None) -> DataUnit | None:
        # (the visibilities are sums of the flux of the pixels: of their
        # surface brightness times their area, see VisibilitiesPlan)
        return _detail.region_data_unit(dataset, 're', False)

    def light_items(self) -> tuple[str, ...]:
        return ('re', 'im')

    def model_can_lack_values(self) -> bool:
        return False

    def check_instrument(self, instrument: Instrument) -> None:
        if instrument.psf() is not None:
            raise ConfigError(
                "visibilities have no PSF: their baselines sample the "
                "transform of the sky itself (the dirty beam is their "
                "sampling); remove the psf of the instrument")

    def _require_matching_coordinates(
            self, dataset: DatasetVisibilities
    ) -> None:
        if dataset.coverage() != self._coverage:
            raise ConfigError(
                "the data and the observable have different coverages")

    def output(self, data: np.ndarray) -> np.ndarray:
        return data

    def images(
            self,
            driver: Driver,
            values: Mapping[str, np.ndarray],
            weights: Mapping[str, np.ndarray] | None,
            units: Mapping[str, astropy.units.UnitBase | None] | None
    ) -> dict[str, gridutils.GridData]:
        # (of visibilities in Jy, a dirty image is in Jy/beam)
        image = self._dirty_image(
            driver, values['re'] + 1j * values['im'], weights)
        unit = None if units is None else units['re']
        if unit is not None:
            image = image._replace(
                unit=unitutils.format_unit(unit / astropy.units.beam))
        return dict(dirty=image)

    def psf_images(
            self,
            driver: Driver,
            weights: Mapping[str, np.ndarray] | None
    ) -> dict[str, gridutils.GridData]:
        # (the dirty image of a point of 1 at the phase centre)
        shape = (self._coverage.nchannels(), self._coverage.nrows())
        return dict(dirty_beam=self._dirty_image(
            driver, np.ones(shape, complex), weights))

    def _dirty_image(
            self,
            driver: Driver,
            visibilities: np.ndarray,
            weights: Mapping[str, np.ndarray] | None
    ) -> gridutils.GridData:
        """
        Return the dirty image of visibilities, of natural weights: the
        adjoint of the transform of the weighted visibilities, over the sum
        of the weights of each channel (a point of flux S at a pixel peaks
        at S there), on the grid; without values where the weights are not
        finite or the visibilities are NaN.
        """
        w = np.ones(visibilities.shape) if weights is None \
            else np.asarray(weights['re'], float)
        valid = np.isfinite(visibilities) & np.isfinite(w)
        w = np.where(valid, w, 0)
        weighted = np.where(valid, visibilities, 0) * w
        frequencies = self._coverage.frequencies()
        if frequencies[1] > frequencies[0]:
            # (the channels of the grid are from the highest frequency)
            frequencies, weighted, w = frequencies[::-1], weighted[::-1], w[::-1]
        transform = uv_transforms[self._transform](
            driver, self.grid().spatial(), self._coverage.uvw(), frequencies,
            self._coverage.phase_centre(),
            _EPS[np.dtype(np.float64)] if self._eps is None else self._eps)
        images = driver.mem_alloc_d(self.size()[::-1], np.complex128)
        transform.adjoint(
            driver.mem_copy_h2d(np.ascontiguousarray(weighted)), images)
        with np.errstate(divide='ignore', invalid='ignore'):
            image = driver.mem_copy_d2h(images).real \
                / w.sum(axis=1)[:, None, None]
        return gridutils.GridData(image, self._grid.coords, 2)

    def plan(
            self,
            driver: Driver,
            scene: 'Scene',
            line_of_sight: 'LineOfSight',
            foreground: 'Foreground',
            instrument: Instrument,
            oversampling: Sequence[int],
            dtype: np.dtype,
            fft_dtype: np.dtype,
            pixel_weights: 'Weights | None',
            flux_scale: str | None
    ) -> 'VisibilitiesPlan':
        if tuple(oversampling[:2]) != (1, 1):
            raise ConfigError(
                f"the pixels of visibilities are the points of their "
                f"transform, not oversampled (the means of their subpixels "
                f"would damp the long baselines): give a smaller step "
                f"instead; the spatial oversampling is "
                f"{tuple(oversampling[:2])}")
        if pixel_weights is not None:
            raise ConfigError(
                "pixel_weights do not apply to visibilities, which are not "
                "on pixels")
        dcube = _dcube.DCube(
            size=self.size(),
            step=self.step(),
            rpix=self.rpix(),
            rval=self.rval(),
            rota=self.rota(),
            rest=self.rest(),
            medium=None,
            oversampling=tuple(oversampling),
            primary_beam=instrument.primary_beam(),
            psf=None,
            lsf=instrument.lsf(),
            smooth_weights=False,
            dtype=dtype,
            fft_dtype=fft_dtype)
        # The channels of the cube are those of the data from the highest
        # frequency
        frequencies = self._coverage.frequencies()
        reversed_ = bool(frequencies[1] > frequencies[0])
        transform = uv_transforms[self._transform](
            driver, self.grid().spatial(), self._coverage.uvw(),
            frequencies[::-1] if reversed_ else frequencies,
            self._coverage.phase_centre(),
            _EPS[np.dtype(dtype)] if self._eps is None else self._eps)
        return VisibilitiesPlan(
            transform, abs(self.step()[0] * self.step()[1]), reversed_, dcube,
            driver, scene, line_of_sight, foreground, dtype, flux_scale)


class VisibilitiesPlan(_detail.DCubePlanBase):
    """
    The evaluation of visibilities: the transform of the flux of the
    pixels of the cube of the model, recorded as the instrument records it
    (see DCubePlanBase): of its surface brightness (per arcsec^2, see
    Visibilities.data_unit) times the area of its pixels; split into their
    real and imaginary parts. With the channels of the data in the order
    of increasing frequency, the transform of the cube (from the highest
    frequency) goes into them in reverse.

    Its mock data are the visibilities of the model plus noise, drawn for
    each part on their grid: the rows along x and the channels along y
    (see noise_grid), for a noise correlated along the channels (e.g. by
    Hanning smoothing; not on the sky, which visibilities are not on).
    """

    def __init__(
            self,
            transform: UVTransform,
            area: float,
            reversed_: bool,
            dcube: _dcube.DCube,
            driver: Driver,
            scene: 'Scene',
            line_of_sight: 'LineOfSight',
            foreground: 'Foreground',
            dtype: np.dtype,
            flux_scale: str | None
    ):
        super().__init__(
            dcube, driver, scene, line_of_sight, foreground, dtype, None,
            flux_scale)
        self._transform = transform
        self._area = area
        self._reversed = reversed_
        shape = (dcube.size()[2], transform.nrows())
        spectral = dcube.grid().spectral().coords
        self._noise_grid = gridutils.make_grid(
            shape[::-1], (1, spectral.step[0]), (0, 0), (0, spectral.rval[0]),
            spectral_axis=1, rest=spectral.rest)
        self._visibilities = driver.mem_alloc_d(shape, np.complex128)
        self._re = driver.mem_alloc_d(shape, dtype)
        self._im = driver.mem_alloc_d(shape, dtype)

    def mock(
            self,
            params: ParamValues,
            galaxies: Sequence[GalaxyState],
            noise: 'Noise',
            rng: np.random.Generator,
            ndraws: int
    ) -> list[HostModelData]:
        self._evaluate_model(params, galaxies, None)
        self._dcube_plan.record(self._flux_scale(params))
        recorded = self._to_host(self._measure())
        draws = [{
            key: dict(d=(item['d'] + noise.draw(
                rng, self._noise_grid, None, item['d'])).astype(self._dtype),
                m=None)
            for key, item in recorded.items()} for _ in range(ndraws)]
        errors = {key: noise.sigma(self._noise_grid, item['d'])
                  for key, item in recorded.items()}
        return draws, errors

    def has_exact_errors(self) -> bool:
        return True

    def noise_grid(self) -> gridutils.Grid:
        """
        Return the grid of the noise of the visibilities: the rows along x
        and the channels along y, of their velocity step (as the arrays of
        the data items, (nchannels, nrows)).
        """
        return self._noise_grid

    def _measure(self) -> ModelData:
        visibilities = self._visibilities
        self._transform(
            self._dcube_plan.dcube(),
            visibilities[::-1] if self._reversed else visibilities)
        self._driver.math_mul(visibilities, self._area, out=visibilities)
        self._driver.mem_copy_d2d(visibilities.real, self._re)
        self._driver.mem_copy_d2d(visibilities.imag, self._im)
        return dict(
            re=dict(d=self._re, m=None, w=None),
            im=dict(d=self._im, m=None, w=None))


def _nyquist_step(coverage: UVCoverage) -> float:
    """
    Return the step (arcsec) that samples the longest baseline of a
    coverage at its highest frequency: 1 / (2 u_max).
    """
    uv = np.hypot(*coverage.uvw()[:, :2].T).max(initial=0)
    u_max = uv * coverage.frequencies().max() / _C
    if u_max == 0:
        raise ConfigError(
            "the baselines are all of zero length; give the step of the "
            "pixels")
    return float(_ARCSEC / (2 * u_max))
