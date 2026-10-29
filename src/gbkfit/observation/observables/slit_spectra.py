from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias

import numpy as np
from annotated_types import Gt

from gbkfit.dataset import Data, Dataset, DatasetSlitSpectra
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument
from gbkfit.model import DataUnit, LightUnit, LineEmission
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError, check_options
from gbkfit.utils.typeutils import Unit
from . import _dcube, _detail
from ._detail import Rota, Rval, Step
from .base import FluxScale, ModelData, Observable

if TYPE_CHECKING:
    from gbkfit.model import Scene
    from ..line_of_sight import LineOfSight
    from ..foreground import Foreground
    from ..noise import Noise
    from ..weights import Weights


__all__ = [
    'SlitSpectra'
]


# The width of a slit
_Width: TypeAlias = Annotated[float, Gt(0), Unit('arcsec')]


def _slit_extra(data: np.ndarray, grid: gridutils.Grid) -> gridutils.GridData:
    """
    Make an extra output on the low-res grid of DCube a slit (the position
    along the slit and the velocity; the slit is one pixel wide).
    """
    return gridutils.GridData(data[:, 0, :], grid.coords.axes(0, 2), 1)


class SlitSpectra(Observable):
    """
    Spectra along a long slit: the model, seen through the instrument,
    averaged across the slit at each position along it.

    Its configuration has the options below, but those of the grid when
    it has data, which give them (see from_data).

    Parameters
    ----------
    size : Sequence of int
        The number of positions along the slit (one value).
    spectral : SpectralAxis
        The spectral axis: its size, its world coordinates and the
        convention of its velocities (see gridutils.make_cube_grid).
    step, rpix, rval : Sequence of float, optional
        The world coordinates of the positions along the slit (see
        gridutils.Coords; one value each); rpix is the centre by default.
    rota : float, optional
        The angle of the slit on the sky.
    slit_width : float, optional
        The width of the slit; by default, the step along it.
    sky_rval : Sequence of float, optional
        The RA and Dec (degrees) of the reference point of the slit (rpix
        along it), the origin of its frame (see Observable.sky_rval);
        (0, 0) by default, as the grids of data without coordinates.
    smooth_weights : bool, optional
        Whether the weights of the model are smoothed by the PSF and the
        LSF, as the model is.

    Raises
    ------
    ConfigError
        If the spectral axis is not valid (see gridutils.make_cube_grid),
        slit_width is not positive, or sky_rval is not two numbers.
    """

    dataset_class = DatasetSlitSpectra

    emission = LineEmission
    orbit_stars = None

    # The axes of a long-slit spectrum: the position along the slit
    # and the spectral axis
    spectral_axis = 1

    @staticmethod
    def type() -> str:
        return 'slit_spectra'

    @classmethod
    def load(
            cls, info: dict[str, Any],
            dataset: DatasetSlitSpectra | None = None
    ) -> 'SlitSpectra':
        return _detail.load_observable(cls, info, dataset, 1)

    @classmethod
    def from_data(
            cls,
            dataset: DatasetSlitSpectra,
            slit_width: _Width | None = None,
            sky_rval: Rval = (0, 0),
            smooth_weights: bool = False
    ) -> 'SlitSpectra':
        """
        Make the observable of the spectra of a slit, on their grid.

        Parameters
        ----------
        dataset : DatasetSlitSpectra
            The spectra.
        slit_width, sky_rval, smooth_weights : optional
            As those of SlitSpectra.

        Returns
        -------
        SlitSpectra
            The observable.

        Raises
        ------
        ConfigError
            If the dataset is not of slit spectra, or the options are not
            valid.
        """
        check_options(SlitSpectra.from_data, locals())
        return cls(
            **_detail.grid_options(dataset.grid()), slit_width=slit_width,
            sky_rval=sky_rval, smooth_weights=smooth_weights)

    def dump(
            self,
            data: DatasetSlitSpectra | None = None,
            prefix: str = '',
            dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _detail.without_options_from_data(self, dict(
            type=self.type(),
            **_detail.dump_grid(self.grid()),
            slit_width=self._slit_width,
            sky_rval=self._sky_rval,
            smooth_weights=self._smooth_weights), data)

    def __init__(
            self,
            size: Sequence[int],
            spectral: gridutils.SpectralAxis,
            step: Step = (1,),
            rpix: Sequence[float] | None = None,
            rval: Rval = (0,),
            rota: Rota = 0,
            slit_width: _Width | None = None,
            sky_rval: Rval = (0, 0),
            smooth_weights: bool = False
    ):
        check_options(SlitSpectra.__init__, locals())
        super().__init__(size, step, rpix, rval, rota, spectral)
        if slit_width is None:
            slit_width = self.step()[0]
        sky_rval = tuple(sky_rval)
        if len(sky_rval) != 2:
            raise ConfigError(
                f"sky_rval is the RA and Dec of the reference point of the "
                f"slit; it is {sky_rval}")
        self._sky_rval = float(sky_rval[0]), float(sky_rval[1])
        self._slit_width = slit_width
        self._smooth_weights = smooth_weights

    def slit_width(self) -> float:
        """Return the width of the slit."""
        return self._slit_width

    def sky_rval(self) -> tuple[float, float]:
        return self._sky_rval

    def keys(self) -> tuple[str, ...]:
        return ('spectra',)

    def dataset(self, items: Mapping[str, Data]) -> DatasetSlitSpectra:
        return DatasetSlitSpectra(
            items['spectra'], **_detail.dataset_grid_options(self.grid()))

    def data_unit(
            self,
            dataset: Dataset | None,
            light: LightUnit | None = None
    ) -> DataUnit | None:
        # (a pixel spans the step along the slit and the width of the slit)
        return _detail.data_unit(
            dataset, 'spectra', abs(self.step()[0]) * self._slit_width,
            False, light)

    def light_items(self) -> tuple[str, ...]:
        return ('spectra',)

    def flux_scales(self) -> dict[str, FluxScale]:
        # (each spectrum runs along the first axis, at each position along
        # the slit)
        return super().flux_scales() \
            | dict(per_spectrum=FluxScale(('spectra',), 0, self.size()[0]))

    def linear_in_light(self) -> bool:
        # (its spectra of the light, through its instrument: linear)
        return True

    def model_can_lack_values(self) -> bool:
        return False

    def check_observation(
            self,
            instrument: Instrument,
            oversampling: Sequence[int],
            pixel_weights: 'Weights | None'
    ) -> None:
        _detail.warn_unsmoothed_weights(self._smooth_weights, instrument)

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
    ) -> 'SlitSpectraPlan':
        # The cube of a slit is one pixel (the slit width) across the slit,
        # sampled as finely as the pixels along it (the model is the mean
        # over the slit)
        size, step = self.size(), self.step()
        rpix, rval = self.rpix(), self.rval()
        across = oversampling[0] * max(1, round(self._slit_width / step[0]))
        dcube = _dcube.DCube(
            size=(size[0], 1, size[1]),
            step=(step[0], self._slit_width, step[1]),
            rpix=(rpix[0], 0, rpix[1]),
            rval=(rval[0], 0, rval[1]),
            rota=self.rota(),
            rest=self.rest(),
            medium=self.medium(),
            convention=self.convention(),
            oversampling=(oversampling[0], across, oversampling[1]),
            primary_beam=instrument.primary_beam(),
            psf=instrument.psf(),
            lsf=instrument.lsf(),
            smooth_weights=self._smooth_weights,
            dtype=dtype,
            fft_dtype=fft_dtype)
        return SlitSpectraPlan(
            dcube, driver, scene, line_of_sight, foreground, dtype,
            pixel_weights, flux_scale)


class SlitSpectraPlan(_detail.DCubePlanBase):

    def _model_extra(self, value: gridutils.GridData) -> np.ndarray:
        # The slit has no position on the sky, so the extra outputs of the
        # model, which are on the sky, are plain arrays
        return value.data

    def has_exact_errors(self) -> bool:
        return True

    def _exact_errors(
            self, noise: 'Noise', recorded: np.ndarray
    ) -> dict[str, np.ndarray]:
        return dict(
            spectra=noise.sigma(self._dcube.grid(), recorded)[:, 0, :])

    # The slit has no position on the sky, so the extra outputs of the
    # high-res grid are plain arrays
    _extra_lo = staticmethod(_slit_extra)
    _extra_hi = staticmethod(_dcube.plain_extra)

    def _measure(self) -> ModelData:
        plan = self._dcube_plan
        return dict(spectra=dict(
            d=plan.dcube()[:, 0, :],
            m=None,
            w=plan.wcube()[:, 0, :] if plan.wcube() is not None else None))
