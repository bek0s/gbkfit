from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np

from gbkfit.dataset import Data, Dataset, DatasetPixelSpectra
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument
from gbkfit.model import LineEmission
from gbkfit.model.components._units import DataUnit
from . import _dcube, _detail
from .base import ModelData, Observable

if TYPE_CHECKING:
    from .._scene import Scene
    from ..foreground import Foreground
    from ..noise import Noise
    from ..weights import Weights


__all__ = [
    'PixelSpectra'
]


class PixelSpectra(Observable):
    """
    Spectra on a grid of pixels: a spectral cube of the model, seen
    through the instrument.

    Its configuration has the options below, but those of the grid when
    it has data, which give them (see from_data).

    Parameters
    ----------
    size : Sequence of int
        The number of pixels of the cube (nx, ny, nz).
    step, rpix, rval : Sequence of float, optional
        The world coordinates of the cube (see gridutils.Coords; the
        spectral axis in km/s); rpix is the centre by default.
    rota : float, optional
        The rotation of the cube on the sky.
    rest : str or Quantity, optional
        The rest wavelength or frequency of the velocities of the spectral
        axis, if known (see gridutils.make_rest).
    medium : str, optional
        The medium of the wavelengths of a rest wavelength: 'vacuum' (the
        default) or 'air' (see gridutils.make_medium).
    convention : str, optional
        The convention of the velocities of a rest: 'optical' (the default
        of a rest wavelength), 'log' (of an axis uniform in ln w), or
        'radio' (see gridutils.make_convention).
    smooth_weights : bool, optional
        Whether the weights of the model are smoothed by the PSF and the
        LSF, as the model is.
    mask_cutoff : float, optional
        The pixels of the model whose absolute value is not above it are
        masked; no masking if None.
    mask_apply : bool, optional
        Whether the masked pixels of the model are NaN.
    median_size : int, optional
        The size (an odd number of pixels, at least 3) of the median filter
        that smoothed the data on the sky, which smooths the model as well
        (e.g. 3, of cubes smoothed before their maps were made): each
        channel takes the median of the square of the size about each
        pixel, with zero beyond the edges (see DCube); none by default.

    Raises
    ------
    ConfigError
        If median_size is not an odd number of at least 3, mask_cutoff
        is negative, or mask_apply is true without it.
    """

    dataset_class = DatasetPixelSpectra

    emission = LineEmission

    # The axes of a spectral cube: x, y and the spectral axis
    spectral_axis = 2

    @staticmethod
    def type() -> str:
        return 'pixel_spectra'

    @classmethod
    def load(
            cls, info: dict[str, Any],
            dataset: DatasetPixelSpectra | None = None
    ) -> 'PixelSpectra':
        return _detail.load_observable(cls, info, dataset, 3)

    @classmethod
    def from_data(
            cls,
            dataset: DatasetPixelSpectra,
            smooth_weights: bool = False,
            mask_cutoff: float | None = None,
            mask_apply: bool = False,
            median_size: int | None = None
    ) -> 'PixelSpectra':
        """
        Make the observable of a spectral cube, on its grid.

        Parameters
        ----------
        dataset : DatasetPixelSpectra
            The spectral cube.
        smooth_weights, mask_cutoff, mask_apply, median_size : optional
            As those of PixelSpectra.

        Returns
        -------
        PixelSpectra
            The observable.

        Raises
        ------
        ConfigError
            If the dataset is not of spectra on pixels, or the options are
            not valid.
        """
        cls._require_dataset_class(dataset)
        return cls(
            **_detail.grid_options(dataset.grid()),
            smooth_weights=smooth_weights, mask_cutoff=mask_cutoff,
            mask_apply=mask_apply, median_size=median_size)

    def dump(
            self,
            data: DatasetPixelSpectra | None = None,
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
            rest=_detail.dump_rest(self.rest()),
            medium=self.medium(),
            convention=self.convention(),
            smooth_weights=self._smooth_weights,
            mask_cutoff=self._mask_cutoff,
            mask_apply=self._mask_apply,
            median_size=self._median_size), data)

    def __init__(
            self,
            size: Sequence[int],
            step: Sequence[float] = (1, 1, 1),
            rpix: Sequence[float] | None = None,
            rval: Sequence[float] = (0, 0, 0),
            rota: float = 0,
            rest: str | astropy.units.Quantity | None = None,
            medium: str | None = None,
            convention: str | None = None,
            smooth_weights: bool = False,
            mask_cutoff: float | None = None,
            mask_apply: bool = False,
            median_size: int | None = None
    ):
        super().__init__(
            size, step, rpix, rval, rota, rest, medium, convention)
        _detail.check_mask_options(mask_cutoff, mask_apply)
        _detail.check_median_size(median_size)
        self._smooth_weights = smooth_weights
        self._mask_cutoff = mask_cutoff
        self._mask_apply = mask_apply
        self._median_size = median_size

    def keys(self) -> tuple[str, ...]:
        return ('spectra',)

    def dataset(self, items: Mapping[str, Data]) -> DatasetPixelSpectra:
        return DatasetPixelSpectra(
            items['spectra'], **_detail.dataset_grid_options(self.grid()))

    def data_unit(self, dataset: Dataset | None) -> DataUnit | None:
        return _detail.data_unit(
            dataset, 'spectra', abs(self.step()[0] * self.step()[1]), False)

    def spectra_item(self) -> str | None:
        return 'spectra'

    def model_can_lack_values(self) -> bool:
        return self._mask_apply

    def check_instrument(self, instrument: Instrument) -> None:
        _detail.warn_unsmoothed_weights(self._smooth_weights, instrument)

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
    ) -> 'PixelSpectraPlan':
        dcube = _dcube.DCube(
            size=self.size(),
            step=self.step(),
            rpix=self.rpix(),
            rval=self.rval(),
            rota=self.rota(),
            rest=self.rest(),
            medium=self.medium(),
            convention=self.convention(),
            scale=tuple(scale),
            primary_beam=instrument.primary_beam(),
            psf=instrument.psf(),
            lsf=instrument.lsf(),
            smooth_weights=self._smooth_weights,
            mask_cutoff=self._mask_cutoff,
            mask_apply=self._mask_apply,
            dtype=dtype,
            fft_dtype=fft_dtype,
            median_size=self._median_size)
        return PixelSpectraPlan(
            dcube, driver, scene, foreground, dtype, weights, calibration)


class PixelSpectraPlan(_detail.DCubePlanBase):

    def has_exact_errors(self) -> bool:
        return True

    def _exact_errors(
            self, noise: 'Noise', recorded: np.ndarray
    ) -> dict[str, np.ndarray]:
        return dict(spectra=noise.sigma(self._dcube.grid(), recorded))

    def _measure(self) -> ModelData:
        plan = self._dcube_plan
        return dict(spectra=dict(
            d=plan.dcube(), m=plan.mcube(), w=plan.wcube()))
