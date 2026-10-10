from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np

from gbkfit.dataset import DatasetPixelSpectra
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument
from gbkfit.model.base import Model, ModelSCube, Selection
from . import _dcube, _detail
from .base import ModelData, Observable

if TYPE_CHECKING:
    from ..foreground import Foreground


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
    smooth_weights : bool, optional
        Whether the weights of the model are smoothed by the PSF and the
        LSF, as the model is.
    mask_cutoff : float, optional
        The pixels of the model whose absolute value is not above it are
        masked; no masking if None.
    mask_apply : bool, optional
        Whether the masked pixels of the model are NaN.

    Raises
    ------
    ConfigError
        If mask_cutoff is negative, or mask_apply is true without it.
    """

    dataset_class = DatasetPixelSpectra

    # The axes of a spectral cube: x, y and the spectral axis
    spectral_axis = 2

    @staticmethod
    def type() -> str:
        return 'pixel_spectra'

    @staticmethod
    def is_compatible(model: Model) -> bool:
        return isinstance(model, ModelSCube)

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
            mask_apply: bool = False
    ) -> 'PixelSpectra':
        """
        Make the observable of a spectral cube, on its grid.

        Parameters
        ----------
        dataset : DatasetPixelSpectra
            The spectral cube.
        smooth_weights, mask_cutoff, mask_apply : optional
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
            mask_apply=mask_apply)

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
            smooth_weights=self._smooth_weights,
            mask_cutoff=self._mask_cutoff,
            mask_apply=self._mask_apply), data)

    def __init__(
            self,
            size: Sequence[int],
            step: Sequence[float] = (1, 1, 1),
            rpix: Sequence[float] | None = None,
            rval: Sequence[float] = (0, 0, 0),
            rota: float = 0,
            rest: str | astropy.units.Quantity | None = None,
            smooth_weights: bool = False,
            mask_cutoff: float | None = None,
            mask_apply: bool = False
    ):
        super().__init__(size, step, rpix, rval, rota, rest)
        _detail.check_mask_options(mask_cutoff, mask_apply)
        self._smooth_weights = smooth_weights
        self._mask_cutoff = mask_cutoff
        self._mask_apply = mask_apply

    def keys(self) -> tuple[str, ...]:
        return ('spectra',)

    def check_instrument(self, instrument: Instrument) -> None:
        _detail.warn_unsmoothed_weights(self._smooth_weights, instrument)

    def plan(
            self,
            driver: Driver,
            model: Model,
            foreground: 'Foreground',
            instrument: Instrument,
            scale: Sequence[int],
            dtype: np.dtype,
            selection: Selection
    ) -> 'PixelSpectraPlan':
        dcube = _dcube.DCube(
            size=self.size(),
            step=self.step(),
            rpix=self.rpix(),
            rval=self.rval(),
            rota=self.rota(),
            rest=self.rest(),
            scale=tuple(scale),
            primary_beam=instrument.primary_beam(),
            psf=instrument.psf(),
            lsf=instrument.lsf(),
            smooth_weights=self._smooth_weights,
            mask_cutoff=self._mask_cutoff,
            mask_apply=self._mask_apply,
            dtype=dtype)
        return PixelSpectraPlan(
            dcube, driver, model, foreground, dtype, selection)


class PixelSpectraPlan(_detail.DCubePlanBase):

    def evaluate(
            self,
            params: dict[str, float | np.ndarray],
            out_extra: dict[str, Any] | None
    ) -> ModelData:
        self._evaluate_cube(
            params, out_extra, _dcube.cube_extra, _dcube.cube_extra)
        plan = self._dcube_plan
        return dict(spectra=dict(
            d=plan.dcube(), m=plan.mcube(), w=plan.wcube()))
