from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np

from gbkfit.dataset import DatasetSlitSpectra
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument
from gbkfit.model.base import Model, ModelSCube, Selection
from gbkfit.utils import gridutils
from . import _dcube, _detail
from .base import ModelData, Observable

if TYPE_CHECKING:
    from ..foreground import Foreground


__all__ = [
    'SlitSpectra'
]


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
        The number of positions along the slit and of channels.
    step, rpix, rval : Sequence of float, optional
        The world coordinates of the positions along the slit and of the
        spectral axis (see gridutils.Coords; km/s); rpix is the centre by
        default.
    rota : float, optional
        The angle of the slit on the sky.
    rest : str or Quantity, optional
        The rest wavelength or frequency of the velocities of the spectral
        axis, if known (see gridutils.make_rest).
    slit_width : float, optional
        The width of the slit; by default, the step along it.
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
        If slit_width is not positive, mask_cutoff is negative, or
        mask_apply is true without it.
    """

    dataset_class = DatasetSlitSpectra

    # The axes of a long-slit spectrum: the position along the slit
    # and the spectral axis
    spectral_axis = 1

    @staticmethod
    def type() -> str:
        return 'slit_spectra'

    @staticmethod
    def is_compatible(model: Model) -> bool:
        return isinstance(model, ModelSCube)

    @classmethod
    def load(
            cls, info: dict[str, Any],
            dataset: DatasetSlitSpectra | None = None
    ) -> 'SlitSpectra':
        return _detail.load_observable(cls, info, dataset, 2)

    @classmethod
    def from_data(
            cls,
            dataset: DatasetSlitSpectra,
            slit_width: float | None = None,
            smooth_weights: bool = False,
            mask_cutoff: float | None = None,
            mask_apply: bool = False
    ) -> 'SlitSpectra':
        """
        Make the observable of the spectra of a slit, on their grid.

        Parameters
        ----------
        dataset : DatasetSlitSpectra
            The spectra.
        slit_width, smooth_weights, mask_cutoff, mask_apply : optional
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
        cls._require_dataset_class(dataset)
        return cls(
            **_detail.grid_options(dataset.grid()), slit_width=slit_width,
            smooth_weights=smooth_weights, mask_cutoff=mask_cutoff,
            mask_apply=mask_apply)

    def dump(
            self,
            data: DatasetSlitSpectra | None = None,
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
            slit_width=self._slit_width,
            smooth_weights=self._smooth_weights,
            mask_cutoff=self._mask_cutoff,
            mask_apply=self._mask_apply), data)

    def __init__(
            self,
            size: Sequence[int],
            step: Sequence[float] = (1, 1),
            rpix: Sequence[float] | None = None,
            rval: Sequence[float] = (0, 0),
            rota: float = 0,
            rest: str | astropy.units.Quantity | None = None,
            slit_width: float | None = None,
            smooth_weights: bool = False,
            mask_cutoff: float | None = None,
            mask_apply: bool = False
    ):
        super().__init__(size, step, rpix, rval, rota, rest)
        if slit_width is None:
            slit_width = self.step()[0]
        _detail.check_positive('slit_width', slit_width)
        _detail.check_mask_options(mask_cutoff, mask_apply)
        self._slit_width = slit_width
        self._smooth_weights = smooth_weights
        self._mask_cutoff = mask_cutoff
        self._mask_apply = mask_apply

    def slit_width(self) -> float:
        """Return the width of the slit."""
        return self._slit_width

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
    ) -> 'SlitSpectraPlan':
        # The cube of a slit is one pixel (the slit width) across the slit,
        # sampled as finely as the pixels along it (the model is the mean
        # over the slit)
        size, step = self.size(), self.step()
        rpix, rval = self.rpix(), self.rval()
        across = scale[0] * max(1, round(self._slit_width / step[0]))
        dcube = _dcube.DCube(
            size=(size[0], 1, size[1]),
            step=(step[0], self._slit_width, step[1]),
            rpix=(rpix[0], 0, rpix[1]),
            rval=(rval[0], 0, rval[1]),
            rota=self.rota(),
            rest=self.rest(),
            scale=(scale[0], across, scale[1]),
            primary_beam=instrument.primary_beam(),
            psf=instrument.psf(),
            lsf=instrument.lsf(),
            smooth_weights=self._smooth_weights,
            mask_cutoff=self._mask_cutoff,
            mask_apply=self._mask_apply,
            dtype=dtype)
        return SlitSpectraPlan(
            dcube, driver, model, foreground, dtype, selection)


class SlitSpectraPlan(_detail.DCubePlanBase):

    def _model_extra(self, value: gridutils.GridData) -> np.ndarray:
        # The slit has no position on the sky, so the extra outputs of the
        # model, which are on the sky, are plain arrays
        return value.data

    def evaluate(
            self,
            params: dict[str, float | np.ndarray],
            out_extra: dict[str, Any] | None
    ) -> ModelData:
        # The slit has no position on the sky, so the extra outputs of the
        # high-res grid are plain arrays
        self._evaluate_cube(
            params, out_extra, _slit_extra, _dcube.plain_extra)
        plan = self._dcube_plan
        return dict(spectra=dict(
            d=plan.dcube()[:, 0, :],
            m=plan.mcube()[:, 0, :] if plan.mcube() is not None else None,
            w=plan.wcube()[:, 0, :] if plan.wcube() is not None else None))
