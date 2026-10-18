from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from gbkfit.dataset import Data, Dataset, DatasetPixelBrightness
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument
from gbkfit.model import BandEmission
from gbkfit.model.components._units import DataUnit
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError
from . import _dcube, _detail
from .base import ModelData, Observable

if TYPE_CHECKING:
    from .._scene import Scene
    from ..foreground import Foreground
    from ..noise import Noise
    from ..weights import Weights


__all__ = [
    'PixelBrightness'
]


def _image_extra(data: np.ndarray, grid: gridutils.Grid) -> gridutils.GridData:
    """Make an extra output on a grid of DCube an image (its one channel)."""
    return gridutils.GridData(data[0], grid.spatial().coords, None)


class PixelBrightness(Observable):
    """
    Brightness on a grid of pixels: an image of the model, seen through
    the instrument (which has no LSF).

    Its configuration has the options below, but those of the grid when
    it has data, which give them (see from_data).

    Parameters
    ----------
    size : Sequence of int
        The number of pixels of the image (nx, ny).
    step, rpix, rval : Sequence of float, optional
        The world coordinates of the image (see gridutils.Coords); rpix is
        the centre by default.
    rota : float, optional
        The rotation of the image on the sky.
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

    dataset_class = DatasetPixelBrightness

    emission = BandEmission

    # An image has no spectral axis
    spectral_axis = None

    @staticmethod
    def type() -> str:
        return 'pixel_brightness'

    @classmethod
    def load(
            cls, info: dict[str, Any],
            dataset: DatasetPixelBrightness | None = None
    ) -> 'PixelBrightness':
        return _detail.load_observable(cls, info, dataset, 2)

    @classmethod
    def from_data(
            cls,
            dataset: DatasetPixelBrightness,
            mask_cutoff: float | None = None,
            mask_apply: bool = False
    ) -> 'PixelBrightness':
        """
        Make the observable of an image, on its grid.

        Parameters
        ----------
        dataset : DatasetPixelBrightness
            The image.
        mask_cutoff, mask_apply : optional
            As those of PixelBrightness.

        Returns
        -------
        PixelBrightness
            The observable.

        Raises
        ------
        ConfigError
            If the dataset is not of brightness, or the options are not
            valid.
        """
        cls._require_dataset_class(dataset)
        return cls(
            **_detail.grid_options(dataset.grid()), mask_cutoff=mask_cutoff,
            mask_apply=mask_apply)

    def dump(
            self,
            data: DatasetPixelBrightness | None = None,
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
            mask_cutoff=self._mask_cutoff,
            mask_apply=self._mask_apply), data)

    def __init__(
            self,
            size: Sequence[int],
            step: Sequence[float] = (1, 1),
            rpix: Sequence[float] | None = None,
            rval: Sequence[float] = (0, 0),
            rota: float = 0,
            mask_cutoff: float | None = None,
            mask_apply: bool = False
    ):
        super().__init__(size, step, rpix, rval, rota)
        _detail.check_mask_options(mask_cutoff, mask_apply)
        self._mask_cutoff = mask_cutoff
        self._mask_apply = mask_apply

    def keys(self) -> tuple[str, ...]:
        return ('brightness',)

    def dataset(self, items: Mapping[str, Data]) -> DatasetPixelBrightness:
        return DatasetPixelBrightness(
            items['brightness'], **_detail.dataset_grid_options(self.grid()))

    def data_unit(self, dataset: Dataset | None) -> DataUnit | None:
        return _detail.data_unit(
            dataset, 'brightness', abs(self.step()[0] * self.step()[1]),
            False)

    def model_can_lack_values(self) -> bool:
        return self._mask_apply

    def check_instrument(self, instrument: Instrument) -> None:
        if instrument.lsf() is not None:
            raise ConfigError(
                "an image has no spectral axis for an LSF; remove the LSF "
                "of the instrument")

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
    ) -> 'PixelBrightnessPlan':
        # The cube of an image has one channel
        dcube = _dcube.DCube(
            size=self.size() + (1,),
            step=self.step() + (0,),
            rpix=self.rpix() + (0,),
            rval=self.rval() + (0,),
            rota=self.rota(),
            rest=None,
            medium=None,
            scale=tuple(scale) + (1,),
            primary_beam=instrument.primary_beam(),
            psf=instrument.psf(),
            lsf=None,
            smooth_weights=False,
            mask_cutoff=self._mask_cutoff,
            mask_apply=self._mask_apply,
            dtype=dtype,
            fft_dtype=fft_dtype)
        return PixelBrightnessPlan(
            dcube, driver, scene, foreground, dtype, weights, calibration)


class PixelBrightnessPlan(_detail.DCubePlanBase):

    def _model_grid(self) -> gridutils.Grid:
        # Image models are evaluated on the x and y axes
        return self._dcube_plan.scratch_grid().spatial()

    def exact_errors(self, noise: 'Noise') -> dict[str, np.ndarray]:
        return dict(brightness=noise.sigma(self._dcube.grid())[0])

    _extra_lo = staticmethod(_image_extra)
    _extra_hi = staticmethod(_image_extra)

    def _measure(self) -> ModelData:
        plan = self._dcube_plan
        return dict(brightness=dict(
            d=plan.dcube()[0, :, :],
            m=plan.mcube()[0, :, :] if plan.mcube() is not None else None,
            w=plan.wcube()[0, :, :] if plan.wcube() is not None else None))
