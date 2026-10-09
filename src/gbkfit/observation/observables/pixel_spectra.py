from collections.abc import Sequence

import astropy.units

from gbkfit.dataset.datasets import DatasetPixelSpectra
from gbkfit.model.core import GModelSCube
from . import _dcube, _detail
from .core import Observable


__all__ = [
    'PixelSpectra'
]


class PixelSpectra(Observable):

    # The form of the data this observable measures
    dataset_class = DatasetPixelSpectra

    # The axes of a spectral cube: x, y and the spectral axis
    spectral_axis = 2

    @staticmethod
    def type():
        return 'pixel_spectra'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, GModelSCube)

    @classmethod
    def load(cls, info, dataset=None):
        return cls(**_detail.load_observable_common(
            cls, info, 3, dataset, DatasetPixelSpectra))

    def dump(self, data=None, prefix='', dump_path=True, overwrite=False):
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
            step: Sequence[int | float] = (1, 1, 1),
            rpix: Sequence[int | float] | None = None,
            rval: Sequence[int | float] = (0, 0, 0),
            rota: int | float = 0,
            rest: str | astropy.units.Quantity | None = None,
            smooth_weights: bool = False,
            mask_cutoff: int | float | None = None,
            mask_apply: bool = False
    ):
        """
        rest is the rest wavelength or frequency of the velocities of the
        spectral axis (see fitsutils.Coords), if known.
        """
        super().__init__(size, step, rpix, rval, rota, rest)
        self._smooth_weights = smooth_weights
        self._mask_cutoff = mask_cutoff
        self._mask_apply = mask_apply

    def keys(self):
        return ['spectra']

    def plan(
            self, driver, gmodel, foreground, instrument, scale, dtype,
            selection):
        dcube = _dcube.DCube(
            self.size(), self.step(), self.rpix(), self.rval(), self.rota(),
            self.rest(), tuple(scale), instrument.primary_beam(),
            instrument.psf(), instrument.lsf(),
            self._smooth_weights, self._mask_cutoff, self._mask_apply, dtype)
        return PixelSpectraPlan(dcube, driver, gmodel, foreground, dtype,
            selection)


class PixelSpectraPlan(_detail.DCubePlanBase):

    def evaluate(self, params, out_extra):
        self._evaluate_cube(
            params, out_extra, _dcube.cube_extra, _dcube.cube_extra)
        plan = self._dcube_plan
        return dict(spectra=dict(
            d=plan.dcube(), m=plan.mcube(), w=plan.wcube()))
