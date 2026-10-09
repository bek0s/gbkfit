from collections.abc import Sequence

import astropy.units

from gbkfit.dataset.datasets import DatasetSlitSpectra
from gbkfit.model.base import GModelSCube
from gbkfit.utils import gridutils
from . import _dcube, _detail
from .base import Observable


__all__ = [
    'SlitSpectra'
]


def _slit_extra(data, grid):
    """
    An extra output on the low-res grid of DCube, as a slit (the position
    along the slit and the velocity; the slit is one pixel wide).
    """
    return gridutils.GridData(data[:, 0, :], grid.coords.axes(0, 2), 1)


class SlitSpectra(Observable):

    # The form of the data this observable measures
    dataset_class = DatasetSlitSpectra

    # The axes of a long-slit spectrum: the position along the slit
    # and the spectral axis
    spectral_axis = 1

    @staticmethod
    def type():
        return 'slit_spectra'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, GModelSCube)

    @classmethod
    def load(cls, info, dataset=None):
        return cls(**_detail.load_observable_common(
            cls, info, 2, dataset, DatasetSlitSpectra))

    def dump(self, data=None, prefix='', dump_path=True, overwrite=False):
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
        """
        rest is the rest wavelength or frequency of the velocities of the
        spectral axis (see gridutils.Coords), if known.
        """
        super().__init__(size, step, rpix, rval, rota, rest)
        self._slit_width = slit_width if slit_width is not None \
            else self.step()[0]
        self._smooth_weights = smooth_weights
        self._mask_cutoff = mask_cutoff
        self._mask_apply = mask_apply

    def slit_width(self):
        return self._slit_width

    def keys(self):
        return ['spectra']

    def plan(
            self, driver, gmodel, foreground, instrument, scale, dtype,
            selection):
        # The cube of a slit is one pixel (the slit width) across the slit,
        # sampled as finely as the pixels along it (the model is the mean
        # over the slit)
        size, step = self.size(), self.step()
        rpix, rval = self.rpix(), self.rval()
        across = scale[0] * max(1, round(self._slit_width / step[0]))
        dcube = _dcube.DCube(
            (size[0], 1, size[1]), (step[0], self._slit_width, step[1]),
            (rpix[0], 0, rpix[1]), (rval[0], 0, rval[1]), self.rota(),
            self.rest(), (scale[0], across, scale[1]),
            instrument.primary_beam(), instrument.psf(),
            instrument.lsf(), self._smooth_weights, self._mask_cutoff,
            self._mask_apply, dtype)
        return SlitSpectraPlan(dcube, driver, gmodel, foreground, dtype,
            selection)


class SlitSpectraPlan(_detail.DCubePlanBase):

    def _gmodel_extra(self, value):
        # The slit has no position on the sky, so the extra outputs of the
        # gmodel, which are on the sky, are plain arrays
        return value.data

    def evaluate(self, params, out_extra):
        # The slit has no position on the sky, so the extra outputs of the
        # high-res grid are plain arrays
        self._evaluate_cube(
            params, out_extra, _slit_extra, _dcube.plain_extra)
        plan = self._dcube_plan
        return dict(spectra=dict(
            d=plan.dcube()[:, 0, :],
            m=plan.mcube()[:, 0, :] if plan.mcube() is not None else None,
            w=plan.wcube()[:, 0, :] if plan.wcube() is not None else None))
