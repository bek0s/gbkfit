from collections.abc import Sequence

from gbkfit.dataset.datasets import DatasetImage
from gbkfit.model.core import GModelImage
from gbkfit.utils import fitsutils
from . import _dcube, _detail
from .core import Observable


__all__ = [
    'Image'
]


def _image_extra(data, grid):
    """An extra output on a grid of DCube, as an image (its one channel)."""
    return fitsutils.GridData(data[0], grid.spatial().coords, None)


class Image(Observable):

    # The form of the data this observable measures
    dataset_class = DatasetImage

    # An image has no spectral axis
    _spectral_axis = None

    @staticmethod
    def type():
        return 'image'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, GModelImage)

    @classmethod
    def load(cls, info, dataset=None):
        return cls(**_detail.load_observable_common(
            cls, info, 2, dataset, DatasetImage))

    def dump(self, data=None, prefix='', dump_path=True, overwrite=False):
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
            step: Sequence[int | float] = (1, 1),
            rpix: Sequence[int | float] | None = None,
            rval: Sequence[int | float] = (0, 0),
            rota: float = 0,
            mask_cutoff: int | float | None = None,
            mask_apply: bool = False
    ):
        super().__init__(size, step, rpix, rval, rota)
        self._mask_cutoff = mask_cutoff
        self._mask_apply = mask_apply

    def keys(self):
        return ['image']

    def plan(
            self, driver, gmodel, foreground, instrument, scale, dtype,
            selection):
        if instrument.lsf() is not None:
            raise RuntimeError("an image has no spectral axis for an lsf")
        # The cube of an image has one channel
        dcube = _dcube.DCube(
            self.size() + (1,), self.step() + (0,), self.rpix() + (0,),
            self.rval() + (0,), self.rota(), None, tuple(scale) + (1,),
            instrument.primary_beam(), instrument.psf(), None, False,
            self._mask_cutoff, self._mask_apply, dtype)
        return ImagePlan(dcube, driver, gmodel, foreground, dtype,
            selection)


class ImagePlan(_detail.DCubePlanBase):

    def _gmodel_grid(self):
        # Image gmodels are evaluated on the x and y axes
        return self._dcube_plan.scratch_grid().spatial()

    def evaluate(self, params, out_extra):
        self._evaluate_cube(params, out_extra, _image_extra, _image_extra)
        plan = self._dcube_plan
        return dict(image=dict(
            d=plan.dcube()[0, :, :],
            m=plan.mcube()[0, :, :] if plan.mcube() is not None else None,
            w=plan.wcube()[0, :, :] if plan.wcube() is not None else None))
