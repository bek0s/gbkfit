import logging
from collections.abc import Sequence

import astropy.units

from gbkfit.dataset.datasets import DatasetMMaps
from gbkfit.model.core import GModelSCube
from gbkfit.utils import fitsutils, parseutils
from . import _dcube, _detail, _moments
from .core import Observable


__all__ = [
    'MMaps'
]


_log = logging.getLogger(__name__)


class MMaps(Observable):

    # The form of the data this observable measures
    dataset_class = DatasetMMaps

    # Moment maps have no spectral axis
    _spectral_axis = None

    @staticmethod
    def type():
        return 'mmaps'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, GModelSCube)

    @classmethod
    def load(cls, info, dataset=None):
        # Without a spectral axis, cover the velocities of the data
        spectral_options = ('spec_size', 'spec_rval')
        if (dataset is not None and 'mmap1' in dataset
                and all(info.get(key) is None for key in spectral_options)):
            info = info | _moments.spectral_axis_from_data(
                dataset, info.get('spec_step', _moments.SPEC_STEP))
        return cls(**_detail.load_observable_common(
            cls, info, 2, dataset, DatasetMMaps))

    def dump(self, data=None):
        return _detail.without_options_from_data(self, dict(
            type=self.type(),
            size=self.size(),
            step=self.step(),
            rpix=self.rpix(),
            rval=self.rval(),
            rota=self.rota(),
            mask_cutoff=self._mask_cutoff,
            orders=self.orders(),
            spec_size=self.spec_size(),
            spec_step=self.spec_step(),
            spec_rval=self.spec_rval(),
            spec_rest=_detail.dump_rest(self._spec_rest),
            method=self._method), data)

    def __init__(
            self,
            size: Sequence[int],
            step: Sequence[int | float] = (1, 1),
            rpix: Sequence[int | float] | None = None,
            rval: Sequence[int | float] = (0, 0),
            rota: int | float = 0,
            mask_cutoff: int | float = 1e-6,
            orders: Sequence[int] = (0, 1, 2),
            spec_size: int | None = None,
            spec_step: int | float = _moments.SPEC_STEP,
            spec_rval: int | float = 0,
            spec_rest: str | astropy.units.Quantity | None = None,
            method: str = 'moments'
    ):
        """
        The moments are computed from a spectral cube with the spatial
        axes of the maps, and a spectral axis of spec_size channels of
        spec_step (km/s) centred on spec_rval (km/s). By default it spans
        1000 km/s. load() derives it from the moment maps of a dataset,
        unless it is given. spec_rest is the rest wavelength or frequency of
        its velocities (see fitsutils.Coords), if known. method is how the
        maps are measured from the spectra (see _moments.METHODS): their
        moments, or a Gaussian fitted to each, as the maps of the data
        were.
        """
        super().__init__(size, step, rpix, rval, rota)
        if spec_size is None:
            spec_size = _moments.default_spec_size(spec_step)
        orders = _moments.check_moment_options(
            parseutils.make_typed_desc(self.__class__, 'observable'),
            orders, mask_cutoff, method)
        self._mask_cutoff = mask_cutoff
        self._method = method
        self._orders = orders
        self._spec_size = spec_size
        self._spec_step = spec_step
        self._spec_rval = spec_rval
        self._spec_rest = fitsutils.make_rest(spec_rest)

    def keys(self):
        return tuple([f'mmap{i}' for i in self._orders])

    def orders(self):
        return self._orders

    def mask_cutoff(self):
        return self._mask_cutoff

    def method(self):
        return self._method

    def spec_size(self):
        return self._spec_size

    def spec_step(self):
        return self._spec_step

    def spec_rval(self):
        return self._spec_rval

    def plan(
            self, driver, gmodel, foreground, instrument, scale, dtype,
            selection):
        if self._method == 'gaussian_fit' and gmodel.has_weights():
            raise RuntimeError(
                "the method gaussian_fit does not support gmodels with "
                "weights (wtraits) yet")
        psf, lsf = instrument.psf(), instrument.lsf()
        if (psf or lsf) and self._mask_cutoff == 0:
            _log.warning(
                "mask_cutoff is 0, but a psf or lsf is given: the fft-based "
                "convolution leaves noise in the faint parts of the model, "
                "whose moments can give artefacts in the maps; a "
                "mask_cutoff greater than 0 is highly recommended")
        # The moments are computed from a cube with the spatial axes of
        # the maps; the masking of DCube is disabled, the maps are masked
        # by the moments
        spec_size = self._spec_size
        dcube = _dcube.DCube(
            self.size() + (spec_size,),
            self.step() + (self._spec_step,),
            self.rpix() + (spec_size / 2 - 0.5,),
            self.rval() + (self._spec_rval,),
            self.rota(), self._spec_rest, tuple(scale) + (1,),
            instrument.primary_beam(), psf, lsf,
            False, None, False, dtype)
        return MMapsPlan(self, dcube, driver, gmodel, foreground, dtype,
            selection)


class MMapsPlan(_detail.DCubePlanBase):

    def __init__(
            self, mmaps, dcube, driver, gmodel, foreground, dtype,
            selection):
        super().__init__(
            dcube, driver, gmodel, foreground, dtype, selection)
        self._moments = _moments.MomentsPlan(
            driver, mmaps.size(), mmaps.orders(), mmaps.mask_cutoff(),
            mmaps.method(), dtype)

    def evaluate(self, params, out_extra):
        self._evaluate_cube(
            params, out_extra, _dcube.cube_extra, _dcube.cube_extra)
        # The moment maps, one mask shared by all of them, and the weight
        # map of each moment
        plan = self._dcube_plan
        return self._moments.evaluate(
            self._dcube.step(), self._dcube.zero(), plan.dcube(),
            plan.wcube())
