import logging
from collections.abc import Sequence

import numpy as np

import gbkfit.math
from gbkfit.dataset.datasets import DatasetMMaps
from gbkfit.model.core import GModelSCube
from gbkfit.utils import parseutils
from . import _dcube, _detail
from .core import Observable


__all__ = [
    'MMaps'
]


_log = logging.getLogger(__name__)


# The default channel width (km/s) of the spectral axis of the cube that
# the moments are computed from, and its default size (km/s)
_SPEC_STEP = 1
_SPEC_RANGE = 1000

# The smallest dispersion (km/s) a spectral axis derived from the data
# leaves room for (see _spectral_axis_from_data)
_MIN_DISPERSION = 100


def _spectral_axis_from_data(dataset, spec_step):
    """
    The size and the centre of a spectral axis with the given channel
    width that covers the velocities of a moment map dataset: the range
    of mmap1, and three times the largest dispersion on each side. That
    is the largest value of mmap2, but at least _MIN_DISPERSION (also
    without mmap2), so that the lines of a model with a larger
    dispersion than the data's are not cut.
    """
    velocity = dataset['mmap1'].data()
    dispersion = _MIN_DISPERSION
    if 'mmap2' in dataset:
        dispersion = max(dispersion, np.nanmax(dataset['mmap2'].data()))
    margin = 3 * dispersion
    vmin = np.nanmin(velocity) - margin
    vmax = np.nanmax(velocity) + margin
    return dict(
        spec_size=int(gbkfit.math.roundu_odd((vmax - vmin) / spec_step)),
        spec_rval=float((vmin + vmax) / 2))


class MMaps(Observable):

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
            info = info | _spectral_axis_from_data(
                dataset, info.get('spec_step', _SPEC_STEP))
        return cls(**_detail.load_observable_common(
            cls, info, 2, dataset, DatasetMMaps))

    def dump(self):
        return dict(
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
            spec_rval=self.spec_rval())

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
            spec_step: int | float = _SPEC_STEP,
            spec_rval: int | float = 0
    ):
        """
        The moments are computed from a spectral cube with the spatial
        axes of the maps, and a spectral axis of spec_size channels of
        spec_step (km/s) centred on spec_rval (km/s). By default it spans
        1000 km/s. load() derives it from the moment maps of a dataset,
        unless it is given.
        """
        super().__init__(size, step, rpix, rval, rota)
        if spec_size is None:
            spec_size = int(gbkfit.math.roundu_odd(_SPEC_RANGE / spec_step))
        orders = tuple(sorted(set(orders)))
        if not orders:
            raise RuntimeError("at least one moment order is required")
        if any(order < 0 or order > 7 for order in orders):
            raise RuntimeError("moment orders must be between 0 and 7")
        if mask_cutoff is None:
            desc = parseutils.make_typed_desc(self.__class__, 'observable')
            raise RuntimeError(
                f"masking cannot be disabled for {desc}; "
                f"set the mask_cutoff to a value greater or equal to 0")
        self._mask_cutoff = mask_cutoff
        self._orders = orders
        self._spec_size = spec_size
        self._spec_step = spec_step
        self._spec_rval = spec_rval

    def keys(self):
        return tuple([f'mmap{i}' for i in self._orders])

    def orders(self):
        return self._orders

    def mask_cutoff(self):
        return self._mask_cutoff

    def spec_size(self):
        return self._spec_size

    def spec_step(self):
        return self._spec_step

    def spec_rval(self):
        return self._spec_rval

    def plan(self, driver, gmodel, instrument, scale, dtype):
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
            self.rota(), tuple(scale) + (1,), psf, lsf,
            False, None, False, dtype)
        return MMapsPlan(self, dcube, driver, gmodel, dtype)


class MMapsPlan(_detail.DCubePlanBase):

    def __init__(self, mmaps, dcube, driver, gmodel, dtype):
        super().__init__(dcube, driver, gmodel, dtype)
        self._mmaps = mmaps
        orders = mmaps.orders()
        size_all = mmaps.size() + (len(orders),)
        size_one = mmaps.size()
        self._mmaps_o = driver.mem_alloc_d(len(orders), np.int32)
        self._mmaps_d = driver.mem_alloc_d(size_all[::-1], dtype)
        self._mmaps_m = driver.mem_alloc_d(size_one[::-1], dtype)
        self._mmaps_w = driver.mem_alloc_d(size_all[::-1], dtype)
        driver.mem_copy_h2d(np.array(orders, dtype=np.int32), self._mmaps_o)
        driver.mem_fill(self._mmaps_d, np.nan)
        driver.mem_fill(self._mmaps_m, 0)
        driver.mem_fill(self._mmaps_w, 1)
        self._backend = driver.native_class('DModel', dtype)()

    def evaluate(self, params, out_extra):
        self._evaluate_cube(
            params, out_extra, _dcube.cube_extra, _dcube.cube_extra)
        # The moment maps, one mask shared by all of them, and the weight
        # map of each moment
        plan = self._dcube_plan
        self._backend.mmaps_moments(
            self._dcube.step(),
            self._dcube.zero(),
            plan.dcube(),
            plan.wcube(),
            self._mmaps.mask_cutoff(),
            self._mmaps_o,
            self._mmaps_d,
            self._mmaps_m,
            self._mmaps_w)
        return {
            key: dict(
                d=self._mmaps_d[i, :, :],
                m=self._mmaps_m,
                w=self._mmaps_w[i, :, :])
            for i, key in enumerate(self._mmaps.keys())}
