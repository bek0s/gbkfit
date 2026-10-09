
import logging
from collections.abc import Sequence

import numpy as np

import gbkfit.math
from gbkfit.dataset.datasets import DatasetMMaps
from gbkfit.model.core import DModel, GModelSCube
from gbkfit.psflsf import LSF, PSF, lsf_parser, psf_parser
from gbkfit.utils import parseutils
from . import _dcube, _detail


__all__ = [
    'DModelMMaps'
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


class DModelMMaps(DModel):

    # Moment maps have no spectral axis
    _spectral_axis = None

    @staticmethod
    def type():
        return 'mmaps'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, GModelSCube)

    @classmethod
    def load(cls, info, *args, **kwargs):
        dataset = kwargs.get('dataset')
        # Without a spectral axis, cover the velocities of the data
        spectral_options = ('spec_size', 'spec_rval')
        if (dataset is not None and 'mmap1' in dataset
                and all(info.get(key) is None for key in spectral_options)):
            info.update(_spectral_axis_from_data(
                dataset, info.get('spec_step', _SPEC_STEP)))
        opts = _detail.load_dmodel_common(
            cls, info, 2, True, True, dataset, DatasetMMaps)
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            size=self.size(),
            step=self.step(),
            rpix=self.rpix(),
            rval=self.rval(),
            rota=self.rota(),
            scale=self.scale(),
            psf=psf_parser.dump(self.psf()),
            lsf=lsf_parser.dump(self.lsf()),
            mask_cutoff=self._mask_cutoff,
            orders=self.orders(),
            spec_size=self.spec_size(),
            spec_step=self.spec_step(),
            spec_rval=self.spec_rval(),
            dtype=self.dtype().name)

    def __init__(
            self,
            size: Sequence[int],
            step: Sequence[int | float] = (1, 1),
            rpix: Sequence[int | float] | None = None,
            rval: Sequence[int | float] = (0, 0),
            rota: int | float = 0,
            scale: Sequence[int] = (1, 1),
            psf: PSF | None = None,
            lsf: LSF | None = None,
            mask_cutoff: int | float = 1e-6,
            orders: Sequence[int] = (0, 1, 2),
            spec_size: int | None = None,
            spec_step: int | float = _SPEC_STEP,
            spec_rval: int | float = 0,
            dtype: str = 'float32'
    ):
        """
        The moments are computed from a spectral cube with the spatial
        axes of the maps, and a spectral axis of spec_size channels of
        spec_step (km/s) centred on spec_rval (km/s). By default it spans
        1000 km/s. load() derives it from the moment maps of a dataset,
        unless it is given.
        """
        super().__init__()
        if rpix is None:
            rpix = tuple((np.array(size) / 2 - 0.5).tolist())
        if spec_size is None:
            spec_size = int(gbkfit.math.roundu_odd(_SPEC_RANGE / spec_step))
        size = tuple(size) + (spec_size,)
        step = tuple(step) + (spec_step,)
        rpix = tuple(rpix) + (spec_size / 2 - 0.5,)
        rval = tuple(rval) + (spec_rval,)
        scale = tuple(scale) + (1,)
        orders = tuple(sorted(set(orders)))
        dtype = np.dtype(dtype)
        if not orders:
            raise RuntimeError("at least one moment order is required")
        if any(order < 0 or order > 7 for order in orders):
            raise RuntimeError("moment orders must be between 0 and 7")
        if mask_cutoff is None:
            desc = parseutils.make_typed_desc(self.__class__, 'dmodel')
            raise RuntimeError(
                f"masking cannot be disabled for {desc}; "
                f"set the mask_cutoff to a value greater or equal to 0")
        if (psf or lsf) and mask_cutoff == 0:
            _log.warning(
                "mask_cutoff is 0, but a psf or lsf is given: the fft-based "
                "convolution leaves noise in the faint parts of the model, "
                "whose moments can give artefacts in the maps; a "
                "mask_cutoff greater than 0 is highly recommended")
        self._orders = orders
        self._dcube = _dcube.DCube(
            size, step, rpix, rval, rota, scale, psf, lsf,
            # Disable DCube masking. We deal with it in this class.
            False, None, False, dtype)
        self._mmaps_o = None
        self._mmaps_d = None
        self._mmaps_m = None
        self._mmaps_w = None
        self._mask_cutoff = mask_cutoff

    def keys(self):
        return tuple([f'mmap{i}' for i in self._orders])

    def size(self):
        return self._dcube.size()[:2]

    def step(self):
        return self._dcube.step()[:2]

    def zero(self):
        return self._dcube.zero()[:2]

    def rpix(self):
        return self._dcube.rpix()[:2]

    def rval(self):
        return self._dcube.rval()[:2]

    def rota(self):
        return self._dcube.rota()

    def scale(self):
        return self._dcube.scale()[:2]

    def orders(self):
        return self._orders

    def spec_size(self):
        return self._dcube.size()[2]

    def spec_step(self):
        return self._dcube.step()[2]

    def spec_rval(self):
        return self._dcube.rval()[2]

    def psf(self):
        return self._dcube.psf()

    def lsf(self):
        return self._dcube.lsf()

    def dtype(self):
        return self._dcube.dtype()

    def _prepare_impl(self, gmodel):
        driver = self._driver
        dtype = self.dtype()
        orders = self.orders()
        # Calculate data sizes
        mmaps_size_all = self.size() + (len(orders),)
        mmaps_size_one = self.size()
        # Allocate memory
        self._mmaps_o = driver.mem_alloc_d(len(orders), np.int32)
        self._mmaps_d = driver.mem_alloc_d(mmaps_size_all[::-1], dtype)
        self._mmaps_m = driver.mem_alloc_d(mmaps_size_one[::-1], dtype)
        self._mmaps_w = driver.mem_alloc_d(mmaps_size_all[::-1], dtype)
        # Initialize memory
        driver.mem_copy_h2d(np.array(orders, dtype=np.int32), self._mmaps_o)
        driver.mem_fill(self._mmaps_d, np.nan)
        driver.mem_fill(self._mmaps_m, 0)
        driver.mem_fill(self._mmaps_w, 1)
        # Prepare dcube
        self._dcube.prepare(driver, gmodel.has_weights())
        # Create backend
        self._backend = driver.native_class('DModel', dtype)()

    def _evaluate_impl(self, params, out_dmodel_extra, out_gmodel_extra):
        driver = self._driver
        gmodel = self._gmodel
        dcube = self._dcube
        backend = self._backend
        # The gmodel adds to the data cube, so clear it
        driver.mem_fill(dcube.scratch_dcube(), 0)
        # Evaluate gmodel on DModel's arrays
        gmodel.evaluate_scube(
            driver, params,
            dcube.scratch_dcube(),
            dcube.scratch_wcube(),
            dcube.scratch_size(),
            dcube.scratch_step(),
            dcube.scratch_zero(),
            dcube.rota(),
            dcube.dtype(),
            out_gmodel_extra)
        # Evaluate gmodel on DCube's arrays
        dcube.evaluate(out_dmodel_extra)
        # Extract moment maps from DCube's arrays
        # Also evaluate one mask map and one weight map
        backend.mmaps_moments(
            dcube.step(),
            dcube.zero(),
            dcube.dcube(),
            dcube.wcube(),
            self._mask_cutoff,
            self._mmaps_o,
            self._mmaps_d,
            self._mmaps_m,
            self._mmaps_w)
        # Model evaluation complete
        # Return data, mask, and weight arrays
        # The data and weight maps are different for each moment
        # The same mask map is shared across all moments
        out = dict()
        for i, key in enumerate(self.keys()):
            out[key] = dict(
                d=self._mmaps_d[i, :, :],
                m=self._mmaps_m,
                w=self._mmaps_w[i, :, :])
        return out
