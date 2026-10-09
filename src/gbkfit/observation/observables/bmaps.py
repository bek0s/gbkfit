import logging
from collections.abc import Sequence
from numbers import Real

import astropy.units
import numpy as np

from gbkfit.dataset.datasets import DatasetBMaps
from gbkfit.dataset.regions import Regions, regions_parser
from gbkfit.model.core import GModelSCube
from gbkfit.utils import fitsutils, parseutils
from . import _dcube, _detail, _moments
from ._regions import RegionSumsPlan
from .core import Observable


__all__ = [
    'BMaps'
]


_log = logging.getLogger(__name__)


class BMaps(Observable):
    """
    Moments of the spectra in regions of the sky (see Regions; e.g.
    Voronoi bins, fibres): the moments of the sum of the cube of the
    model, seen through the instrument, in each region, as the moments of
    binned data are those of their summed spectra. The spatial axes of the
    cube are those of the regions if they are on a grid (bins), or given
    (apertures); its spectral axis is as that of mmaps.
    """

    # The form of the data this observable measures
    dataset_class = DatasetBMaps

    # The moments have no spectral axis
    _spectral_axis = None

    @staticmethod
    def type():
        return 'bmaps'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, GModelSCube)

    @classmethod
    def options_from_data(cls, dataset):
        # The regions, and the spatial grid of regions on a grid
        return (
            ('regions',)
            + _detail.spatial_options_from_regions(dataset.regions()))

    @classmethod
    def load(cls, info, dataset=None):
        desc = parseutils.make_typed_desc(cls, 'observable')
        if dataset is not None:
            if not isinstance(dataset, DatasetBMaps):
                dataset_desc = parseutils.make_typed_desc(
                    dataset.__class__, 'dataset')
                raise RuntimeError(
                    f"{desc} cannot be compared with {dataset_desc}")
            _detail.require_no_options_from_data(cls, info, dataset)
            info = info | dict(regions=dataset.regions())
            # Without a spectral axis, cover the velocities of the data
            spectral_options = ('spec_size', 'spec_rval')
            if ('mmap1' in dataset
                    and all(info.get(k) is None for k in spectral_options)):
                info = info | _moments.spectral_axis_from_data(
                    dataset, info.get('spec_step', _moments.SPEC_STEP))
        else:
            parseutils.load_option_and_update_info(
                regions_parser, info, 'regions', True, False)
        parseutils.sanitize_dimensional_options(info, dict(
            size=int, step=int | float, rpix=int | float,
            rval=int | float), 2)
        return cls(**parseutils.parse_options_for_callable(
            info, desc, cls.__init__))

    def dump(self, data=None):
        info = dict(type=self.type())
        if data is None:
            info.update(regions=regions_parser.dump(self._regions))
        return info | _detail.dump_spatial_grid(self, self._regions) | dict(
            mask_cutoff=self._mask_cutoff,
            orders=self._orders,
            spec_size=self._spec_size,
            spec_step=self._spec_step,
            spec_rval=self._spec_rval,
            spec_rest=_detail.dump_rest(self._spec_rest),
            method=self._method)

    def __init__(
            self,
            regions: Regions,
            size: Sequence[int] | None = None,
            step: Sequence[Real] | None = None,
            rpix: Sequence[Real] | None = None,
            rval: Sequence[Real] | None = None,
            rota: Real | None = None,
            mask_cutoff: Real = 1e-6,
            orders: Sequence[int] = (0, 1, 2),
            spec_size: int | None = None,
            spec_step: Real = _moments.SPEC_STEP,
            spec_rval: Real = 0,
            spec_rest: str | astropy.units.Quantity | None = None,
            method: str = 'moments'
    ):
        """
        The spatial grid (size, step, rpix, rval, rota; see
        fitsutils.make_grid) is that of the regions if they have one
        (bins), and must not be given; else size is required. The
        spectral axis and the moments are as those of mmaps (see MMaps,
        also for method): the regions whose moment 0 is not above
        mask_cutoff are masked.
        """
        spatial = _detail.spatial_grid_of_regions(
            regions, size, step, rpix, rval, rota)
        coords = spatial.coords
        super().__init__(
            spatial.size, coords.step, coords.rpix, coords.rval, coords.rota)
        if spec_size is None:
            spec_size = _moments.default_spec_size(spec_step)
        self._orders = _moments.check_moment_options(
            parseutils.make_typed_desc(self.__class__, 'observable'),
            orders, mask_cutoff, method)
        self._regions = regions
        self._mask_cutoff = mask_cutoff
        self._method = method
        self._spec_size = spec_size
        self._spec_step = spec_step
        self._spec_rval = spec_rval
        self._spec_rest = fitsutils.make_rest(spec_rest)
        # The weights of the pixels in the regions (an error if apertures
        # are not inside the grid)
        self._weights = regions.weights(spatial)

    def regions(self) -> Regions:
        return self._regions

    def orders(self) -> tuple[int, ...]:
        return self._orders

    def mask_cutoff(self) -> Real:
        return self._mask_cutoff

    def method(self) -> str:
        return self._method

    def spec_size(self) -> int:
        return self._spec_size

    def spec_step(self) -> Real:
        return self._spec_step

    def spec_rval(self) -> Real:
        return self._spec_rval

    def keys(self):
        return tuple(f'mmap{i}' for i in self._orders)

    def _require_matching_coordinates(self, dataset):
        if dataset.regions() != self._regions:
            raise RuntimeError(
                "the data and the observable have different regions")

    def output(self, data):
        """
        A vector of the regions (e.g. a moment) as an output: for bins, a
        map of the value of each bin on its pixels (NaN outside the bins;
        a GridData); for apertures, the vector.
        """
        grid = self._regions.grid()
        if grid is None:
            return data
        index = self._regions.index()
        image = np.full(index.shape, np.nan, dtype=data.dtype)
        image[index >= 0] = data[index[index >= 0]]
        return fitsutils.GridData(image, grid.coords, None)

    def plan(
            self, driver, gmodel, foreground, instrument, scale, dtype,
            components):
        if gmodel.has_weights():
            raise RuntimeError(
                "bmaps does not support gmodels with weights (wtraits) yet")
        psf, lsf = instrument.psf(), instrument.lsf()
        if (psf or lsf) and self._mask_cutoff == 0:
            _log.warning(
                "mask_cutoff is 0, but a psf or lsf is given: the fft-based "
                "convolution leaves noise in the faint parts of the model, "
                "whose moments can give artefacts; a mask_cutoff greater "
                "than 0 is highly recommended")
        # The masking of DCube is disabled: every pixel of a region adds
        # to its spectrum, and the moments are masked by moment 0
        spec_size = self._spec_size
        dcube = _dcube.DCube(
            self.size() + (spec_size,),
            self.step() + (self._spec_step,),
            self.rpix() + (spec_size / 2 - 0.5,),
            self.rval() + (self._spec_rval,),
            self.rota(), self._spec_rest, tuple(scale) + (1,),
            instrument.primary_beam(), psf, lsf,
            False, None, False, dtype)
        return BMapsPlan(
            self._weights, self._orders, self._mask_cutoff, self._method,
            dcube, driver, gmodel, foreground, dtype,
            components)


class BMapsPlan(_detail.DCubePlanBase):

    def __init__(
            self, weights, orders, mask_cutoff, method, dcube, driver,
            gmodel, foreground, dtype, components):
        super().__init__(
            dcube, driver, gmodel, foreground, dtype, components)
        self._sums = RegionSumsPlan(weights, driver, dtype)
        nregions = self._sums.nregions()
        # The spectra of the regions, as a cube of one row of regions
        self._spectra = driver.mem_alloc_d(
            (dcube.size()[2], 1, nregions), dtype)
        self._moments = _moments.MomentsPlan(
            driver, (nregions, 1), orders, mask_cutoff, method, dtype)

    def evaluate(self, params, out_extra):
        self._evaluate_cube(
            params, out_extra, _dcube.cube_extra, _dcube.cube_extra)
        self._sums.evaluate(
            self._dcube_plan.dcube(), self._spectra[:, 0, :])
        moments = self._moments.evaluate(
            self._dcube.step(), self._dcube.zero(), self._spectra, None)
        # The moments of the regions are vectors
        return {
            key: dict(d=value['d'][0], m=value['m'][0], w=value['w'][0])
            for key, value in moments.items()}
