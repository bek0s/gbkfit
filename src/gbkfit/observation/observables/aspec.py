from collections.abc import Sequence
from numbers import Real

import astropy.units

from gbkfit.dataset.datasets import DatasetASpec
from gbkfit.dataset.regions import Regions, regions_parser
from gbkfit.model.core import GModelSCube
from gbkfit.utils import fitsutils, parseutils
from . import _dcube, _detail
from ._regions import RegionSumsPlan
from .core import Observable


__all__ = [
    'ASpec'
]


class ASpec(Observable):
    """
    Spectra in regions of the sky (see Regions; e.g. fibres, apertures,
    bins, or the whole field for an integrated spectrum): the sum of the
    cube of the model, seen through the instrument, in each region, in
    each channel. The spatial axes of the cube are those of the regions if
    they are on a grid (bins), or given (apertures); its spectral axis is
    that of the spectra.
    """

    # The form of the data this observable measures
    dataset_class = DatasetASpec

    # The axes of the cube of the model: x, y and the spectral axis
    _spectral_axis = 2

    @staticmethod
    def type():
        return 'aspec'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, GModelSCube)

    @classmethod
    def options_from_data(cls, dataset):
        # The regions and the spectral axis, and the spatial grid of
        # regions on a grid
        return (
            ('regions', 'spec_size', 'spec_step', 'spec_rpix', 'spec_rval',
             'spec_rest')
            + _detail.spatial_options_from_regions(dataset.regions()))

    @classmethod
    def load(cls, info, dataset=None):
        desc = parseutils.make_typed_desc(cls, 'observable')
        if dataset is not None:
            if not isinstance(dataset, DatasetASpec):
                dataset_desc = parseutils.make_typed_desc(
                    dataset.__class__, 'dataset')
                raise RuntimeError(
                    f"{desc} cannot be compared with {dataset_desc}")
            _detail.require_no_options_from_data(cls, info, dataset)
            spectral = dataset.spectral_grid()
            info.update(
                regions=dataset.regions(),
                spec_size=spectral.size[0],
                spec_step=spectral.coords.step[0],
                spec_rpix=spectral.coords.rpix[0],
                spec_rval=spectral.coords.rval[0],
                spec_rest=spectral.coords.rest)
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
            info.update(
                regions=regions_parser.dump(self._regions),
                spec_size=self.size()[2],
                spec_step=self.step()[2],
                spec_rpix=self.rpix()[2],
                spec_rval=self.rval()[2],
                spec_rest=_detail.dump_rest(self.rest()))
        return info | _detail.dump_spatial_grid(self, self._regions)

    def __init__(
            self,
            regions: Regions,
            spec_size: int,
            spec_step: Real = 1,
            spec_rpix: Real | None = None,
            spec_rval: Real = 0,
            spec_rest: str | astropy.units.Quantity | None = None,
            size: Sequence[int] | None = None,
            step: Sequence[Real] | None = None,
            rpix: Sequence[Real] | None = None,
            rval: Sequence[Real] | None = None,
            rota: Real | None = None
    ):
        """
        The spectral axis has spec_size channels of spec_step (km/s), with
        the velocity spec_rval (km/s) at the channel spec_rpix (by default
        the centre), and velocities of the rest wavelength or frequency
        spec_rest (see fitsutils.Coords), if known. The spatial grid (size, step, rpix, rval, rota; see
        fitsutils.make_grid) is that of the regions if they have one
        (bins), and must not be given; else size is required.
        """
        spatial = _detail.spatial_grid_of_regions(
            regions, size, step, rpix, rval, rota)
        if spec_rpix is None:
            spec_rpix = spec_size / 2 - 0.5
        coords = spatial.coords
        super().__init__(
            spatial.size + (spec_size,),
            coords.step + (spec_step,),
            coords.rpix + (spec_rpix,),
            coords.rval + (spec_rval,),
            coords.rota, spec_rest)
        self._regions = regions
        # The weights of the pixels in the regions (an error if apertures
        # are not inside the grid)
        self._weights = regions.weights(spatial)

    def regions(self) -> Regions:
        return self._regions

    def spectral_grid(self) -> fitsutils.Grid:
        """The grid of the spectral axis (one axis)."""
        return self._grid.spectral()

    def keys(self):
        return ['aspec']

    def _require_matching_coordinates(self, dataset):
        if dataset.regions() != self._regions:
            raise RuntimeError(
                "the data and the observable have different regions")
        if dataset.spectral_grid() != self.spectral_grid():
            raise RuntimeError(
                f"the spectra have the spectral axis "
                f"{dataset.spectral_grid()}, but the observable has "
                f"{self.spectral_grid()}")

    def output(self, data):
        return fitsutils.SpectraData(data, self.spectral_grid().coords)

    def plan(
            self, driver, gmodel, foreground, instrument, scale, dtype,
            selection):
        if gmodel.has_weights():
            raise RuntimeError(
                "aspec does not support gmodels with weights (wtraits) yet")
        # The masking of DCube is disabled: every pixel of a region adds
        # to its spectrum
        dcube = _dcube.DCube(
            self.size(), self.step(), self.rpix(), self.rval(), self.rota(),
            self.rest(), tuple(scale), instrument.primary_beam(),
            instrument.psf(), instrument.lsf(), False, None, False, dtype)
        return ASpecPlan(
            self._weights, dcube, driver, gmodel, foreground, dtype,
            selection)


class ASpecPlan(_detail.DCubePlanBase):

    def __init__(
            self, weights, dcube, driver, gmodel, foreground, dtype,
            selection):
        super().__init__(
            dcube, driver, gmodel, foreground, dtype, selection)
        self._sums = RegionSumsPlan(weights, driver, dtype)
        self._spectra = driver.mem_alloc_d(
            (dcube.size()[2], self._sums.nregions()), dtype)

    def evaluate(self, params, out_extra):
        self._evaluate_cube(
            params, out_extra, _dcube.cube_extra, _dcube.cube_extra)
        self._sums.evaluate(self._dcube_plan.dcube(), self._spectra)
        return dict(aspec=dict(d=self._spectra, m=None, w=None))
