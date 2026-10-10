from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np

from gbkfit.dataset import Dataset, DatasetRegionMoments
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument
from gbkfit.model.base import Model, ModelSCube, Selection
from gbkfit.region import Regions, regions_parser
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _dcube, _detail, _moments
from ._region_sums import RegionSumsPlan, flux_weights
from .base import ModelData, Observable

if TYPE_CHECKING:
    import scipy.sparse

    from ..foreground import Foreground


__all__ = [
    'RegionMoments'
]


class RegionMoments(Observable):
    """
    Moments of the spectra in regions of the sky (see Regions; e.g.
    Voronoi bins, fibres): the moments of the sum of the cube of the
    model, seen through the instrument, in each region, as the moments of
    binned data are those of their summed spectra.

    The spatial axes of the cube are those of the regions if they are on
    a grid (bins), or given (apertures); its spectral axis is as that of
    PixelMoments. Its configuration has the options below, but the
    regions and the orders when it has data, which give them, and the
    spatial grid of regions on a grid (see from_data).

    Parameters
    ----------
    regions : Regions
        The regions.
    size, step, rpix, rval, rota : optional
        The spatial grid of the cube (see gridutils.make_grid), for
        regions on the sky (apertures): size is then required. Regions on
        a grid (bins) give it, and it must not be given.
    mask_cutoff, orders, spec_size, spec_step, spec_rval, spec_rest, \
method : optional
        As those of PixelMoments: the regions whose moment 0 is not above
        mask_cutoff are masked.

    Raises
    ------
    ConfigError
        If the spatial grid is given for regions on a grid, or not for
        regions on the sky, or the options of the moments are not valid
        (see PixelMoments).
    """

    dataset_class = DatasetRegionMoments

    # The moments have no spectral axis
    spectral_axis = None

    @staticmethod
    def type() -> str:
        return 'region_moments'

    @staticmethod
    def is_compatible(model: Model) -> bool:
        return isinstance(model, ModelSCube)

    @classmethod
    def options_from_data(cls, dataset: Dataset) -> tuple[str, ...]:
        # The regions and the orders, and the spatial grid of regions on a
        # grid
        return (
            ('regions', 'orders')
            + _detail.spatial_options_from_regions(dataset.regions()))

    @classmethod
    def load(
            cls, info: dict[str, Any],
            dataset: DatasetRegionMoments | None = None
    ) -> 'RegionMoments':
        if dataset is None:
            parseutils.load_option_and_update_info(
                regions_parser, info, 'regions', required=True)
        return _detail.load_observable(cls, info, dataset, 2)

    @classmethod
    def from_data(
            cls,
            dataset: DatasetRegionMoments,
            size: Sequence[int] | None = None,
            step: Sequence[float] | None = None,
            rpix: Sequence[float] | None = None,
            rval: Sequence[float] | None = None,
            rota: float | None = None,
            mask_cutoff: float = 1e-6,
            spec_size: int | None = None,
            spec_step: float = _moments.SPEC_STEP,
            spec_rval: float | None = None,
            spec_rest: str | astropy.units.Quantity | None = None,
            method: str = 'moments'
    ) -> 'RegionMoments':
        """
        Make the observable of the moments of regions, of their regions
        and orders.

        Parameters
        ----------
        dataset : DatasetRegionMoments
            The moments.
        size, step, rpix, rval, rota : optional
            As those of RegionMoments: the spatial grid of the cube, for
            regions on the sky.
        mask_cutoff, spec_size, spec_step, spec_rval, spec_rest, method \
: optional
            As those of PixelMoments.from_data.

        Returns
        -------
        RegionMoments
            The observable.

        Raises
        ------
        ConfigError
            If the dataset is not of moments in regions, or the options
            are not valid.
        """
        cls._require_dataset_class(dataset)
        return cls(
            dataset.regions(), size, step, rpix, rval, rota,
            **_moments.spectral_axis_from_data(
                dataset, spec_step, spec_size, spec_rval),
            mask_cutoff=mask_cutoff, orders=dataset.orders(),
            spec_step=spec_step, spec_rest=spec_rest, method=method)

    def dump(
            self,
            data: DatasetRegionMoments | None = None,
            prefix: str = '',
            dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        info = dict(type=self.type())
        if data is None:
            info.update(
                regions=regions_parser.dump(
                    self._regions, prefix=prefix, dump_path=dump_path,
                    overwrite=overwrite),
                orders=self._orders)
        return info | _detail.dump_spatial_grid(self, self._regions) | dict(
            mask_cutoff=self._mask_cutoff,
            spec_size=self._spec_size,
            spec_step=self._spec_step,
            spec_rval=self._spec_rval,
            spec_rest=_detail.dump_rest(self._spec_rest),
            method=self._method)

    def __init__(
            self,
            regions: Regions,
            size: Sequence[int] | None = None,
            step: Sequence[float] | None = None,
            rpix: Sequence[float] | None = None,
            rval: Sequence[float] | None = None,
            rota: float | None = None,
            mask_cutoff: float = 1e-6,
            orders: Sequence[int] = (0, 1, 2),
            spec_size: int | None = None,
            spec_step: float = _moments.SPEC_STEP,
            spec_rval: float = 0,
            spec_rest: str | astropy.units.Quantity | None = None,
            method: str = 'moments'
    ):
        spatial = _detail.spatial_grid_of_regions(
            regions, size, step, rpix, rval, rota)
        coords = spatial.coords
        super().__init__(
            spatial.size, coords.step, coords.rpix, coords.rval, coords.rota)
        self._orders = _moments.check_moment_options(
            orders, mask_cutoff, method, spec_size, spec_step)
        if spec_size is None:
            spec_size = _moments.default_spec_size(spec_step)
        self._regions = regions
        self._mask_cutoff = mask_cutoff
        self._method = method
        self._spec_size = spec_size
        self._spec_step = spec_step
        self._spec_rval = spec_rval
        self._spec_rest = gridutils.make_rest(spec_rest)
        # The area of each pixel in the regions, so that their sums are
        # fluxes (an error if apertures are not inside the grid)
        self._weights = flux_weights(regions, spatial)

    def regions(self) -> Regions:
        """Return the regions."""
        return self._regions

    def orders(self) -> tuple[int, ...]:
        return self._orders

    def mask_cutoff(self) -> float:
        return self._mask_cutoff

    def method(self) -> str:
        return self._method

    def spec_size(self) -> int:
        return self._spec_size

    def spec_step(self) -> float:
        return self._spec_step

    def spec_rval(self) -> float:
        return self._spec_rval

    def spec_rest(self) -> astropy.units.Quantity | None:
        return self._spec_rest

    def keys(self) -> tuple[str, ...]:
        return tuple(f'moment{i}' for i in self._orders)

    def check_instrument(self, instrument: Instrument) -> None:
        _moments.warn_unmasked_noise(self._mask_cutoff, instrument)

    def _require_matching_coordinates(
            self, dataset: DatasetRegionMoments
    ) -> None:
        if dataset.regions() != self._regions:
            raise ConfigError(
                "the data and the observable have different regions")

    def output(self, data: np.ndarray) -> gridutils.GridData | np.ndarray:
        """
        Return a vector of the regions (e.g. a moment) as an output.

        Parameters
        ----------
        data : np.ndarray
            The vector.

        Returns
        -------
        GridData or np.ndarray
            For bins, a map of the value of each bin on its pixels (NaN
            outside the bins); for apertures, the vector.
        """
        grid = self._regions.grid()
        if grid is None:
            return data
        index = self._regions.index()
        image = np.full(index.shape, np.nan, dtype=data.dtype)
        image[index >= 0] = data[index[index >= 0]]
        return gridutils.GridData(image, grid.coords, None)

    def plan(
            self,
            driver: Driver,
            model: Model,
            foreground: 'Foreground',
            instrument: Instrument,
            scale: Sequence[int],
            dtype: np.dtype,
            selection: Selection
    ) -> 'RegionMomentsPlan':
        if model.has_weights():
            raise ConfigError(
                "region_moments does not support models with weights "
                "(wtraits) yet")
        dcube = _moments.spectra_dcube(self, scale, instrument, dtype)
        return RegionMomentsPlan(
            self, self._weights, dcube, driver, model, foreground, dtype,
            selection)


class RegionMomentsPlan(_detail.DCubePlanBase):

    def __init__(
            self,
            moments: RegionMoments,
            weights: 'scipy.sparse.csr_array',
            dcube: _dcube.DCube,
            driver: Driver,
            model: Model,
            foreground: 'Foreground',
            dtype: np.dtype,
            selection: Selection
    ):
        super().__init__(
            dcube, driver, model, foreground, dtype, selection)
        self._sums = RegionSumsPlan(weights, driver, dtype)
        nregions = self._sums.nregions()
        # The spectra of the regions, as a cube of one row of regions
        self._spectra = driver.mem_alloc_d(
            (dcube.size()[2], 1, nregions), dtype)
        self._moments = _moments.MomentsPlan(
            driver, (nregions, 1), moments.orders(), moments.mask_cutoff(),
            moments.method(), dtype)

    def evaluate(
            self,
            params: dict[str, float | np.ndarray],
            out_extra: dict[str, Any] | None
    ) -> ModelData:
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
