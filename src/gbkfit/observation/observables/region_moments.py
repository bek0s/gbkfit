"""
The observable of moments of the spectra in regions of the sky (e.g.
Voronoi bins, fibres); those of the stars on orbits whose weights are
solved are in _orbit_moments.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from gbkfit.config import check_options
from gbkfit.dataset import Data, Dataset, DatasetRegionMoments
from gbkfit.driver import Driver
from gbkfit.grid import GridData, SpectralAxis
from gbkfit.instrument import Instrument
from gbkfit.model import DataUnit, LightUnit, LineEmission
from gbkfit.region import Regions
from . import _model_cube, _moments
from ._data_units import flux_data_unit
from ._moments import FluxFloor, Method
from ._options import Rota, Rpix, Rval, Size, Step
from ._orbit_moments import OrbitMomentsPlan, check_orbits
from ._regions import (
    RegionsObservable, RegionsObservablePlan, spatial_options_from_regions)
from .base import ModelData, ObservablePlan

if TYPE_CHECKING:
    import scipy.sparse

    from gbkfit.model import Scene
    from ..line_of_sight import LineOfSight
    from ..foreground import Foreground
    from ..weights import Weights


__all__ = [
    'RegionMoments'
]


class RegionMoments(RegionsObservable):
    """
    Moments of the spectra in regions of the sky (e.g. fibres or Voronoi
    bins): the moments of the summed spectrum of each region. The spectra
    are the sum of the cube of the model, seen through the instrument, in
    each region (see Regions), as the moments of binned data are those of
    their summed spectra.

    The spatial axes of the cube are those of the regions if they are on
    a grid (bins), or given (apertures); its spectral axis is as that of
    PixelMoments. Its configuration has the options below, but the
    regions and the orders when it has data, which give them, and the
    spatial grid of regions on a grid (see from_data).

    E.g. the kinematics of the stars of a spheroid (lines ['stars'] of its
    observation, see model.StarsEmission) in the Voronoi bins of pPXF.
    Those are of the distribution of the velocities of the stars itself
    (the fit of the spectra took the LSF out), seen through the PSF: the
    instrument of their observation has a PSF and no LSF. The velocities
    of pPXF are c ln(1 + z), the convention 'log' (without a rest: the
    light of stars has none). Models that predict the
    mean and the dispersion of the stars, but not the shape of their
    distribution (e.g. Jeans kinematics, see model.SpheroidKinematics),
    are fitted to the mean and the dispersion of the series of pPXF (see
    prep.gauss_hermite_moments), of the method 'moments' (orders 1 and
    2); models of its shape, to its V, sigma, h3 and h4, of the method
    'gauss_hermite' (orders 1 to 4).

    Parameters
    ----------
    regions : Regions
        The regions.
    spectral : SpectralAxis
        As that of PixelMoments.
    size, step, rpix, rval, rota : optional
        The spatial grid of the cube (see make_grid), for
        regions on the sky (apertures): size is then required. Regions on
        a grid (bins) give it, and it must not be given.
    flux_floor, orders, method : optional
        As those of PixelMoments: the regions whose moment 0 is not above
        flux_floor times the largest of the regions are masked.

    Raises
    ------
    ValueError
        If the spatial grid is given for regions on a grid, or not for
        regions on the sky, or the options of the moments are not valid
        (see PixelMoments).
    """

    dataset_class = DatasetRegionMoments

    emission = LineEmission
    orbit_stars = 'moments'

    # The moments have no spectral axis
    spectral_axis = None

    @staticmethod
    def type() -> str:
        return 'region_moments'

    @staticmethod
    def type_title() -> str:
        return 'Region moments (bins, fibres)'

    @classmethod
    def options_from_data(cls, dataset: Dataset) -> tuple[str, ...]:
        # The regions and the orders, and the spatial grid of regions on a
        # grid
        return (
            ('regions', 'orders')
            + spatial_options_from_regions(dataset.regions()))

    @classmethod
    def from_data(
            cls,
            dataset: DatasetRegionMoments,
            spectral: SpectralAxis,
            size: Size | None = None,
            step: Step | None = None,
            rpix: Rpix | None = None,
            rval: Rval | None = None,
            rota: Rota | None = None,
            flux_floor: FluxFloor = 1e-6,
            method: Method = 'moments'
    ) -> 'RegionMoments':
        """
        Make the observable of the moments of regions, of their regions
        and orders.

        Parameters
        ----------
        dataset : DatasetRegionMoments
            The moments.
        spectral : SpectralAxis
            As that of PixelMoments.from_data.
        size, step, rpix, rval, rota : optional
            As those of RegionMoments: the spatial grid of the cube, for
            regions on the sky.
        flux_floor, method : optional
            As those of PixelMoments.from_data.

        Returns
        -------
        RegionMoments
            The observable.

        Raises
        ------
        ValueError
            If the dataset is not of moments in regions, or the options
            are not valid.
        """
        check_options(RegionMoments.from_data, locals())
        return cls(
            dataset.regions(), _moments.spectral_axis(spectral, dataset),
            size, step, rpix, rval, rota, flux_floor=flux_floor,
            orders=dataset.orders(), method=method)

    def _dump_options(
            self, prefix: str, dump_path: bool, overwrite: bool
    ) -> dict[str, Any]:
        return dict(orders=self._orders) | self._dump_spatial_grid() | dict(
            spectral=self._spectral.dump(),
            flux_floor=self._flux_floor,
            method=self._method)

    def __init__(
            self,
            regions: Regions,
            spectral: SpectralAxis,
            size: Size | None = None,
            step: Step | None = None,
            rpix: Rpix | None = None,
            rval: Rval | None = None,
            rota: Rota | None = None,
            flux_floor: FluxFloor = 1e-6,
            orders: Sequence[int] = (0, 1, 2),
            method: Method = 'moments'
    ):
        check_options(RegionMoments.__init__, locals())
        super().__init__(regions, size, step, rpix, rval, rota)
        self._orders = _moments.check_moment_orders(orders, method)
        self._spectral = _moments.spectral_axis(spectral)
        self._flux_floor = flux_floor
        self._method = method
        if regions.valid() is not None:
            raise ValueError(
                "the moments of regions are of whole spectra: their regions "
                "take no valid pixels of each channel (those of the spectra "
                "of regions)")

    def orders(self) -> tuple[int, ...]:
        return self._orders

    def flux_floor(self) -> float:
        return self._flux_floor

    def method(self) -> str:
        return self._method

    def spectral(self) -> SpectralAxis:
        """
        Return the spectral axis of the cube the moments are measured from.
        """
        return self._spectral

    def keys(self) -> tuple[str, ...]:
        return tuple(
            _moments.moment_item(i, self._method) for i in self._orders)

    def make_dataset(self, items: Mapping[str, Data]) -> DatasetRegionMoments:
        moments, hermite = _moments.moment_maps(
            items, self._orders, self._method)
        return DatasetRegionMoments(moments, self.regions(), hermite)

    def data_unit(
            self,
            dataset: Dataset | None,
            light: LightUnit | None = None
    ) -> DataUnit | None:
        return flux_data_unit(
            dataset, 'moment0' if 0 in self._orders else None, True, light)

    def light_items(self) -> tuple[str, ...]:
        return ('moment0',) if 0 in self._orders else ()

    def model_can_have_missing_values(self) -> bool:
        # (moment 0, the flux, always has a value)
        return any(order >= 1 for order in self._orders)

    def check_observation(
            self,
            instrument: Instrument,
            oversampling: Sequence[int],
            pixel_weights: 'Weights | None'
    ) -> None:
        super().check_observation(instrument, oversampling, pixel_weights)
        _moments.warn_unmasked_noise(self._flux_floor, instrument)

    def check_scene(
            self,
            scene: 'Scene',
            foreground: 'Foreground',
            instrument: Instrument,
            flux_scale: str | None
    ) -> None:
        _moments.check_one_line(scene)
        if scene.orbit_emitters():
            check_orbits(self, scene, foreground, instrument, flux_scale)

    def check_dataset(self, dataset: Dataset) -> None:
        _moments.check_moment_items(dataset, self._method)
        super().check_dataset(dataset)

    def output(self, data: np.ndarray) -> GridData | np.ndarray:
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
        return GridData(image, grid.coords, None)

    def plan(
            self,
            driver: Driver,
            scene: 'Scene',
            line_of_sight: 'LineOfSight',
            foreground: 'Foreground',
            instrument: Instrument,
            oversampling: Sequence[int],
            dtype: np.dtype,
            fft_dtype: np.dtype,
            pixel_weights: 'Weights | None',
            flux_scale: str | None
    ) -> ObservablePlan:
        if scene.orbit_emitters():
            return OrbitMomentsPlan(
                self, scene, instrument, oversampling, driver, dtype)
        model_cube = _moments.spectra_model_cube(
            self, oversampling, instrument, dtype, fft_dtype)
        return RegionMomentsPlan(
            self, self._flux_weights, model_cube, driver, scene,
            line_of_sight, foreground, dtype, flux_scale)


class RegionMomentsPlan(RegionsObservablePlan):

    def __init__(
            self,
            moments: RegionMoments,
            weights: 'scipy.sparse.csr_array',
            model_cube: _model_cube.ModelCube,
            driver: Driver,
            scene: 'Scene',
            line_of_sight: 'LineOfSight',
            foreground: 'Foreground',
            dtype: np.dtype,
            flux_scale: str | None
    ):
        super().__init__(
            weights, model_cube, driver, scene, line_of_sight, foreground,
            dtype, flux_scale)
        nchannels, nregions = self._spectra.shape
        # The spectra of the regions, as a cube of one row of regions
        self._spectra_cube = self._spectra.reshape(nchannels, 1, nregions)
        self._moments = _moments.MomentsPlan(
            driver, (nregions, 1), moments.orders(), moments.flux_floor(),
            moments.method(), dtype)

    def _measure(self) -> ModelData:
        self._sum_regions(self._model_cube_plan.cube())
        grid = self._model_cube.grid()
        moments = self._moments.evaluate(
            grid.coords.step, grid.zero(), self._spectra_cube, None)
        # The moments of the regions are vectors
        return {
            key: dict(d=value['d'][0], m=value['m'][0], w=value['w'][0])
            for key, value in moments.items()}
