"""
The observation of moments of the spectra in regions of the sky (e.g.
Voronoi bins, fibres); those of the stars on orbits whose weights are
solved are in _orbit_moments.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from gbkfit.config import check_options
from gbkfit.dataset import Data, Dataset, DatasetRegionMoments, Regions
from gbkfit.driver import Driver
from gbkfit.grid import GridData, SpectralAxis
from gbkfit.instrument import Instrument
from gbkfit.model import Band, DataUnit, LightUnit, LineEmission
from gbkfit.moments import (
    FluxFloor, Method, MomentsPlan, check_moment_items, check_moment_orders,
    moment_item, split_moments)
from ..depth import Depth
from ..foreground import Foreground
from ..plan import ModelData, ObservationPlan
from ..weights import Weights
from . import _model_cube, _moments
from ._data_units import flux_data_unit
from ._options import Rota, Rpix, Rval, Size, Step
from ._orbit_moments import OrbitMomentsPlan, check_orbits
from ._regions import (
    ObservationRegions, RegionsPlan, spatial_options_from_regions)
from .base import DType, SkyOptions

if TYPE_CHECKING:
    import scipy.sparse

    from gbkfit.model import Scene


__all__ = [
    'ObservationRegionMoments'
]


class ObservationRegionMoments(ObservationRegions):
    """
    Moments of the spectra in regions of the sky (e.g. fibres or Voronoi
    bins): the moments of the summed spectrum of each region. The spectra
    are the sum of the cube of the model, seen through the instrument, in
    each region (see Regions), as the moments of binned data are those of
    their summed spectra.

    The spatial axes of the cube are those of the regions if they are on a
    grid (bins), or given (apertures); its spectral axis is as that of
    ObservationPixelMoments. Its options are those below, then those of
    every observation on the sky (see ObservationSky). Its data give its
    regions and its orders, and regions on a grid give the spatial grid
    (given too, they serve only without data); without data, regions are
    required.

    E.g. the kinematics of the stars of a spheroid (lines ['stars'] of its
    observation, see model.StarsEmission) in the Voronoi bins of pPXF.
    Those are of the distribution of the velocities of the stars itself
    (the fit of the spectra took the LSF out), seen through the PSF: the
    instrument of their observation has a PSF and no LSF. The velocities
    of pPXF are c ln(1 + z), the convention 'log' (without a rest: the
    light of stars has none). Models that predict the mean and the
    dispersion of the stars, but not the shape of their distribution
    (e.g. Jeans kinematics, see model.SpheroidKinematics), are fitted to
    the mean and the dispersion of the series of pPXF (see
    gbkfit.moments.gauss_hermite_moments), of the method 'moments'
    (orders 1 and 2); models of its shape, to its V, sigma, h3 and h4,
    of the method 'gauss_hermite' (orders 1 to 4).

    Parameters
    ----------
    spectral : SpectralAxis
        As that of ObservationPixelMoments.
    regions : Regions, optional
        The regions.
    size, step, rpix, rval, rota : optional
        The spatial grid of the cube (see make_grid), for
        regions on the sky (apertures): size is then required. Regions on
        a grid (bins) give it, and it must not be given.
    flux_floor, orders, method : optional
        As those of ObservationPixelMoments: the regions whose moment 0
        is not above flux_floor times the largest of the regions are
        masked.

    Raises
    ------
    ValueError
        If the spatial grid is given for regions on a grid, or not for
        regions on the sky, the options of the moments are not valid (see
        ObservationPixelMoments), the regions take valid pixels of each
        channel, or an option of every observation on the sky is not
        valid (see ObservationSky).
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

    defaults_without_data = dict(orders=(0, 1, 2))

    @classmethod
    def _options_from_data(
            cls, dataset: DatasetRegionMoments, options: dict[str, Any]
    ) -> dict[str, Any]:
        # (the regions, which give the spatial grid of regions on a grid,
        # the orders, and the spectral axis of the cube, as that of
        # ObservationPixelMoments)
        spectral = options['spectral']
        covered = {} if spectral.size() is not None \
            and spectral.rval() is not None else dict(
                spectral=_moments.spectral_axis(spectral, dataset))
        return dict(regions=dataset.regions(), orders=dataset.orders()) \
            | covered

    def _dump_options(
            self, prefix: str, dump_path: bool, overwrite: bool
    ) -> dict[str, Any]:
        return dict(orders=self._orders) | self._dump_spatial_grid() | dict(
            spectral=self._spectral.dump(),
            flux_floor=self._flux_floor,
            method=self._method)

    def _kind_options(self) -> dict[str, Any]:
        return dict(
            regions=self._regions, orders=self._orders,
            spectral=self._spectral, flux_floor=self._flux_floor,
            method=self._method) | self._dump_spatial_grid()

    def __init__(
            self,
            spectral: SpectralAxis,
            regions: Regions | None = None,
            size: Size | None = None,
            step: Step | None = None,
            rpix: Rpix | None = None,
            rval: Rval | None = None,
            rota: Rota | None = None,
            flux_floor: FluxFloor = 1e-6,
            orders: Sequence[int] | None = None,
            method: Method = 'moments',
            foreground: Foreground | None = None,
            instrument: Instrument | None = None,
            galaxies: Sequence[str] | None = None,
            components: Sequence[str] | None = None,
            lines: Sequence[str] | None = None,
            band: str | Band | None = None,
            pixel_weights: Weights | None = None,
            flux_scale: str | None = None,
            offsets: Mapping[str, bool] | None = None,
            dataset: Dataset | None = None,
            oversampling: Sequence[int] | None = None,
            depth: Depth | None = None,
            dtype: DType | None = None,
            fft_dtype: DType = 'float64',
            name: str | None = None,
            unit: str | None = None
    ):
        check_options(ObservationRegionMoments.__init__, locals())
        options = self._options(dataset, dict(
            spectral=spectral, regions=regions, size=size, step=step,
            rpix=rpix, rval=rval, rota=rota, orders=orders))
        self._orders = check_moment_orders(options.pop('orders'), method)
        self._spectral = _moments.spectral_axis(options.pop('spectral'))
        self._flux_floor = flux_floor
        self._method = method
        grid = self._regions_grid(**options)
        if self._regions.valid() is not None:
            raise ValueError(
                "the moments of regions are of whole spectra: their regions "
                "take no valid pixels of each channel (those of the spectra "
                "of regions)")
        super().__init__(
            grid, foreground, instrument, galaxies, components, lines, band,
            pixel_weights, flux_scale, offsets, dataset, oversampling, depth,
            dtype, fft_dtype, name, unit)

    def orders(self) -> tuple[int, ...]:
        """Return the orders of the moments, sorted."""
        return self._orders

    def flux_floor(self) -> float:
        """
        Return the fraction of the largest moment 0 of the regions that the
        moment 0 of a region must be above to have moments.
        """
        return self._flux_floor

    def method(self) -> str:
        """
        Return how the moments are measured (see
        gbkfit.moments.MOMENT_METHODS).
        """
        return self._method

    def spectral(self) -> SpectralAxis:
        """
        Return the spectral axis of the cube the moments are measured from.
        """
        return self._spectral

    def keys(self) -> tuple[str, ...]:
        return tuple(moment_item(i, self._method) for i in self._orders)

    def make_dataset(self, items: Mapping[str, Data]) -> DatasetRegionMoments:
        moments, hermite = split_moments(items)
        return DatasetRegionMoments(moments, self.regions(), hermite)

    def _data_unit(
            self,
            dataset: Dataset | None,
            light: LightUnit | None
    ) -> DataUnit | None:
        return flux_data_unit(
            dataset, 'moment0' if 0 in self._orders else None, True, light)

    def light_items(self) -> tuple[str, ...]:
        return ('moment0',) if 0 in self._orders else ()

    def model_can_have_missing_values(self) -> bool:
        # (moment 0, the flux, always has a value)
        return any(order >= 1 for order in self._orders)

    def _check_options(self, options: SkyOptions) -> None:
        super()._check_options(options)
        _moments.warn_unmasked_noise(self._flux_floor, options.instrument)

    def _check_scene(self, scene: 'Scene') -> None:
        options = self._sky_options
        _moments.check_one_line(scene)
        if scene.orbit_emitters():
            check_orbits(self, scene, options)

    def check_dataset(self, dataset: Dataset) -> None:
        check_moment_items(dataset, self._method)
        super().check_dataset(dataset)

    def _output(self, data: np.ndarray) -> GridData | np.ndarray:
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

    def _plan(self, driver: Driver, scene: 'Scene') -> ObservationPlan:
        options = self._sky_options
        if scene.orbit_emitters():
            return OrbitMomentsPlan(self, driver, scene, options)
        model_cube = _moments.spectra_model_cube(self, options)
        return RegionMomentsPlan(
            self, self._flux_weights, model_cube, driver, scene, options)


class RegionMomentsPlan(RegionsPlan):

    def __init__(
            self,
            moments: ObservationRegionMoments,
            weights: 'scipy.sparse.csr_array',
            model_cube: _model_cube.ModelCube,
            driver: Driver,
            scene: 'Scene',
            options: SkyOptions
    ):
        super().__init__(weights, model_cube, driver, scene, options)
        nchannels, nregions = self._spectra.shape
        # The spectra of the regions, as a cube of one row of regions
        self._spectra_cube = self._spectra.reshape(nchannels, 1, nregions)
        self._moments = MomentsPlan(
            driver, (nregions, 1), moments.orders(), moments.flux_floor(),
            moments.method(), options.dtype)

    def _measure(self, draw: bool = False) -> ModelData:
        # (of the model, or of a mock draw: data, see _measure_draw)
        self._sum_regions(self._model_cube_plan.cube())
        grid = self._model_cube.grid()
        measure = self._moments.measure if draw else self._moments.evaluate
        return _moments.model_data(measure(
            grid.coords.step, grid.zero(), self._spectra_cube, None),
            regions=True)

    def _measure_draw(self) -> ModelData:
        return self._measure(draw=True)
