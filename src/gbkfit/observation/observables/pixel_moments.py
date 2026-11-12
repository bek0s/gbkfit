"""
The observable of moments of the spectra on a grid of pixels: moment
maps.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from gbkfit.config import check_options
from gbkfit.dataset import Data, Dataset, DatasetPixelMoments
from gbkfit.driver import Driver
from gbkfit.grid import SpectralAxis, grid_options
from gbkfit.model import DataUnit, LightUnit, LineEmission
from . import _model_cube, _moments
from ._cube_plan import CubeObservablePlan
from ._data_units import data_unit
from ._moments import FluxFloor, Method
from ._options import (
    MedianSize, Rota, Rpix, Rval, Size, Step, check_median_size)
from .base import ModelData, Observable, ObservationSetup

if TYPE_CHECKING:
    from gbkfit.model import Scene


__all__ = [
    'PixelMoments'
]


class PixelMoments(Observable):
    """
    Moments of the spectra on a grid of pixels: moment maps of a spectral
    cube of the model, seen through the instrument. The maps are of some
    orders (0 to 7), as the data items moment0 to moment7, but h3 and h4
    of the method gauss_hermite (see _moments.moment_item).

    The moments are measured from a spectral cube with the spatial axes
    of the maps, and a spectral axis (spectral; see _moments.spectral_axis
    for its defaults).

    Its configuration has the options below, but those of the grid and
    the orders when it has data, which give them (see from_data).

    Parameters
    ----------
    size : int or Sequence of int
        The number of pixels of the maps (nx, ny).
    spectral : SpectralAxis
        The spectral axis of the cube the maps are measured from (see
        SpectralAxis), and the convention of the velocities of
        the maps, as the data were measured, which is required (or its
        rest; see check_convention): none is right for all data
        (e.g. radio for HI, optical for lines of ionised gas, log for the
        stars of pPXF). By default, its channels are of 1 km/s and span
        1000 km/s, centred on 0.
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the maps (see Coords); rpix is
        the centre by default.
    rota : float, optional
        The rotation of the maps on the sky.
    flux_floor : float, optional
        The spectra of the model whose moment 0 is not above this fraction
        of the largest of the maps (in absolute value) have no moments
        (they are masked): 1e-6 by default, from 0 to below 1. It keeps
        out spectra of hardly any flux, whose moments are noise (e.g. of
        the rounding of convolutions); what the data lack is for their
        mask and their likelihood (e.g. upper limits).
    orders : Sequence of int, optional
        The orders of the moments.
    method : str, optional
        How the maps are measured from the spectra, as the maps of the
        data were: 'moments'; 'gaussian_fit' (the flux, centre and
        dispersion of a Gaussian fitted to each, for the orders 0 to 2);
        or 'gauss_hermite' (of a Gauss-Hermite series fitted to each, for
        the orders 0 to 4: its flux, centre, dispersion, h3 and h4, with
        the polynomials of van der Marel & Franx 1993, as pPXF).
    median_size : int, optional
        The size (an odd number of pixels, at least 3) of the median filter
        that smoothed the cube on the sky before the maps of the data were
        measured, which smooths the spectra of the model as well (e.g. 3,
        of cubes of KMOS3D): each channel takes the median of the square of
        the size about each pixel, with zero beyond the edges (see ModelCube).
        The median of a sum is not the sum of the medians: the channels of
        the model (the step of spectral) are then those of that cube. None
        by default.

    Raises
    ------
    ValueError
        If the orders are not valid for the method, flux_floor is not from
        0 to below 1, the convention of the velocities is not known, or
        median_size is not an odd number of at least 3.
    """

    dataset_class = DatasetPixelMoments

    emission = LineEmission
    orbit_stars = None

    # Moment maps have no spectral axis
    spectral_axis = None

    @staticmethod
    def type() -> str:
        return 'pixel_moments'

    @staticmethod
    def type_title() -> str:
        return 'Pixel moments (maps)'

    @classmethod
    def options_from_data(cls, dataset: Dataset) -> tuple[str, ...]:
        # The grid and the orders
        return super().options_from_data(dataset) + ('orders',)

    @classmethod
    def from_data(
            cls,
            dataset: DatasetPixelMoments,
            spectral: SpectralAxis,
            flux_floor: FluxFloor = 1e-6,
            method: Method = 'moments',
            median_size: MedianSize | None = None
    ) -> 'PixelMoments':
        """
        Make the observable of moment maps, on their grid and of their
        orders.

        Parameters
        ----------
        dataset : DatasetPixelMoments
            The moment maps.
        spectral : SpectralAxis
            As that of PixelMoments. Its size and rval by default, with
            moment 1, cover the velocities of the data: the range of moment
            1, and three times the largest dispersion (moment 2, but at
            least 100 km/s) on each side.
        flux_floor, method, median_size : optional
            As those of PixelMoments.

        Returns
        -------
        PixelMoments
            The observable.

        Raises
        ------
        TypeError
            If the dataset is not of moment maps.
        ValueError
            If the options are not valid.
        """
        check_options(PixelMoments.from_data, locals())
        return cls(
            **grid_options(dataset.grid()),
            spectral=_moments.spectral_axis(spectral, dataset),
            flux_floor=flux_floor, orders=dataset.orders(), method=method,
            median_size=median_size)

    def _dump_options(
            self, prefix: str, dump_path: bool, overwrite: bool
    ) -> dict[str, Any]:
        return dict(
            spectral=self._spectral.dump(),
            flux_floor=self._flux_floor,
            orders=self._orders,
            method=self._method,
            median_size=self._median_size)

    def __init__(
            self,
            size: Size,
            spectral: SpectralAxis,
            step: Step = (1, 1),
            rpix: Rpix | None = None,
            rval: Rval = (0, 0),
            rota: Rota = 0,
            flux_floor: FluxFloor = 1e-6,
            orders: Sequence[int] = (0, 1, 2),
            method: Method = 'moments',
            median_size: MedianSize | None = None
    ):
        check_options(PixelMoments.__init__, locals())
        super().__init__(size, step, rpix, rval, rota)
        check_median_size(median_size)
        self._median_size = median_size
        self._orders = _moments.check_moment_orders(orders, method)
        self._spectral = _moments.spectral_axis(spectral)
        self._flux_floor = flux_floor
        self._method = method

    def keys(self) -> tuple[str, ...]:
        return tuple(
            _moments.moment_item(i, self._method) for i in self._orders)

    def make_dataset(self, items: Mapping[str, Data]) -> DatasetPixelMoments:
        moments, hermite = _moments.moment_maps(
            items, self._orders, self._method)
        return DatasetPixelMoments(
            moments, hermite, **grid_options(self.grid(), sizes=False))

    def data_unit(
            self,
            dataset: Dataset | None,
            light: LightUnit | None = None
    ) -> DataUnit | None:
        return data_unit(
            dataset, 'moment0' if 0 in self._orders else None,
            abs(self.step()[0] * self.step()[1]), True, light)

    def light_items(self) -> tuple[str, ...]:
        return ('moment0',) if 0 in self._orders else ()

    def model_can_have_missing_values(self) -> bool:
        # (moment 0, the flux, always has a value)
        return any(order >= 1 for order in self._orders)

    def orders(self) -> tuple[int, ...]:
        """Return the orders of the moments, sorted."""
        return self._orders

    def flux_floor(self) -> float:
        """
        Return the fraction of the largest moment 0 of the model that the
        moment 0 of a spectrum must be above to have moments.
        """
        return self._flux_floor

    def method(self) -> str:
        """Return how the moments are measured (see MOMENT_METHODS)."""
        return self._method

    def spectral(self) -> SpectralAxis:
        """
        Return the spectral axis of the cube the maps are measured from.
        """
        return self._spectral

    def check_observation(self, setup: ObservationSetup) -> None:
        if setup.pixel_weights is not None and self._method != 'moments':
            raise ValueError(
                f"the method {self._method} does not support pixel_weights "
                f"yet")
        _moments.warn_unmasked_noise(self._flux_floor, setup.instrument)

    def check_scene(self, scene: 'Scene', setup: ObservationSetup) -> None:
        _moments.check_one_line(scene)
        super().check_scene(scene, setup)

    def check_dataset(self, dataset: Dataset) -> None:
        _moments.check_moment_items(dataset, self._method)
        super().check_dataset(dataset)

    def plan(
            self,
            driver: Driver,
            scene: 'Scene',
            setup: ObservationSetup
    ) -> 'PixelMomentsPlan':
        model_cube = _moments.spectra_model_cube(
            self, setup, self._median_size)
        return PixelMomentsPlan(self, model_cube, driver, scene, setup)


class PixelMomentsPlan(CubeObservablePlan):

    def __init__(
            self,
            moments: PixelMoments,
            model_cube: _model_cube.ModelCube,
            driver: Driver,
            scene: 'Scene',
            setup: ObservationSetup
    ):
        super().__init__(model_cube, driver, scene, setup)
        self._moments = _moments.MomentsPlan(
            driver, moments.size(), moments.orders(), moments.flux_floor(),
            moments.method(), setup.dtype)

    def _measure(self) -> ModelData:
        # The moment maps, one mask shared by all of them, and the weight
        # map of each moment
        plan = self._model_cube_plan
        grid = self._model_cube.grid()
        return self._moments.evaluate(
            grid.coords.step, grid.zero(), plan.cube(), plan.weights())
