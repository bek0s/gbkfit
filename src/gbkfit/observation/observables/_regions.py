"""
The base of the observables of data in regions of the sky (see
RegionsObservable), its plan, and the grids and weights of regions.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse

from gbkfit.dataset import Dataset
from gbkfit.driver import DeviceArray, Driver
from gbkfit.grid import Grid, SpectralAxis, make_grid, per_axis
from gbkfit.instrument import Instrument
from gbkfit.region import Regions, regions_parser
from ._cube_plan import CubeObservablePlan
from ._model_cube import ModelCube
from ._options import (
    GRID_OPTIONS, Rota, Rpix, Rval, Size, Step, grid_options,
    without_options_from_data)
from ._sparse_sums import SparseSumsPlan
from .base import Observable

if TYPE_CHECKING:
    from gbkfit.model import Scene
    from ..line_of_sight import LineOfSight
    from ..foreground import Foreground
    from ..weights import Weights


def spatial_grid_of_regions(
        regions: Regions,
        size: Size | None,
        step: Step | None,
        rpix: Rpix | None,
        rval: Rval | None,
        rota: float | None
) -> Grid:
    """
    Return the spatial grid of the cube of an observable of data in
    regions: that of the regions if they have one (bins), when the grid
    options must not be given; else (apertures) the grid of the options
    (see make_grid), of which size is required.
    """
    grid = regions.grid()
    given = [
        key for key, value in zip(
            GRID_OPTIONS, (size, step, rpix, rval, rota))
        if value is not None]
    if grid is not None and given:
        raise ValueError(
            f"the regions are on a grid, which is that of the model; "
            f"remove the options {given}")
    if grid is None:
        if size is None:
            raise ValueError(
                "the regions are on the sky; the size of the grid of the "
                "model is required")
        grid = make_grid(
            per_axis('size', size, 2), step, rpix, rval, rota)
    return grid


def spatial_options_from_regions(regions: Regions) -> tuple[str, ...]:
    """
    Return the options of the spatial grid of an observable of data in
    regions that the regions give (see spatial_grid_of_regions).
    """
    return GRID_OPTIONS if regions.grid() is not None else ()


def flux_weights(
        regions: Regions, grid: Grid
) -> scipy.sparse.csr_array:
    """
    Return the weights of the pixels of a spatial grid in regions (see
    Regions.weights) times the area of a pixel (arcsec^2): the area of
    each pixel inside each region. The sums of the surface brightness of
    the model with them are the fluxes of the regions, whatever the size
    of the pixels of the grid.
    """
    step_x, step_y = grid.coords.step[:2]
    return regions.weights(grid) * abs(step_x * step_y)


class RegionsObservable(Observable):
    """
    The base of the observables of data in regions of the sky (e.g.
    fibres, apertures or bins; see Regions): sums of the cube of the
    model, seen through the instrument, over the pixels of each region,
    times the area of each (see flux_weights), so that its data are
    fluxes (see flux_data_unit).

    The spatial axes of the cube are those of the regions if they are on
    a grid (bins), or given (apertures; see spatial_grid_of_regions). Its
    data give its regions, which its configuration without data has in a
    file (see dump). It does not support pixel weights yet.

    Parameters
    ----------
    regions : Regions
        The regions.
    size, step, rpix, rval, rota : optional
        The spatial grid of the cube (see make_grid), for regions on the
        sky (apertures): size is then required. Regions on a grid (bins)
        give it, and it must not be given.
    spectral : SpectralAxis, optional
        The spectral axis of an observable with one (see Observable).

    Raises
    ------
    ValueError
        If the spatial grid is given for regions on a grid, or not for
        regions on the sky.
    """

    def __init__(
            self,
            regions: Regions,
            size: Size | None,
            step: Step | None,
            rpix: Rpix | None,
            rval: Rval | None,
            rota: Rota | None,
            spectral: SpectralAxis | None = None
    ):
        spatial = spatial_grid_of_regions(
            regions, size, step, rpix, rval, rota)
        coords = spatial.coords
        super().__init__(
            spatial.size, coords.step, coords.rpix, coords.rval, coords.rota,
            spectral)
        self._regions = regions
        # The area of each pixel in the regions, so that their sums are
        # fluxes (an error if apertures are not inside the grid)
        self._flux_weights = flux_weights(regions, spatial)

    def dump(
            self,
            data: Dataset | None = None,
            prefix: str = '',
            dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        # The dump of the regions writes a file: only without data, which
        # give them (the other options that the data give are left out of
        # the dump after it is made, see without_options_from_data)
        info = dict(type=self.type())
        if data is None:
            info.update(regions=regions_parser.dump(
                self._regions, prefix=prefix, dump_path=dump_path,
                overwrite=overwrite))
        return info | without_options_from_data(
            self, self._dump_options(prefix, dump_path, overwrite), data)

    def _dump_spatial_grid(self) -> dict[str, Any]:
        """
        Return the options of the spatial grid of the cube, unless the
        regions give it (see spatial_grid_of_regions).
        """
        if self._regions.grid() is not None:
            return {}
        options = grid_options(self._grid)
        return {key: options[key] for key in GRID_OPTIONS}

    def regions(self) -> Regions:
        """Return the regions."""
        return self._regions

    def check_observation(
            self,
            instrument: Instrument,
            oversampling: Sequence[int],
            pixel_weights: 'Weights | None'
    ) -> None:
        if pixel_weights is not None:
            raise ValueError(
                f"{self.type()} does not support pixel_weights yet")

    def _check_dataset_coordinates(self, dataset: Dataset) -> None:
        if dataset.regions() != self._regions:
            raise ValueError(
                "the data and the observable have different regions")


class RegionsObservablePlan(CubeObservablePlan):
    """
    The evaluation of an observable of data in regions (see
    RegionsObservable): its cube (see CubeObservablePlan; without pixel
    weights), and the sums of its pixels in each region, with their flux
    weights (see flux_weights), in each of its channels: the spectra of
    the regions (see _sum_regions).
    """

    def __init__(
            self,
            weights: scipy.sparse.csr_array,
            model_cube: ModelCube,
            driver: Driver,
            scene: 'Scene',
            line_of_sight: 'LineOfSight',
            foreground: 'Foreground',
            dtype: np.dtype,
            flux_scale: str | None
    ):
        super().__init__(
            model_cube, driver, scene, line_of_sight, foreground, dtype, None,
            flux_scale)
        self._sums = SparseSumsPlan(weights, driver, dtype)
        # The spectra of the regions (nchannels, nregions)
        self._spectra = driver.mem_alloc_d(
            (model_cube.grid().size[2], self._sums.nrows()), dtype)

    def _sum_regions(self, cube: DeviceArray) -> DeviceArray:
        """
        Return the spectra of the regions (nchannels, nregions) of a cube
        of the grid of the low-resolution cube: the sums of its pixels in
        each region (with their flux weights) in each channel.
        """
        self._sums.evaluate(cube, self._spectra)
        return self._spectra
