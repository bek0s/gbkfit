"""
The observable of spectra in regions of the sky (e.g. fibres,
apertures, bins, or the whole field for an integrated spectrum).
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from gbkfit.config import check_options
from gbkfit.dataset import Data, Dataset, DatasetRegionSpectra
from gbkfit.driver import Driver
from gbkfit.grid import Grid, SpectraData, SpectralAxis
from gbkfit.model import DataUnit, LightUnit, LineEmission
from gbkfit.region import Regions
from . import _model_cube
from ._data_units import flux_data_unit
from ._options import Rota, Rpix, Rval, Size, Step
from ._regions import (
    RegionsObservable, RegionsObservablePlan, spatial_options_from_regions)
from .base import FluxScale, ModelData, ModelItem, ObservationSetup

if TYPE_CHECKING:
    import scipy.sparse

    from gbkfit.model import Scene


__all__ = [
    'RegionSpectra'
]


class RegionSpectra(RegionsObservable):
    """
    Spectra in regions of the sky (e.g. fibres, apertures, bins, or the
    whole field): the summed spectrum of each region. The spectra are the
    sum of the cube of the model, seen through the instrument, in each
    region (see Regions), in each channel; that of the whole field is an
    integrated spectrum.

    The spatial axes of the cube are those of the regions if they are on
    a grid (bins), or given (apertures). Its configuration has the options
    below, but the regions and the spectral axis when it has data, which
    give them, and the spatial grid of regions on a grid (see from_data).

    Parameters
    ----------
    regions : Regions
        The regions.
    spectral : SpectralAxis
        The spectral axis of the cube: its size, its world coordinates and
        the convention of its velocities (see make_cube_grid).
    size, step, rpix, rval, rota : optional
        The spatial grid of the cube (see make_grid), for
        regions on the sky (apertures): size is then required. Regions on
        a grid (bins) give it, and it must not be given.

    Raises
    ------
    ValueError
        If the spatial grid is given for regions on a grid, or not for
        regions on the sky, or the spectral axis is not valid (see
        make_cube_grid).
    """

    dataset_class = DatasetRegionSpectra

    emission = LineEmission
    orbit_stars = None

    # The axes of the cube of the model: x, y and the spectral axis
    spectral_axis = 2

    @staticmethod
    def type() -> str:
        return 'region_spectra'

    @staticmethod
    def type_title() -> str:
        return 'Region spectra (bins, fibres)'

    @classmethod
    def options_from_data(cls, dataset: Dataset) -> tuple[str, ...]:
        # The regions and the spectral axis, and the spatial grid of
        # regions on a grid
        return (
            ('regions', 'spectral')
            + spatial_options_from_regions(dataset.regions()))

    @classmethod
    def from_data(
            cls,
            dataset: DatasetRegionSpectra,
            size: Size | None = None,
            step: Step | None = None,
            rpix: Rpix | None = None,
            rval: Rval | None = None,
            rota: Rota | None = None
    ) -> 'RegionSpectra':
        """
        Make the observable of the spectra of regions, of their regions
        and on their spectral axis.

        Parameters
        ----------
        dataset : DatasetRegionSpectra
            The spectra.
        size, step, rpix, rval, rota : optional
            As those of RegionSpectra: the spatial grid of the cube, for
            regions on the sky.

        Returns
        -------
        RegionSpectra
            The observable.

        Raises
        ------
        ValueError
            If the dataset is not of spectra in regions, or the options
            are not valid.
        """
        check_options(RegionSpectra.from_data, locals())
        return cls(
            dataset.regions(),
            SpectralAxis.of(dataset.spectral_grid()),
            size, step, rpix, rval, rota)

    def _dump_options(
            self, prefix: str, dump_path: bool, overwrite: bool
    ) -> dict[str, Any]:
        return dict(spectral=self.spectral().dump()) \
            | self._dump_spatial_grid()

    def __init__(
            self,
            regions: Regions,
            spectral: SpectralAxis,
            size: Size | None = None,
            step: Step | None = None,
            rpix: Rpix | None = None,
            rval: Rval | None = None,
            rota: Rota | None = None
    ):
        check_options(RegionSpectra.__init__, locals())
        super().__init__(regions, size, step, rpix, rval, rota, spectral)
        valid = regions.valid()
        if valid is not None and len(valid) != self.spectral_grid().size[0]:
            raise ValueError(
                f"the valid pixels of the regions are of {len(valid)} "
                f"channels; the spectral axis has "
                f"{self.spectral_grid().size[0]}")

    def spectral_grid(self) -> Grid:
        """Return the grid of the spectral axis (one axis)."""
        return self._grid.spectral()

    def keys(self) -> tuple[str, ...]:
        return ('spectra',)

    def make_dataset(self, items: Mapping[str, Data]) -> DatasetRegionSpectra:
        return DatasetRegionSpectra(
            items['spectra'], self.regions(), self.spectral())

    def data_unit(
            self,
            dataset: Dataset | None,
            light: LightUnit | None = None
    ) -> DataUnit | None:
        return flux_data_unit(dataset, 'spectra', False, light)

    def light_items(self) -> tuple[str, ...]:
        return ('spectra',)

    def flux_scales(self) -> dict[str, FluxScale]:
        # (each spectrum runs along the first axis, of each region)
        nregions = self._regions.nregions()
        return super().flux_scales() \
            | dict(per_spectrum=FluxScale(('spectra',), 0, nregions))

    def linear_in_light(self) -> bool:
        # (the sums of its spectra of the light, through its instrument:
        # linear)
        return True

    def model_can_have_missing_values(self) -> bool:
        return False

    def _check_dataset_coordinates(
            self, dataset: DatasetRegionSpectra
    ) -> None:
        super()._check_dataset_coordinates(dataset)
        if dataset.spectral_grid() != self.spectral_grid():
            raise ValueError(
                f"the spectra have the spectral axis "
                f"{dataset.spectral_grid()}, but the observable has "
                f"{self.spectral_grid()}")

    def output(self, data: np.ndarray) -> SpectraData:
        return SpectraData(data, self.spectral_grid().coords)

    def plan(
            self,
            driver: Driver,
            scene: 'Scene',
            setup: ObservationSetup
    ) -> 'RegionSpectraPlan':
        # The masking of ModelCube is disabled: every pixel of a region adds
        # to its spectrum
        model_cube = _model_cube.ModelCube(
            self._grid,
            oversampling=setup.oversampling,
            primary_beam=setup.instrument.primary_beam(),
            psf=setup.instrument.psf(),
            lsf=setup.instrument.lsf(),
            smooth_weights=False,
            dtype=setup.dtype,
            fft_dtype=setup.fft_dtype)
        return RegionSpectraPlan(
            self._flux_weights, self._regions.valid(), model_cube, driver,
            scene, setup)


class RegionSpectraPlan(RegionsObservablePlan):
    """
    The evaluation of a RegionSpectra: the sums of the pixels of the
    regions in each channel of the cube, of those valid there if the
    regions say which (see Regions.valid; the cube times them first).
    """

    def __init__(
            self,
            weights: 'scipy.sparse.csr_array',
            valid: np.ndarray | None,
            model_cube: _model_cube.ModelCube,
            driver: Driver,
            scene: 'Scene',
            setup: ObservationSetup
    ):
        super().__init__(weights, model_cube, driver, scene, setup)
        self._valid = self._valid_cube = None
        if valid is not None:
            self._valid = driver.mem_copy_h2d(valid.astype(setup.dtype))
            self._valid_cube = driver.mem_alloc_d(valid.shape, setup.dtype)

    def _measure(self) -> ModelData:
        cube = self._model_cube_plan.cube()
        if self._valid is not None:
            # (on any driver: its arrays take the ufuncs of numpy)
            np.multiply(cube, self._valid, out=self._valid_cube)
            cube = self._valid_cube
        return dict(spectra=ModelItem(self._sum_regions(cube)))
