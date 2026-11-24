"""
The observation of spectra in regions of the sky (e.g. fibres,
apertures, bins, or the whole field for an integrated spectrum).
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from gbkfit.config import check_options
from gbkfit.dataset import Data, Dataset, DatasetRegionSpectra, Regions
from gbkfit.driver import Driver
from gbkfit.grid import Grid, SpectraData, SpectralAxis
from gbkfit.instrument import Instrument
from gbkfit.lensing import SourceGrid
from gbkfit.model import Band, DataUnit, LightUnit
from ..depth import Depth
from ..noise import Noise
from ..plan import FluxScale, ModelData, ModelItem
from ..weights import Weights
from . import _model_cube
from ._cube_plan import SkyDrawing
from ._data_units import flux_data_unit
from ._options import Oversampling3, Rota, Rpix, Rval, Size, Step
from ._regions import (
    ObservationRegions, RegionsPlan, spatial_options_from_regions)
from .base import DType, SkyOptions

if TYPE_CHECKING:
    import scipy.sparse


__all__ = [
    'ObservationRegionSpectra'
]


class ObservationRegionSpectra(ObservationRegions):
    """
    Spectra in regions of the sky (e.g. fibres, apertures, bins, or the
    whole field): the summed spectrum of each region. The spectra are the
    sum of the cube of the model, seen through the instrument, in each
    region (see Regions), in each channel; that of the whole field is an
    integrated spectrum.

    The spatial axes of the cube are those of the regions if they are on a
    grid (bins), or given (apertures). Its options are those below, then
    those of every observation on the sky (see ObservationSky). Data paired
    with it give its regions and its spectral axis, and regions on a grid
    give the spatial grid (given too, they serve only without data);
    without data, regions and spectral are required.

    Parameters
    ----------
    regions : Regions, optional
        The regions.
    spectral : SpectralAxis, optional
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
        regions on the sky, the spectral axis is not valid (see
        make_cube_grid), the valid pixels of the regions are not of its
        channels, or an option of every observation on the sky is not
        valid (see ObservationSky).
    """

    dataset_class = DatasetRegionSpectra

    spectral_view = True
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
    def sampling_from_data(cls, dataset: Dataset) -> tuple[str, ...]:
        # The regions and the spectral axis, and the spatial grid of
        # regions on a grid
        return (
            ('regions', 'spectral')
            + spatial_options_from_regions(dataset.regions()))

    required_without_data = ('regions', 'spectral')

    @classmethod
    def _options_from_data(
            cls, dataset: DatasetRegionSpectra, options: dict[str, Any]
    ) -> dict[str, Any]:
        # (the regions, which give the spatial grid of regions on a grid,
        # and the spectral axis)
        return dict(
            regions=dataset.regions(),
            spectral=SpectralAxis.of(dataset.spectral_grid()))

    def _dump_options(
            self, prefix: str, dump_path: bool, overwrite: bool
    ) -> dict[str, Any]:
        return dict(spectral=self.spectral().dump()) \
            | self._dump_spatial_grid()

    def _kind_options(self) -> dict[str, Any]:
        return dict(regions=self._regions, spectral=self.spectral()) \
            | self._dump_spatial_grid()

    def __init__(
            self,
            regions: Regions | None = None,
            spectral: SpectralAxis | None = None,
            size: Size | None = None,
            step: Step | None = None,
            rpix: Rpix | None = None,
            rval: Rval | None = None,
            rota: Rota | None = None,
            source_grid: SourceGrid | None = None,
            instrument: Instrument | None = None,
            galaxies: Sequence[str] | None = None,
            components: Sequence[str] | None = None,
            lines: Sequence[str] | None = None,
            continuum: bool = False,
            band: str | Band | None = None,
            pixel_weights: Weights | None = None,
            flux_scale: str | None = None,
            offsets: Mapping[str, bool] | None = None,
            noise: Noise | None = None,
            oversampling: Oversampling3 = (1, 1, 1),
            depth: Depth | None = None,
            dtype: DType | None = None,
            fft_dtype: DType = 'float64',
            name: str | None = None,
            unit: str | None = None
    ):
        check_options(ObservationRegionSpectra.__init__, locals())
        options = self._options(dict(
            regions=regions, spectral=spectral, size=size, step=step,
            rpix=rpix, rval=rval, rota=rota))
        grid = self._regions_grid(**options)
        channels = self._regions.valid_channels()
        if channels is not None and channels != grid.spectral().size[0]:
            raise ValueError(
                f"the valid pixels of the regions are of {channels} "
                f"channels; the spectral axis has "
                f"{grid.spectral().size[0]}")
        super().__init__(
            grid, source_grid, instrument, galaxies, components, lines,
            continuum, band, pixel_weights, flux_scale, offsets, noise,
            oversampling, depth, dtype, fft_dtype, name, unit)

    def spectral_grid(self) -> Grid:
        """Return the grid of the spectral axis (one axis)."""
        return self._grid.spectral()

    def keys(self) -> tuple[str, ...]:
        return ('spectra',)

    def make_dataset(self, items: Mapping[str, Data]) -> DatasetRegionSpectra:
        return DatasetRegionSpectra(
            items['spectra'], self.regions(), self.spectral())

    def _data_unit(
            self,
            dataset: Dataset | None,
            light: LightUnit | None
    ) -> DataUnit | None:
        return flux_data_unit(dataset, 'spectra', False, light)

    def light_items(self) -> tuple[str, ...]:
        return ('spectra',)

    def flux_scale_kinds(self) -> dict[str, FluxScale]:
        # (each spectrum runs along the first axis, of each region)
        nregions = self._regions.nregions()
        return super().flux_scale_kinds() \
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
                f"{dataset.spectral_grid()}, but the observation has "
                f"{self.spectral_grid()}")

    def _output(self, data: np.ndarray) -> SpectraData:
        return SpectraData(data, self.spectral_grid().coords)

    def _plan(
            self, driver: Driver, drawing: SkyDrawing
    ) -> 'RegionSpectraPlan':
        options = self._sky_options
        # The masking of ModelCube is disabled: every pixel of a region adds
        # to its spectrum
        model_cube = _model_cube.ModelCube(
            self._grid,
            oversampling=options.oversampling,
            primary_beam=options.instrument.primary_beam(),
            psf=options.instrument.psf(),
            lsf=options.instrument.lsf(),
            smooth_weights=False,
            dtype=options.dtype,
            fft_dtype=options.fft_dtype)
        return RegionSpectraPlan(
            self._flux_weights, self._regions.valid(), model_cube, driver,
            drawing, options)


class RegionSpectraPlan(RegionsPlan):
    """
    The evaluation of an ObservationRegionSpectra: the sums of the
    pixels of the regions in each channel of the cube, of those valid
    there if the regions say which (see Regions.valid; the cube times
    them first).
    """

    def __init__(
            self,
            weights: 'scipy.sparse.csr_array',
            valid: np.ndarray | None,
            model_cube: _model_cube.ModelCube,
            driver: Driver,
            drawing: SkyDrawing,
            options: SkyOptions
    ):
        super().__init__(weights, model_cube, driver, drawing, options)
        self._valid = self._valid_cube = None
        if valid is not None:
            self._valid = driver.mem_copy_h2d(valid.astype(options.dtype))
            self._valid_cube = driver.mem_alloc_d(valid.shape, options.dtype)

    def _measure(self) -> ModelData:
        cube = self._model_cube_plan.cube()
        if self._valid is not None:
            # (on any driver: its arrays take the ufuncs of numpy)
            np.multiply(cube, self._valid, out=self._valid_cube)
            cube = self._valid_cube
        return dict(spectra=ModelItem(self._sum_regions(cube)))
