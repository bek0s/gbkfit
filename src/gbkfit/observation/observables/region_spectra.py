from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from gbkfit.dataset import Data, Dataset, DatasetRegionSpectra
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument
from gbkfit.model import DataUnit, LineEmission
from gbkfit.region import Regions, regions_parser
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _dcube, _detail
from ._region_sums import RegionSumsPlan, flux_weights
from .base import FluxScale, ModelData, Observable

if TYPE_CHECKING:
    import scipy.sparse

    from gbkfit.model import Scene
    from ..line_of_sight import LineOfSight
    from ..foreground import Foreground
    from ..weights import Weights


__all__ = [
    'RegionSpectra'
]


class RegionSpectra(Observable):
    """
    Spectra in regions of the sky (see Regions; e.g. fibres, apertures,
    bins, or the whole field for an integrated spectrum): the sum of the
    cube of the model, seen through the instrument, in each region, in
    each channel.

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
        the convention of its velocities (see gridutils.make_cube_grid).
    size, step, rpix, rval, rota : optional
        The spatial grid of the cube (see gridutils.make_grid), for
        regions on the sky (apertures): size is then required. Regions on
        a grid (bins) give it, and it must not be given.

    Raises
    ------
    ConfigError
        If the spatial grid is given for regions on a grid, or not for
        regions on the sky, or the spectral axis is not valid (see
        gridutils.make_cube_grid).
    """

    dataset_class = DatasetRegionSpectra

    emission = LineEmission

    # The axes of the cube of the model: x, y and the spectral axis
    spectral_axis = 2

    @staticmethod
    def type() -> str:
        return 'region_spectra'

    @classmethod
    def options_from_data(cls, dataset: Dataset) -> tuple[str, ...]:
        # The regions and the spectral axis, and the spatial grid of
        # regions on a grid
        return (
            ('regions', 'spectral')
            + _detail.spatial_options_from_regions(dataset.regions()))

    @classmethod
    def load(
            cls, info: dict[str, Any],
            dataset: DatasetRegionSpectra | None = None
    ) -> 'RegionSpectra':
        if dataset is None:
            parseutils.load_option_and_update_info(
                regions_parser, info, 'regions', required=True)
        return _detail.load_observable(cls, info, dataset, 2)

    @classmethod
    def from_data(
            cls,
            dataset: DatasetRegionSpectra,
            size: Sequence[int] | None = None,
            step: Sequence[float] | None = None,
            rpix: Sequence[float] | None = None,
            rval: Sequence[float] | None = None,
            rota: float | None = None
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
        ConfigError
            If the dataset is not of spectra in regions, or the options
            are not valid.
        """
        cls._require_dataset_class(dataset)
        return cls(
            dataset.regions(),
            gridutils.SpectralAxis.of(dataset.spectral_grid()),
            size, step, rpix, rval, rota)

    def dump(
            self,
            data: DatasetRegionSpectra | None = None,
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
                spectral=self.spectral().dump())
        return info | _detail.dump_spatial_grid(self, self._regions)

    def __init__(
            self,
            regions: Regions,
            spectral: gridutils.SpectralAxis,
            size: Sequence[int] | None = None,
            step: Sequence[float] | None = None,
            rpix: Sequence[float] | None = None,
            rval: Sequence[float] | None = None,
            rota: float | None = None
    ):
        spatial = _detail.spatial_grid_of_regions(
            regions, size, step, rpix, rval, rota)
        coords = spatial.coords
        super().__init__(
            spatial.size, coords.step, coords.rpix, coords.rval, coords.rota,
            spectral)
        self._regions = regions
        # The area of each pixel in the regions, so that their sums are
        # fluxes (an error if apertures are not inside the grid)
        self._weights = flux_weights(regions, spatial)

    def regions(self) -> Regions:
        """Return the regions."""
        return self._regions

    def spectral_grid(self) -> gridutils.Grid:
        """Return the grid of the spectral axis (one axis)."""
        return self._grid.spectral()

    def keys(self) -> tuple[str, ...]:
        return ('spectra',)

    def dataset(self, items: Mapping[str, Data]) -> DatasetRegionSpectra:
        return DatasetRegionSpectra(
            items['spectra'], self.regions(), self.spectral())

    def data_unit(self, dataset: Dataset | None) -> DataUnit | None:
        return _detail.region_data_unit(dataset, 'spectra', False)

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

    def model_can_lack_values(self) -> bool:
        return False

    def _require_matching_coordinates(
            self, dataset: DatasetRegionSpectra
    ) -> None:
        if dataset.regions() != self._regions:
            raise ConfigError(
                "the data and the observable have different regions")
        if dataset.spectral_grid() != self.spectral_grid():
            raise ConfigError(
                f"the spectra have the spectral axis "
                f"{dataset.spectral_grid()}, but the observable has "
                f"{self.spectral_grid()}")

    def output(self, data: np.ndarray) -> gridutils.SpectraData:
        return gridutils.SpectraData(data, self.spectral_grid().coords)

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
    ) -> 'RegionSpectraPlan':
        if pixel_weights is not None:
            raise ConfigError(
                "region_spectra does not support pixel_weights yet")
        # The masking of DCube is disabled: every pixel of a region adds
        # to its spectrum
        dcube = _dcube.DCube(
            size=self.size(),
            step=self.step(),
            rpix=self.rpix(),
            rval=self.rval(),
            rota=self.rota(),
            rest=self.rest(),
            medium=self.medium(),
            convention=self.convention(),
            oversampling=tuple(oversampling),
            primary_beam=instrument.primary_beam(),
            psf=instrument.psf(),
            lsf=instrument.lsf(),
            smooth_weights=False,
            dtype=dtype,
            fft_dtype=fft_dtype)
        return RegionSpectraPlan(
            self._weights, dcube, driver, scene, line_of_sight, foreground,
            dtype, flux_scale)


class RegionSpectraPlan(_detail.DCubePlanBase):

    def __init__(
            self,
            weights: 'scipy.sparse.csr_array',
            dcube: _dcube.DCube,
            driver: Driver,
            scene: 'Scene',
            line_of_sight: 'LineOfSight',
            foreground: 'Foreground',
            dtype: np.dtype,
            flux_scale: str | None
    ):
        super().__init__(
            dcube, driver, scene, line_of_sight, foreground, dtype, None,
            flux_scale)
        self._sums = RegionSumsPlan(weights, driver, dtype)
        self._spectra = driver.mem_alloc_d(
            (dcube.size()[2], self._sums.nregions()), dtype)

    def _measure(self) -> ModelData:
        self._sums.evaluate(self._dcube_plan.dcube(), self._spectra)
        return dict(spectra=dict(d=self._spectra, m=None, w=None))
