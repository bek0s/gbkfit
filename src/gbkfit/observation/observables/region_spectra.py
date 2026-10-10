from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np

from gbkfit.dataset import Dataset, DatasetRegionSpectra
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument
from gbkfit.model.base import GModel, GModelSCube, Selection
from gbkfit.region import Regions, regions_parser
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _dcube, _detail
from ._region_sums import RegionSumsPlan, flux_weights
from .base import ModelData, Observable

if TYPE_CHECKING:
    import scipy.sparse

    from ..foreground import Foreground


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
    spec_size : int
        The number of channels.
    spec_step : float, optional
        The width of the channels (km/s).
    spec_rpix : float, optional
        The channel of spec_rval; by default, the centre.
    spec_rval : float, optional
        The velocity of the channel spec_rpix (km/s).
    spec_rest : str or Quantity, optional
        The rest wavelength or frequency of the velocities of the spectral
        axis, if known (see gridutils.make_rest).
    size, step, rpix, rval, rota : optional
        The spatial grid of the cube (see gridutils.make_grid), for
        regions on the sky (apertures): size is then required. Regions on
        a grid (bins) give it, and it must not be given.

    Raises
    ------
    ConfigError
        If the spatial grid is given for regions on a grid, or not for
        regions on the sky.
    """

    dataset_class = DatasetRegionSpectra

    # The axes of the cube of the model: x, y and the spectral axis
    spectral_axis = 2

    @staticmethod
    def type() -> str:
        return 'region_spectra'

    @staticmethod
    def is_compatible(gmodel: GModel) -> bool:
        return isinstance(gmodel, GModelSCube)

    @classmethod
    def options_from_data(cls, dataset: Dataset) -> tuple[str, ...]:
        # The regions and the spectral axis, and the spatial grid of
        # regions on a grid
        return (
            ('regions', 'spec_size', 'spec_step', 'spec_rpix', 'spec_rval',
             'spec_rest')
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
        spectral = dataset.spectral_grid()
        coords = spectral.coords
        return cls(
            dataset.regions(), spectral.size[0], coords.step[0],
            coords.rpix[0], coords.rval[0], coords.rest,
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
            spec_step: float = 1,
            spec_rpix: float | None = None,
            spec_rval: float = 0,
            spec_rest: str | astropy.units.Quantity | None = None,
            size: Sequence[int] | None = None,
            step: Sequence[float] | None = None,
            rpix: Sequence[float] | None = None,
            rval: Sequence[float] | None = None,
            rota: float | None = None
    ):
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
            gmodel: GModel,
            foreground: 'Foreground',
            instrument: Instrument,
            scale: Sequence[int],
            dtype: np.dtype,
            selection: Selection
    ) -> 'RegionSpectraPlan':
        if gmodel.has_weights():
            raise ConfigError(
                "region_spectra does not support gmodels with weights "
                "(wtraits) yet")
        # The masking of DCube is disabled: every pixel of a region adds
        # to its spectrum
        dcube = _dcube.DCube(
            size=self.size(),
            step=self.step(),
            rpix=self.rpix(),
            rval=self.rval(),
            rota=self.rota(),
            rest=self.rest(),
            scale=tuple(scale),
            primary_beam=instrument.primary_beam(),
            psf=instrument.psf(),
            lsf=instrument.lsf(),
            smooth_weights=False,
            mask_cutoff=None,
            mask_apply=False,
            dtype=dtype)
        return RegionSpectraPlan(
            self._weights, dcube, driver, gmodel, foreground, dtype,
            selection)


class RegionSpectraPlan(_detail.DCubePlanBase):

    def __init__(
            self,
            weights: 'scipy.sparse.csr_array',
            dcube: _dcube.DCube,
            driver: Driver,
            gmodel: GModel,
            foreground: 'Foreground',
            dtype: np.dtype,
            selection: Selection
    ):
        super().__init__(
            dcube, driver, gmodel, foreground, dtype, selection)
        self._sums = RegionSumsPlan(weights, driver, dtype)
        self._spectra = driver.mem_alloc_d(
            (dcube.size()[2], self._sums.nregions()), dtype)

    def evaluate(
            self,
            params: dict[str, float | np.ndarray],
            out_extra: dict[str, Any] | None
    ) -> ModelData:
        self._evaluate_cube(
            params, out_extra, _dcube.cube_extra, _dcube.cube_extra)
        self._sums.evaluate(self._dcube_plan.dcube(), self._spectra)
        return dict(spectra=dict(d=self._spectra, m=None, w=None))
