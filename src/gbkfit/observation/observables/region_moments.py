from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np
import scipy.signal
import scipy.sparse

from gbkfit.dataset import Data, Dataset, DatasetRegionMoments
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument
from gbkfit.model import DataUnit, Extras, GalaxyState, LineEmission
from gbkfit.model.components.spheroids import OrbitIntegrator, SkyGrid
from gbkfit.params import ParamValues
from gbkfit.region import Regions, regions_parser
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _dcube, _detail, _moments, _orbit_outputs
from ._region_sums import RegionSumsPlan, flux_weights
from .base import HostModelData, ModelData, Observable, ObservablePlan

if TYPE_CHECKING:
    from gbkfit.model import Scene
    from gbkfit.model.components.spheroids import (
        OrbitLibraryState, OrbitRecords)
    from ..line_of_sight import LineOfSight
    from ..foreground import Foreground
    from ..noise import Noise
    from ..weights import Weights


__all__ = [
    'RegionMoments'
]


# The smallest share of a bin in the light of a point of the sky that the
# sky grids of orbits keep (see RegionMoments.sky_grid)
_SHARE_FLOOR = 1e-6


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

    E.g. the kinematics of the stars of a spheroid (lines ['stars'] of its
    observation, see model.StarsEmission) in the Voronoi bins of pPXF.
    Those are of the distribution of the velocities of the stars itself
    (the fit of the spectra took the LSF out), seen through the PSF: the
    instrument of their observation has a PSF and no LSF. The velocities
    of pPXF are c ln(1 + z), the convention 'log' (with any spec_rest
    wavelength: the light of stars has no rest). Models that predict the
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
    size, step, rpix, rval, rota : optional
        The spatial grid of the cube (see gridutils.make_grid), for
        regions on the sky (apertures): size is then required. Regions on
        a grid (bins) give it, and it must not be given.
    flux_floor, orders, spec_size, spec_step, spec_rval, spec_rest, \
spec_medium, spec_convention, method : optional
        As those of PixelMoments: the regions whose moment 0 is not above
        flux_floor times the largest of the regions are masked.

    Raises
    ------
    ConfigError
        If the spatial grid is given for regions on a grid, or not for
        regions on the sky, or the options of the moments are not valid
        (see PixelMoments).
    """

    dataset_class = DatasetRegionMoments

    emission = LineEmission

    # The moments have no spectral axis
    spectral_axis = None

    @staticmethod
    def type() -> str:
        return 'region_moments'

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
            flux_floor: float = 1e-6,
            spec_size: int | None = None,
            spec_step: float = _moments.SPEC_STEP,
            spec_rval: float | None = None,
            spec_rest: str | astropy.units.Quantity | None = None,
            spec_medium: str | None = None,
            spec_convention: str | None = None,
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
        flux_floor, spec_size, spec_step, spec_rval, spec_rest, \
spec_medium, method : optional
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
            flux_floor=flux_floor, orders=dataset.orders(),
            spec_step=spec_step, spec_rest=spec_rest,
            spec_medium=spec_medium, spec_convention=spec_convention,
            method=method)

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
            flux_floor=self._flux_floor,
            spec_size=self._spec_size,
            spec_step=self._spec_step,
            spec_rval=self._spec_rval,
            spec_rest=_detail.dump_rest(self._spec_rest),
            spec_medium=self._spec_medium,
            spec_convention=self._spec_convention,
            method=self._method)

    def __init__(
            self,
            regions: Regions,
            size: Sequence[int] | None = None,
            step: Sequence[float] | None = None,
            rpix: Sequence[float] | None = None,
            rval: Sequence[float] | None = None,
            rota: float | None = None,
            flux_floor: float = 1e-6,
            orders: Sequence[int] = (0, 1, 2),
            spec_size: int | None = None,
            spec_step: float = _moments.SPEC_STEP,
            spec_rval: float = 0,
            spec_rest: str | astropy.units.Quantity | None = None,
            spec_medium: str | None = None,
            spec_convention: str | None = None,
            method: str = 'moments'
    ):
        spatial = _detail.spatial_grid_of_regions(
            regions, size, step, rpix, rval, rota)
        coords = spatial.coords
        super().__init__(
            spatial.size, coords.step, coords.rpix, coords.rval, coords.rota)
        self._orders = _moments.check_moment_options(
            orders, flux_floor, method, spec_size, spec_step)
        if spec_size is None:
            spec_size = _moments.default_spec_size(spec_step)
        self._regions = regions
        self._flux_floor = flux_floor
        self._method = method
        self._spec_size = spec_size
        self._spec_step = spec_step
        self._spec_rval = spec_rval
        self._spec_rest = gridutils.make_rest(spec_rest)
        self._spec_medium = gridutils.make_medium(
            spec_medium, self._spec_rest)
        _moments.check_velocities(spec_rest, spec_convention)
        self._spec_convention = gridutils.make_convention(
            spec_convention, self._spec_rest)
        # The area of each pixel in the regions, so that their sums are
        # fluxes (an error if apertures are not inside the grid)
        self._weights = flux_weights(regions, spatial)

    def regions(self) -> Regions:
        """Return the regions."""
        return self._regions

    def orders(self) -> tuple[int, ...]:
        return self._orders

    def flux_floor(self) -> float:
        return self._flux_floor

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

    def spec_medium(self) -> str | None:
        return self._spec_medium

    def spec_convention(self) -> str | None:
        return self._spec_convention

    def keys(self) -> tuple[str, ...]:
        return tuple(f'moment{i}' for i in self._orders)

    def dataset(self, items: Mapping[str, Data]) -> DatasetRegionMoments:
        return DatasetRegionMoments(
            {order: items[f'moment{order}'] for order in self.orders()},
            self.regions())

    def data_unit(self, dataset: Dataset | None) -> DataUnit | None:
        return _detail.region_data_unit(
            dataset, 'moment0' if 0 in self._orders else None, True)

    def light_items(self) -> tuple[str, ...]:
        return ('moment0',) if 0 in self._orders else ()

    def model_can_lack_values(self) -> bool:
        # (moment 0, the flux, always has a value)
        return any(order >= 1 for order in self._orders)

    def check_instrument(self, instrument: Instrument) -> None:
        _moments.warn_unmasked_noise(self._flux_floor, instrument)

    def require_matching(self, dataset: Dataset) -> None:
        super().require_matching(dataset)
        _moments.check_moment_units(dataset, self._method)

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
            scene: 'Scene',
            line_of_sight: 'LineOfSight',
            foreground: 'Foreground',
            instrument: Instrument,
            oversampling: Sequence[int],
            dtype: np.dtype,
            fft_dtype: np.dtype,
            pixel_weights: 'Weights | None',
            flux_scale: str | None
    ) -> 'RegionMomentsPlan':
        if pixel_weights is not None:
            raise ConfigError(
                "region_moments does not support pixel_weights yet")
        if scene.orbit_emitters():
            self._check_orbits(scene, flux_scale)
            return OrbitMomentsPlan(
                self, scene, instrument, oversampling, driver, dtype)
        dcube = _moments.spectra_dcube(
            self, oversampling, instrument, dtype, fft_dtype)
        return RegionMomentsPlan(
            self, self._weights, dcube, driver, scene, line_of_sight,
            foreground, dtype, flux_scale)


    def sky_grid(
            self, instrument: Instrument, oversampling: Sequence[int]
    ) -> SkyGrid:
        """
        Return the grid of the sky of its bins for orbits (see SkyGrid),
        through the PSF of an instrument: the pixels of the bins, of the
        given oversampling (along x and y, as that of the model), and
        around them the half size of the PSF; the share of a bin in the
        light of a point in a cell is its mask correlated with the PSF there
        (the share of the light of the point that the PSF puts in it),
        shares below 1e-6 left out.

        Raises
        ------
        ConfigError
            If its regions are not bins (on a grid).
        """
        grid = self._regions.grid()
        if grid is None:
            raise ConfigError(
                "orbits are recorded in bins of the pixels of a grid, not in "
                "apertures")
        fx, fy = oversampling[:2]
        step = grid.coords.step[:2]
        psf = instrument.psf()
        kernel = np.ones((1, 1)) if psf is None else psf.asarray(
            (step[0] / fx, step[1] / fy), rota=grid.coords.rota)
        my, mx = kernel.shape[0] // 2, kernel.shape[1] // 2
        index = np.repeat(
            np.repeat(self._regions.index(), fy, axis=0), fx, axis=1)
        index = np.pad(index, ((my, my), (mx, mx)), constant_values=-1)
        flipped = kernel[::-1, ::-1]
        cells, bins, shares = [], [], []
        for b in range(self._regions.nregions()):
            ys, xs = np.nonzero(index == b)
            y0, y1 = ys.min() - my, ys.max() + my + 1
            x0, x1 = xs.min() - mx, xs.max() + mx + 1
            mask = (index[y0:y1, x0:x1] == b).astype(float)
            share = scipy.signal.fftconvolve(mask, flipped, mode='same')
            yy, xx = np.nonzero(share > _SHARE_FLOOR)
            cells.append((yy + y0) * index.shape[1] + xx + x0)
            bins.append(np.full(len(yy), b))
            shares.append(share[yy, xx])
        # (pixel p covers [p - 1/2, p + 1/2), and f cells along each axis)
        matrix, offset = gridutils.sky_to_pixel(grid)
        f = np.array([fx, fy], dtype=float)
        to_cells = np.column_stack((
            f[:, None] * matrix, f * (offset + 0.5) + (mx, my)))
        return SkyGrid(to_cells, index.shape, scipy.sparse.csr_array(
            (np.concatenate(shares),
             (np.concatenate(cells), np.concatenate(bins))),
            shape=(index.size, self._regions.nregions())))

    def _check_orbits(self, scene: 'Scene', flux_scale: str | None) -> None:
        """
        Raise ConfigError unless it can see stars on orbits whose weights
        are solved (see OrbitMomentsPlan): only them, in bins, of the
        method gauss_hermite and the orders 1 to M (V, sigma, h3 ... hM),
        without a flux scale.
        """
        if sum(len(emitters) for emitters in scene.emitters()) != 1 \
                or len(scene.orbit_emitters()) != 1:
            raise ConfigError(
                "moments of regions that see stars on orbits whose weights "
                "are solved (e.g. of a spheroid of Schwarzschild kinematics) "
                "see only them, of one component: give the observation that "
                "component or the lines of its stars only")
        if self._regions.grid() is None:
            raise ConfigError(
                "moments of stars on orbits whose weights are solved are of "
                "bins of the pixels of a grid (e.g. Voronoi bins), not of "
                "apertures")
        orders = self._orders
        if self._method != 'gauss_hermite' or len(orders) < 2 \
                or orders != tuple(range(1, len(orders) + 1)):
            raise ConfigError(
                f"moments of stars on orbits whose weights are solved are "
                f"Gauss-Hermite series (method gauss_hermite) of the orders "
                f"1 to M (V, sigma, h3 ... hM); they are of the method "
                f"{self._method!r} and the orders {list(orders)}")
        if flux_scale is not None:
            raise ConfigError(
                "moments of stars on orbits whose weights are solved have "
                "no flux scale")


class OrbitMomentsPlan(ObservablePlan):
    """
    The moments of the bins of stars on orbits whose weights are solved
    (see Component.orbit_weights), at given weights (e.g. those that an
    objective solves; it models the bins itself in its fits, see
    objective.Objective): the orbits of the state of their spheroid (see
    OrbitLibraryState), integrated and recorded with the light of their
    weights in the bins, through the PSF (see RegionMoments.sky_grid), as
    the spectrum of each bin on the spectral axis of the moments, whose
    moments are measured as those of the spectra of any regions (see
    RegionMomentsPlan).

    Its extra outputs, wanted, come from one pass that records the orbits
    in the bins and in the pixels of their grid and cells: the cube of the
    pixels ('cube': their spectra convolved with the PSF) and the moments
    of each pixel ('unbinned_moment<m>', measured as those of the bins),
    and of the component: the moments of the light of the orbits in its
    cells and in its spherical shells (the density, the rotation, the
    dispersions and the anisotropy: 'orbit_cells' and 'orbit_profile')
    and each orbit ('orbits': its energy, angular momentum, circularity,
    weight and light).

    Parameters
    ----------
    moments : RegionMoments
        The moments.
    scene : Scene
        What they see: the stars of one component, on orbits.
    instrument : Instrument
        The instrument of their observation.
    oversampling : Sequence of int
        The oversampling of their model (along x and y).
    driver : Driver
        The driver.
    dtype : np.dtype
        The precision of the model.
    """

    def __init__(
            self,
            moments: RegionMoments,
            scene: 'Scene',
            instrument: Instrument,
            oversampling: Sequence[int],
            driver: Driver,
            dtype: np.dtype
    ):
        (galaxy, component), = scene.orbit_emitters()
        self._galaxy = galaxy
        self._component = component
        self._names = scene.component_names(galaxy, component)
        spheroid = scene.component(galaxy, component)
        self._weights = spheroid.orbit_weights()
        # (the ids of the extra outputs of the component)
        self._ids = dict(
            galaxy=scene.galaxy_label(galaxy),
            component=spheroid.name() or f'component{component}')
        self._grid = moments.sky_grid(instrument, oversampling)
        self._driver = driver
        self._dtype = dtype
        nregions = moments.regions().nregions()
        self._nregions = nregions
        # (the spectral axis: its channels, their width, and the velocity
        # of the centre of the first)
        self._nchannels = moments.spec_size()
        self._step = moments.spec_step()
        self._zero = moments.spec_rval() \
            - (self._nchannels / 2 - 0.5) * self._step
        self._spectra = driver.mem_alloc_d(
            (self._nchannels, 1, nregions), dtype)
        self._moments = _moments.MomentsPlan(
            driver, (nregions, 1), moments.orders(), moments.flux_floor(),
            moments.method(), dtype)
        self._integrator: OrbitIntegrator | None = None
        # The pixels of the grid of the bins, their cube on the spectral
        # axis of the moments and their moments, for the extra outputs
        grid = moments.regions().grid()
        assert grid is not None
        shape = moments.regions().index().shape
        self._pixels = _orbit_outputs.PixelCube(
            grid, shape, instrument, oversampling)
        coords = grid.coords
        self._cube_coords = gridutils.Coords(
            coords.step[:2] + (self._step,),
            coords.rpix[:2] + (self._nchannels / 2 - 0.5,),
            coords.rval[:2] + (moments.spec_rval(),), coords.rota,
            moments.spec_rest(), moments.spec_medium(),
            moments.spec_convention())
        self._map_coords = coords
        self._orders = moments.orders()
        self._flux_floor = moments.flux_floor()
        self._method = moments.method()
        self._outputs: tuple[
            OrbitIntegrator, _moments.MomentsPlan] | None = None

    def evaluate(
            self,
            params: ParamValues,
            galaxies: Sequence[GalaxyState],
            out_extra: Extras | None
    ) -> ModelData:
        state = galaxies[self._galaxy].states[self._component]
        values = {name: params[full] for name, full in self._names.items()}
        weights = np.asarray(values[self._weights], dtype=float)
        light = state.orbit_light(weights)
        centre = [values['xpos'], values['ypos']]
        geometry = (values['posa'], values['incl'])
        # (channel j holds the velocities vsys + v from zero + (j - 1/2)
        # step to zero + (j + 1/2) step)
        channels = (1 / self._step,
                    (values['vsys'] - self._zero) / self._step + 0.5)
        if out_extra is None:
            if self._integrator is None:
                self._integrator = OrbitIntegrator(
                    self._driver, self._dtype, len(state.starts),
                    accuracy=state.accuracy, max_steps=state.max_steps(),
                    nsamples=state.samples, nphi=state.copies, twins=True,
                    sky=[self._grid], nchannels=self._nchannels)
            records = self._integrator.integrate(
                state.potential, state.starts, state.turns, geometry,
                centres=np.array([centre]), light=light, channels=channels)
        else:
            integrator, _ = self._outputs_of(state)
            records = integrator.integrate(
                state.potential, state.starts, state.turns, geometry,
                cell_edges=state.cell_edges,
                centres=np.array([centre, centre]), light=light,
                channels=channels)
            self._add_extras(out_extra, state, weights, light, records)
        self._driver.mem_copy_h2d(
            np.ascontiguousarray(
                records.spectra[:, None, :self._nregions], self._dtype),
            self._spectra)
        moments = self._moments.evaluate(
            (1.0, 1.0, self._step), (0.0, 0.0, self._zero), self._spectra,
            None)
        # The moments of the regions are vectors
        return {
            key: dict(d=value['d'][0], m=value['m'][0], w=value['w'][0])
            for key, value in moments.items()}

    def _outputs_of(
            self, state: 'OrbitLibraryState'
    ) -> tuple[OrbitIntegrator, _moments.MomentsPlan]:
        """
        Return the integrator of the extra outputs (of the bins, the pixels
        and the cells; made once) and the moments of the pixels.
        """
        if self._outputs is None:
            integrator = OrbitIntegrator(
                self._driver, self._dtype, len(state.starts),
                accuracy=state.accuracy, max_steps=state.max_steps(),
                nsamples=state.samples, nphi=state.copies, twins=True,
                cells=state.cell_light.shape,
                sky=[self._grid, self._pixels.grid],
                nchannels=self._nchannels, cell_moments=True)
            moments = _moments.MomentsPlan(
                self._driver, self._pixels.shape[::-1], self._orders,
                self._flux_floor, self._method, self._dtype)
            self._outputs = integrator, moments
        return self._outputs

    def _add_extras(
            self,
            out_extra: Extras,
            state: 'OrbitLibraryState',
            weights: np.ndarray,
            light: np.ndarray,
            records: 'OrbitRecords'
    ) -> None:
        """
        Add the extra outputs (see OrbitMomentsPlan) of the records of the
        bins, the pixels and the cells.
        """
        _, moments = self._outputs_of(state)
        cube = self._pixels.cube(records.spectra[:, self._nregions:])
        out_extra.add(
            'cube', gridutils.GridData(cube, self._cube_coords, 2),
            "Cube of the stars on orbits",
            "The spectra of the light of the stars on their orbits (at the "
            "solved weights) in each pixel of the grid of the bins, "
            "through the PSF, on the spectral axis of the moments.")
        d_cube = self._driver.mem_copy_h2d(
            np.ascontiguousarray(cube, self._dtype))
        measured = moments.evaluate(
            (1.0, 1.0, self._step), (0.0, 0.0, self._zero), d_cube, None)
        for key, value in measured.items():
            image = self._driver.mem_copy_d2h(value['d']).astype(float)
            mask = self._driver.mem_copy_d2h(value['m'])
            image[mask == 0] = np.nan
            out_extra.add(
                f'unbinned_{key}',
                gridutils.GridData(image, self._map_coords, None),
                f"Unbinned {key} of the stars on orbits",
                f"The {key} of the spectrum of each pixel of the cube of the "
                f"stars on orbits, measured as those of the bins (NaN where "
                f"masked).")
        cells, profile = _orbit_outputs.cell_tables(
            state, records.cell_moments)
        out_extra.add(
            'orbit_cells', cells, "Moments of the orbits in the cells",
            _orbit_outputs.CELLS_DESCRIPTION, **self._ids)
        out_extra.add(
            'orbit_profile', profile, "Profile of the moments of the orbits",
            _orbit_outputs.PROFILE_DESCRIPTION, **self._ids)
        out_extra.add(
            'orbits', _orbit_outputs.orbit_table(
                state, weights, light, records.energies[:, 0]),
            "Orbits", _orbit_outputs.ORBITS_DESCRIPTION, **self._ids)

    def mock(
            self,
            params: ParamValues,
            galaxies: Sequence[GalaxyState],
            noise: 'Noise',
            rng: np.random.Generator,
            ndraws: int
    ) -> tuple[list[HostModelData], dict[str, np.ndarray] | None]:
        raise NotImplementedError(
            "mock data of the moments of stars on orbits are not available "
            "yet: their spectra have no noise of recorded data")

    def has_exact_errors(self) -> bool:
        return False


class RegionMomentsPlan(_detail.DCubePlanBase):

    def __init__(
            self,
            moments: RegionMoments,
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
        nregions = self._sums.nregions()
        # The spectra of the regions, as a cube of one row of regions
        self._spectra = driver.mem_alloc_d(
            (dcube.size()[2], 1, nregions), dtype)
        self._moments = _moments.MomentsPlan(
            driver, (nregions, 1), moments.orders(), moments.flux_floor(),
            moments.method(), dtype)

    def _measure(self) -> ModelData:
        self._sums.evaluate(
            self._dcube_plan.dcube(), self._spectra[:, 0, :])
        moments = self._moments.evaluate(
            self._dcube.step(), self._dcube.zero(), self._spectra, None)
        # The moments of the regions are vectors
        return {
            key: dict(d=value['d'][0], m=value['m'][0], w=value['w'][0])
            for key, value in moments.items()}
