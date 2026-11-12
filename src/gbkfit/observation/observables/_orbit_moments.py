"""
The moments of stars on orbits whose weights are solved, in bins (see
OrbitMomentsPlan), and the check of the observations that see them (see
check_orbits).
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from gbkfit.driver import Driver
from gbkfit.grid import Coords, Grid, GridData
from gbkfit.model import Extras, GalaxyState
from gbkfit.model.components.spheroids import OrbitIntegrator
from gbkfit.params import ParamValues
from . import _moments, _orbit_outputs
from ._orbit_sky import PixelCube, bins_sky_grid
from .base import HostModelData, ModelData, ObservablePlan, ObservationSetup

if TYPE_CHECKING:
    from gbkfit.model import Scene
    from gbkfit.model.components.spheroids import (
        OrbitLibraryState, OrbitRecords)
    from ..noise import Noise
    from .region_moments import RegionMoments


def check_orbits(
        moments: 'RegionMoments',
        scene: 'Scene',
        setup: ObservationSetup
) -> None:
    """
    Raise ValueError unless moments of regions can see stars on orbits
    whose weights are solved (see OrbitMomentsPlan) in a scene, as an
    observation of the given setup sees them (see Observable.check_scene):
    only them, in bins, of the method gauss_hermite and the orders 1 to M
    (V, sigma, h3 ... hM), without a flux scale; through a PSF that does
    not vary, without an LSF (pPXF takes it out of their kinematics), a
    primary beam, a lens or dust, which the recording of the orbits does
    not apply.
    """
    instrument = setup.instrument
    # (refused here, before the plan: it ignored them, or failed on a
    # varying PSF before the objective refused it)
    for applies, what in (
            (instrument.lsf() is not None,
             "an LSF (pPXF takes it out of their kinematics)"),
            (instrument.psf() is not None and instrument.psf().varies(),
             "a PSF that varies"),
            (instrument.primary_beam() is not None, "a primary beam"),
            (setup.foreground.lens() is not None, "a lens"),
            (scene.has_dust(), "dust in the galaxies it sees")):
        if applies:
            raise ValueError(
                f"moments of stars on orbits whose weights are solved "
                f"are seen without {what}: their recording does not "
                f"apply it")
    if sum(len(emitters) for emitters in scene.emitters()) != 1 \
            or len(scene.orbit_emitters()) != 1:
        raise ValueError(
            "moments of regions that see stars on orbits whose weights "
            "are solved (e.g. of a spheroid of Schwarzschild kinematics) "
            "see only them, of one component: give the observation that "
            "component or the lines of its stars only")
    if moments.regions().grid() is None:
        raise ValueError(
            "moments of stars on orbits whose weights are solved are of "
            "bins of the pixels of a grid (e.g. Voronoi bins), not of "
            "apertures")
    orders = moments.orders()
    if moments.method() != 'gauss_hermite' or len(orders) < 2 \
            or orders != tuple(range(1, len(orders) + 1)):
        raise ValueError(
            f"moments of stars on orbits whose weights are solved are "
            f"Gauss-Hermite series (method gauss_hermite) of the orders "
            f"1 to M (V, sigma, h3 ... hM); they are of the method "
            f"{moments.method()!r} and the orders {list(orders)}")
    if setup.flux_scale_param is not None:
        raise ValueError(
            "moments of stars on orbits whose weights are solved have "
            "no flux scale")


class OrbitMomentsPlan(ObservablePlan):
    """
    The moments of the bins of stars on orbits whose weights are solved
    (see Component.orbit_weights), at given weights: the orbits of the
    state of their spheroid (see OrbitLibraryState), integrated and
    recorded with the light of their weights in the bins, through the PSF
    (see bins_sky_grid), as the spectrum of each bin on the spectral axis
    of the moments, whose moments are measured as those of the spectra of
    any regions (see RegionMomentsPlan). They are the model
    of an observation without data; that of an observation with data is
    of the orbits in the series of its data (see OrbitRecording), which an
    objective fits.

    Its mock data are the moments of the bins of the cube of the pixels
    of their grid plus the noise (see mock), as those of RegionMomentsPlan,
    with data or without.

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
    driver : Driver
        The driver.
    scene : Scene
        What they see: the stars of one component, on orbits.
    setup : ObservationSetup
        The setup of their observation: its instrument, the oversampling
        of its model (along x and y) and its precision.
    """

    def __init__(
            self,
            moments: 'RegionMoments',
            driver: Driver,
            scene: 'Scene',
            setup: ObservationSetup
    ):
        instrument, oversampling = setup.instrument, setup.oversampling
        dtype = setup.dtype
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
        self._grid = bins_sky_grid(
            moments.regions(), instrument, oversampling)
        self._driver = driver
        self._dtype = dtype
        nregions = moments.regions().nregions()
        self._nregions = nregions
        # (the spectral axis: its channels, their width, and the velocity
        # of the centre of the first)
        spectral = moments.spectral()
        self._nchannels = spectral.size()
        self._step = spectral.step()
        self._zero = spectral.rval() - spectral.rpix() * self._step
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
        self._pixels = PixelCube(
            grid, shape, instrument, oversampling)
        coords = grid.coords
        self._cube_coords = Coords(
            coords.step[:2] + (self._step,),
            coords.rpix[:2] + (spectral.rpix(),),
            coords.rval[:2] + (spectral.rval(),), coords.rota,
            spectral.rest(), spectral.medium(), spectral.convention())
        self._map_coords = coords
        self._cube_grid = Grid(
            (shape[1], shape[0], self._nchannels), self._cube_coords, 2)
        self._bin_weights = moments.regions().weights(grid)
        self._psf = instrument.psf()
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
        records, weights, light = self._record(
            params, state, out_extra is not None)
        if out_extra is not None:
            self._add_extras(out_extra, state, weights, light, records)
        return self._measure(records.spectra[:, :self._nregions])

    def _record(
            self,
            params: ParamValues,
            state: 'OrbitLibraryState',
            pixels: bool
    ) -> tuple['OrbitRecords', np.ndarray, np.ndarray]:
        """
        Return the records of the orbits of a state at the values of the
        parameters: the spectra of the bins, and with pixels, also those of
        the pixels and the moments of the cells (see _outputs_of); and the
        weights and the light of the orbits.
        """
        values = {name: params[full] for name, full in self._names.items()}
        weights = np.asarray(values[self._weights], dtype=float)
        light = state.orbit_light(weights)
        centre = [values['xpos'], values['ypos']]
        geometry = (values['posa'], values['incl'])
        # (channel j holds the velocities vsys + v from zero + (j - 1/2)
        # step to zero + (j + 1/2) step)
        channels = (1 / self._step,
                    (values['vsys'] - self._zero) / self._step + 0.5)
        if pixels:
            integrator, _ = self._outputs_of(state)
            records = integrator.integrate(
                state.potential, state.starts, state.turns, geometry,
                cell_edges=state.cell_edges,
                centres=np.array([centre, centre]), light=light,
                channels=channels)
            return records, weights, light
        if self._integrator is None:
            # (in float64, see _orbits.OrbitRecording)
            self._integrator = OrbitIntegrator(
                self._driver, np.float64, len(state.starts),
                tolerance=state.tolerance, max_steps=state.max_steps(),
                nsamples=state.samples, nphi=state.copies, twins=True,
                sky=[self._grid], nchannels=self._nchannels)
        records = self._integrator.integrate(
            state.potential, state.starts, state.turns, geometry,
            centres=np.array([centre]), light=light, channels=channels)
        return records, weights, light

    def _measure(self, spectra: np.ndarray) -> ModelData:
        """
        Return the moments of the spectra of the bins (an array of shape
        (nchannels, nbins)), on the driver.
        """
        self._driver.mem_copy_h2d(
            np.ascontiguousarray(spectra[:, None, :], self._dtype),
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
                self._driver, np.float64, len(state.starts),
                tolerance=state.tolerance, max_steps=state.max_steps(),
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
            'cube', GridData(cube, self._cube_coords, 2),
            "Cube of the stars on orbits",
            "The spectra of the light of the stars on their orbits (at the "
            "solved weights) in each pixel of the grid of the bins, "
            "through the PSF, on the spectral axis of the moments: the light "
            "of the spheroid per arcsec^2 in each channel, as the model "
            "cubes of other stars.")
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
                GridData(image, self._map_coords, None),
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
        # (the cube of the pixels, plus the noise of each draw, summed over
        # the pixels of each bin: as RegionMomentsPlan does)
        state = galaxies[self._galaxy].states[self._component]
        records, _, _ = self._record(params, state, True)
        recorded = self._pixels.cube(records.spectra[:, self._nregions:])
        draws = []
        for _ in range(ndraws):
            noisy = recorded + noise.draw(
                rng, self._cube_grid, self._psf, recorded)
            measured = self._measure(
                (self._bin_weights @ noisy.reshape(len(noisy), -1).T).T)
            draws.append({
                key: dict(
                    d=self._driver.mem_copy_d2h(value['d']),
                    m=self._driver.mem_copy_d2h(value['m']))
                for key, value in measured.items()})
        return draws, None

    def has_exact_errors(self) -> bool:
        return False
