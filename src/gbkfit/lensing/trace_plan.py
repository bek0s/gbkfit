"""
The rays of an observation through the lenses (see TracePlan): where
the rays of the pixels of its data cross each plane behind lenses, on
the grid it draws the plane on.
"""

import logging
from collections.abc import Mapping, Sequence

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.grid import Grid, sky_positions, sky_to_pixel
from gbkfit.model import Reference
from gbkfit.model.mass import SkyMass
from gbkfit.params import (
    ModelEvaluationError, ParamValues, ParamView, StageCache)
from ._mass_deflections import MassBending, MassDeflections
from .pieces import SkyPoints
from .source_grids import SourceGrid, source_step
from .tracer import Tracer, stretch


__all__ = [
    'TracePlan'
]


_log = logging.getLogger(__name__)


class TracePlan:
    """
    The rays of an observation through the lenses (see Tracer), on a
    driver: where the rays of the pixels of its footprint (those whose
    light can reach its data) cross each plane behind lenses that it
    draws, in the pixels of the grid of the plane (see SourceGrid), laid
    out at reference values of the parameters.

    The lenses are on the sky, in the frame of the parameters of the
    galaxies (the group frame), and the observation sees them as it sees
    the galaxies, moved by the shift of its frame: its alignment and its
    offsets, if it fits them (see ObservationGroup). The rays are traced
    from the pixels moved back by the shift, and their points on each
    plane moved by it again, into the frame of the observation, where its
    galaxies are drawn.

    The deflection of the lenses of each plane is evaluated again only
    when the values of their parameters, or the offsets, change, a plane
    in front changes it, or the mass on the sky of a galaxy whose mass
    they are (see LensGalaxyMass) is not that of its last evaluation (the
    stage of a galaxy lays it out again only when what it reads changes,
    see GalaxyStage); that of fixed lenses once. The mass of a galaxy is
    then deflected again layer by layer, only where the unit shape of a
    layer, or the rays, changed (see MassDeflections), by the native
    kernels of its driver (the rays copied to its device and their
    deflections back): a change of a mass scale alone (e.g. a
    mass-to-light ratio) only scales its layers.

    Parameters
    ----------
    tracer : Tracer
        The paths of the light of the planes.
    lens_names : Sequence of Mapping
        The names of the parameters of each lens of the tracer among those
        of the evaluations, by its own names (see Lens.pdescs).
    planes : Sequence of int
        The planes (by index in the tracer) whose rays it gives, near to
        far: those whose light is bent that the observation draws.
    grid : Grid
        The high-resolution grid of the observation (its x and y axes).
    footprint : np.ndarray
        The pixels of the grid whose light can reach the data, a bool
        array of shape (ny, nx).
    origin : tuple of float
        The RA and Dec (degrees) of the origin of the group frame.
    alignment : tuple of float
        Where the origin of the group frame is in the frame of the
        observation (arcsec, x and y).
    offsets : tuple of (str or None)
        The names of the parameters of the offsets of the frame of the
        observation along x and y, None for those it does not fit.
    reference : Reference
        The values of the parameters (e.g. ParamSpace.reference) at which
        the grids of the planes are laid out, and there the masses on the
        sky of the galaxies whose masses the lenses are.
    options : SourceGrid
        How the grids are laid out.
    driver : Driver
        The driver of the observation.
    dtype : np.dtype
        The floating type of its model.

    Raises
    ------
    ValueError
        If no plane is given, a plane is not behind a lens, the footprint
        is not one of the grid or has no pixels, or the lenses have no
        deflection at the reference values (e.g. rays off a map).
    """

    def __init__(
            self,
            tracer: Tracer,
            lens_names: Sequence[Mapping[str, str]],
            planes: Sequence[int],
            grid: Grid,
            footprint: np.ndarray,
            origin: tuple[float, float],
            alignment: tuple[float, float],
            offsets: tuple[str | None, str | None],
            reference: Reference,
            options: SourceGrid,
            driver: Driver,
            dtype: np.dtype
    ):
        planes = tuple(planes)
        if not planes:
            raise ValueError("a trace plan needs planes whose light is bent")
        if unlensed := [q for q in planes if not tracer.lensed(q)]:
            raise ValueError(
                f"the light of planes {unlensed} is not bent: no lens is in "
                f"front of them")
        spatial = grid if grid.spectral_axis is None else grid.spatial()
        footprint = np.asarray(footprint, dtype=bool)
        if footprint.shape != spatial.size[::-1]:
            raise ValueError(
                f"the footprint must be of the shape of the grid "
                f"{spatial.size[::-1]}; it is of {footprint.shape}")
        if not footprint.any():
            raise ValueError("the footprint has no pixels")
        self._tracer = tracer
        self._lens_names = tuple(dict(names) for names in lens_names)
        self._planes = planes
        self._spatial = spatial
        self._footprint = footprint
        x, y = sky_positions(spatial)
        self._points = (x[footprint], y[footprint])
        self._origin = tuple(origin)
        self._alignment = tuple(alignment)
        self._offsets = tuple(offsets)
        self._driver = driver
        self._dtype = np.dtype(dtype)
        # The galaxies whose masses the lenses of each plane are, by name
        self._galaxies = {
            p: tuple(
                name for i in plane.lenses
                for name in tracer.lenses()[i].galaxies())
            for p, plane in enumerate(tracer.planes())}
        self._grids = self._layout(reference, options)
        self._init_caches()
        # The positions of the rays on each plane, in the pixels of its
        # grid, on the host (NaN off the footprint) and on the device; and
        # the last rays, in the frame of the observation
        shape = footprint.shape
        self._host = {q: np.full((2,) + shape, np.nan) for q in planes}
        self._device = {
            q: (driver.mem_alloc_d(shape, dtype),
                driver.mem_alloc_d(shape, dtype))
            for q in planes}
        self._rays: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def planes(self) -> tuple[int, ...]:
        """Return its planes, by index in the tracer."""
        return self._planes

    def grid(self, plane: int) -> Grid:
        """
        Return the grid of a plane, laid out at the reference values.

        Parameters
        ----------
        plane : int
            The plane, by index in the tracer.

        Returns
        -------
        Grid
            Its x and y axes, in the frame of the observation.
        """
        return self._grids[plane]

    def positions(
            self,
            params: ParamValues,
            masses: Mapping[str, SkyMass]
    ) -> dict[int, tuple[DeviceArray, DeviceArray]]:
        """
        Return the positions of the rays on each of its planes at the
        values of the parameters, in the pixels of its grid (see grid),
        each an array (ny, nx) on the device, of x and of y, NaN off the
        footprint (see lens_composite of the native module DModel); each
        written again only if it changed (see the class).

        Parameters
        ----------
        params : ParamValues
            The values of the parameters (those it reads among them).
        masses : Mapping of str to SkyMass
            The mass on the sky of each galaxy whose mass the lenses are
            (see Lens.galaxies), by name.

        Returns
        -------
        dict
            The positions on each plane, by index in the tracer.

        Raises
        ------
        ModelEvaluationError
            If the lenses have no deflection at a ray (e.g. off a map).
        """
        deflections = {}
        for p, cache in self._lens_caches.items():
            held = tuple(masses[name] for name in self._galaxies[p])

            def deflect(view: ParamView, p: int = p) -> tuple[
                    np.ndarray, np.ndarray]:
                return self._deflect(p, view, deflections, masses)
            # (the masses of its last evaluation are held: their objects
            # are never those of others)
            last = self._held.get(p)
            if last is not None and all(
                    new is old for new, old in zip(held, last, strict=True)):
                deflections[p] = cache.evaluate(params, deflect)
            else:
                deflections[p] = cache.refresh(params, deflect)
                self._held[p] = held
        for q, cache in self._placements.items():
            self._rays[q] = cache.evaluate(
                params, lambda view, q=q: self._place(q, view, deflections))
        return dict(self._device)

    def rays(self, plane: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the positions of the rays on a plane at the last evaluation
        (see positions).

        Parameters
        ----------
        plane : int
            The plane, by index in the tracer.

        Returns
        -------
        tuple of np.ndarray
            The positions along x and y (arcsec, in the frame of the
            observation) of the ray of each pixel of its high-resolution
            grid, each of shape (ny, nx), NaN off the footprint.
        """
        x, y = self._rays[plane]
        return x.copy(), y.copy()

    def magnification(self, plane: int) -> np.ndarray:
        """
        Return the magnification of the image of a plane at the last
        evaluation (see positions): 1 / det of the Jacobian of its rays
        against the pixels of the image, by central differences between
        the pixels of the high-resolution grid.

        Parameters
        ----------
        plane : int
            The plane, by index in the tracer.

        Returns
        -------
        np.ndarray
            The magnification at each pixel, of shape (ny, nx): signed
            (negative where the image is mirrored), NaN off the footprint
            and at its edges.
        """
        x, y = self._rays[plane]
        dx_dj, dx_di = np.gradient(x)
        dy_dj, dy_di = np.gradient(y)
        # (the pixels of the image span step_x step_y arcsec^2, whatever
        # their rotation)
        step_x, step_y = self._spatial.coords.step[:2]
        with np.errstate(divide='ignore', invalid='ignore'):
            return step_x * step_y / (dx_di * dy_dj - dx_dj * dy_di)

    def off_grid(self, plane: int) -> int:
        """
        Return the number of rays of the footprint that crossed a plane
        off its grid at the last evaluation (see positions): whose light,
        if any, was lost.

        Parameters
        ----------
        plane : int
            The plane, by index in the tracer.

        Returns
        -------
        int
            The number of rays.
        """
        px, py = self._host[plane][:, self._footprint]
        size_x, size_y = self._grids[plane].size
        return int(np.sum(
            (px < 0) | (px > size_x - 1) | (py < 0) | (py > size_y - 1)))

    def _shift(self, view: ParamView | ParamValues) -> tuple[float, float]:
        """
        Return the shift of the frame of the observation (see the class):
        its alignment and its offsets.
        """
        return tuple(
            align + (0.0 if name is None else float(view[name]))
            for align, name in zip(self._alignment, self._offsets))

    def _group_points(self, shift: tuple[float, float]) -> SkyPoints:
        """
        Return the points of the footprint in the group frame, moved back
        by the shift of the frame of the observation.
        """
        x, y = self._points
        return SkyPoints(x - shift[0], y - shift[1], self._origin)

    def _values(
            self, params: ParamView | ParamValues, lenses: Sequence[int]
    ) -> list[dict[str, float]]:
        """
        Return the values of the parameters of each lens of the tracer by
        its own names: those of the given lenses, and none of the others.
        """
        values = [{} for _ in self._lens_names]
        for i in lenses:
            values[i] = {
                own: params[full] for own, full in self._lens_names[i].items()}
        return values

    def _layout(
            self, reference: Reference, options: SourceGrid
    ) -> dict[int, Grid]:
        """
        Return the grid of each of its planes, laid out at the reference
        values, with the masses there (see SourceGrid).

        Raises
        ------
        ValueError
            If the reference has no mass of a galaxy whose mass a lens is.
        """
        tracer = self._tracer
        if missing := sorted(
                {name for names in self._galaxies.values() for name in names}
                - set(reference.masses)):
            raise ValueError(
                f"the lenses bend light by the masses of the galaxies "
                f"{missing}, which the reference lacks")
        shift = self._shift(reference.values)
        points = self._group_points(shift)
        lenses = [i for plane in tracer.planes() for i in plane.lenses]
        values = self._values(reference.values, lenses)
        # (deflected once each: their layers kept for no evaluation)
        galaxies = {
            name: MassBending(mass, MassDeflections(self._driver))
            for name, mass in reference.masses.items()}
        substep = float(min(np.abs(self._spatial.coords.step[:2])))
        grids = {}
        try:
            rays = tracer.trace(
                points, values, self._driver, max(self._planes), galaxies)
            for q in self._planes:
                step = options.step()
                if step == 'auto':
                    step = source_step(substep, stretch(tracer.jacobian(
                        q, points, values, substep / 2, self._driver,
                        galaxies)))
                grids[q] = options.layout(
                    rays[q].x + shift[0], rays[q].y + shift[1], step,
                    self._spatial.coords.rval)
        except ModelEvaluationError as e:
            raise ValueError(
                f"at the reference values of the parameters, which the "
                f"grids of the planes behind lenses are laid out at: "
                f"{e}") from e
        for q, grid in grids.items():
            _log.info(
                f"plane {q} (z = {tracer.planes()[q].redshift:g}): a grid "
                f"of {grid.size[0]} x {grid.size[1]} pixels of "
                f"{grid.coords.step[0]:.4g} arcsec")
        return grids

    def _init_caches(self) -> None:
        """
        Make the caches of the deflection of each lens plane in front of
        its planes (with the masses of galaxies of their last evaluation,
        and where the deflections of their layers are kept), and of the
        positions of the rays on each of them.
        """
        tracer = self._tracer
        farthest = max(self._planes)
        offsets = tuple(name for name in self._offsets if name is not None)
        self._held: dict[int, tuple[SkyMass, ...]] = {}
        self._kept = {
            p: {name: MassDeflections(self._driver)
                for name in self._galaxies[p]}
            for p in range(farthest)}
        self._lens_caches: dict[int, StageCache] = {}
        for p in range(farthest):
            lenses = tracer.planes()[p].lenses
            if not lenses:
                continue
            reads = [
                name for i in lenses
                for name in self._lens_names[i].values()]
            self._lens_caches[p] = StageCache(
                reads + list(offsets), tuple(self._lens_caches.values()))
        self._placements = {
            q: StageCache(offsets, tuple(
                cache for p, cache in self._lens_caches.items() if p < q))
            for q in self._planes}

    def _deflect(
            self,
            plane: int,
            view: ParamView,
            deflections: Mapping[int, tuple[np.ndarray, np.ndarray]],
            masses: Mapping[str, SkyMass]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the deflection of the lenses of a lens plane at the rays of
        the footprint there, of the deflections of the planes in front and
        the masses on the sky of the galaxies whose masses they are.
        """
        tracer = self._tracer
        ray = tracer.ray(
            plane, self._group_points(self._shift(view)), deflections)
        galaxies = {
            name: MassBending(masses[name], kept)
            for name, kept in self._kept[plane].items()}
        return tracer.deflection(
            plane, ray, self._values(view, tracer.planes()[plane].lenses),
            self._driver, galaxies)

    def _place(
            self,
            plane: int,
            view: ParamView,
            deflections: Mapping[int, tuple[np.ndarray, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Write the positions of the rays on a plane, in the pixels of its
        grid, to the device, of the deflections of the lens planes in
        front; and return the rays (see rays).
        """
        shift = self._shift(view)
        ray = self._tracer.ray(plane, self._group_points(shift), deflections)
        x = ray.x + shift[0]
        y = ray.y + shift[1]
        matrix, offset = sky_to_pixel(self._grids[plane])
        host = self._host[plane]
        host[0][self._footprint] = matrix[0, 0] * x + matrix[0, 1] * y \
            + offset[0]
        host[1][self._footprint] = matrix[1, 0] * x + matrix[1, 1] * y \
            + offset[1]
        for values, device in zip(host, self._device[plane]):
            self._driver.mem_copy_h2d(values.astype(self._dtype), device)
        rays = np.full((2,) + self._footprint.shape, np.nan)
        rays[0][self._footprint] = x
        rays[1][self._footprint] = y
        return rays[0], rays[1]
