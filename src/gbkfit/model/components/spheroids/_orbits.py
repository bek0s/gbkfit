"""
The orbits of the stars of a spheroid in the gravity of its galaxy (of
its Schwarzschild models): the potential of the gravity as a table
(PotentialTable, see potential_table), and the integration of orbits in
it and their records, as they are seen in cells of the spheroid and in
targets of the sky (OrbitIntegrator, on the native kernels).
"""

import dataclasses
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy.interpolate
import scipy.sparse

from gbkfit.driver import DeviceArray, Driver
from ...gravity import GravityField


__all__ = [
    'OrbitIntegrator',
    'OrbitRecords',
    'PotentialTable',
    'SkyGrid',
    'potential_at',
    'potential_table'
]


@dataclasses.dataclass(frozen=True)
class PotentialTable:
    """
    The potential of the gravity of a galaxy in the frame of a spheroid,
    as the orbit kernels take it (see potential_table): on the radii r =
    exp(lnr_zero + i lnr_step) (arcsec) and the angles from the plane b =
    j pi / (2 (nb - 1)), the potential ((km/s)^2) and its derivatives along
    ln r, b and both: an array of shape (nr, nb, 4).
    """
    lnr_zero: float
    lnr_step: float
    values: np.ndarray


def potential_table(
        field: GravityField,
        kpc_per_arcsec: float,
        radii: tuple[float, float],
        nr: int,
        nb: int
) -> PotentialTable:
    """
    Return the potential of the gravity of a galaxy as a table (see
    PotentialTable), from its forces, on nr radii geometric between two
    (arcsec) and nb angles from the plane (2 for spherical gravity).

    The derivatives along ln r and b are those of the forces at the nodes,
    that along both a cubic spline of the second along ln r. The potential
    is their integral by cubic splines: from the last radius in the plane,
    where it is that of a point of the force there (0 at infinity, as the
    kernels take it beyond the table), along b, then inwards along ln r.

    Parameters
    ----------
    field : GravityField
        The gravity, in kpc (its forces off the plane).
    kpc_per_arcsec : float
        The kpc in an arcsec at the distance of the galaxy.
    radii : tuple of float
        The first and the last radius of the table (arcsec).
    nr, nb : int
        The number of the radii and of the angles, at least 2 each.

    Returns
    -------
    PotentialTable
        The table.
    """
    if nr < 2 or nb < 2:
        raise ValueError("a potential table needs 2 radii and 2 angles")
    lnr = np.linspace(np.log(radii[0]), np.log(radii[1]), nr)
    b = np.linspace(0, np.pi / 2, nb)
    r = np.exp(lnr)[:, None]
    k = kpc_per_arcsec
    # (the forces per arcsec)
    force_R, force_z = (
        force * k for force in field.forces(
            r * np.cos(b) * k, r * np.sin(b) * k))
    phi_u = -r * (force_R * np.cos(b) + force_z * np.sin(b))
    phi_b = r * (force_R * np.sin(b) - force_z * np.cos(b))
    # (0 in the plane and on the axis, by symmetry)
    phi_b[:, [0, -1]] = 0
    phi_ub = scipy.interpolate.CubicSpline(lnr, phi_b, axis=0)(lnr, 1)
    outer = -phi_u[-1, 0] + scipy.interpolate.CubicSpline(
        b, phi_b[-1]).antiderivative()(b)
    inward = scipy.interpolate.CubicSpline(
        lnr, phi_u, axis=0).antiderivative()
    phi = outer - (inward(lnr[-1]) - inward(lnr))
    return PotentialTable(
        float(lnr[0]), float(lnr[1] - lnr[0]),
        np.stack((phi, phi_u, phi_b, phi_ub), axis=-1))


@dataclasses.dataclass(frozen=True)
class SkyGrid:
    """
    A grid of cells of the sky and the targets that a point in each cell
    gives its light to: the map of the sky (arcsec, of the frame of the
    geometry of the galaxy) to the cells, an array of shape (2, 3) (a
    point (x, y) is in the cell (floor(cx), floor(cy)) of (cx, cy) = cells
    @ (x, y, 1)), its shape (ny, nx), and shares, a sparse array of shape
    (ny nx, ntargets): the share of each target in the light of a point in
    cell (x, y), at row y nx + x (e.g. of the Voronoi bins of an
    observation, after its PSF).
    """
    cells: np.ndarray
    shape: tuple[int, int]
    shares: scipy.sparse.csr_array


@dataclasses.dataclass(frozen=True)
class OrbitRecords:
    """
    The records of orbits (see OrbitIntegrator.integrate), each optional:
    the time fraction of each orbit in each cell (an array of shape
    (norbits, ncell_r, ncell_b)); its moments in each target of the sky
    (of shape (norbits, ntargets, order + 2): its light, and gamma h_n for
    n in [0, order] of its velocities, see the native OrbitArgs); the
    spectrum of the light of the orbits in each target (of shape
    (nchannels, ntargets)); and the steps of the integration of each
    orbit and its energy ((km/s)^2) at its start and its end (of shape
    (n, 2)). With twins, orbit 2k + 1 of the cells and the moments is
    orbit 2k reversed in time, both of start k.
    """
    cells: np.ndarray | None
    moments: np.ndarray | None
    spectra: np.ndarray | None
    steps: np.ndarray
    energies: np.ndarray


class OrbitIntegrator:
    """
    The integration of orbits in the potential of a galaxy and their
    records (see the native OrbitArgs), on the device of a driver: each
    orbit for some turns of the circular orbit of its energy, in
    time-symmetric leapfrog steps of the given accuracy (each the radians
    of the circular orbit of its radius, and of its crossing time), at
    most max_steps of them, recorded at nsamples times, on the sky in 2
    nphi copies (turned about the axis and mirrored about the plane), and
    with twins reversed in time too. It records the time of each orbit in
    cells, and in the targets of the sky its moments in their series
    (with an order) or the spectrum of all their light (with channels),
    or both.

    Parameters
    ----------
    driver : Driver
        The driver.
    dtype : np.dtype
        The precision of its arrays and of the integration.
    norbits : int
        The number of the orbits integrated.
    accuracy : float
        The accuracy of the steps (radians).
    max_steps : int
        The most steps of an orbit.
    nsamples : int
        The number of the records of an orbit in time.
    nphi : int
        The number of its turned copies on the sky.
    twins : bool
        Whether each orbit is also recorded reversed in time.
    cell_edges : tuple of np.ndarray, optional
        The edges of the cells, increasing: radii (arcsec) and angles from
        the plane (radians, within [0, pi / 2]).
    sky : sequence of SkyGrid, optional
        The grids of the sky, their targets one after the other.
    order : int, optional
        The highest order of the Gauss-Hermite series of the targets, for
        the moments of the orbits.
    nchannels : int, optional
        The number of the channels of the spectra of the targets.

    Raises
    ------
    ValueError
        If the shares of a grid are not one row per cell, or a sky goes
        without moments or spectra.
    """

    def __init__(
            self,
            driver: Driver,
            dtype: npt.DTypeLike,
            norbits: int,
            *,
            accuracy: float,
            max_steps: int,
            nsamples: int,
            nphi: int,
            twins: bool,
            cell_edges: tuple[np.ndarray, np.ndarray] | None = None,
            sky: Sequence[SkyGrid] = (),
            order: int | None = None,
            nchannels: int | None = None
    ):
        if bool(sky) != (order is not None or nchannels is not None):
            raise ValueError(
                "a sky goes with the moments of the orbits or their spectra")
        self._driver = driver
        self._dtype = np.dtype(dtype)
        self._orbits = driver.native_class('Orbits', dtype)
        self._norbits = norbits
        self._options = dict(
            accuracy=accuracy, max_steps=max_steps, nsamples=nsamples,
            nphi=nphi, twins=twins)
        self._order = order
        recorded = norbits * (2 if twins else 1)

        def device(
                array: npt.ArrayLike, dtype: npt.DTypeLike
        ) -> DeviceArray:
            return driver.mem_copy_h2d(np.ascontiguousarray(array, dtype))

        self._cell_edges = None
        self._cells = None
        if cell_edges is not None:
            self._cell_edges = tuple(
                device(edges, dtype) for edges in cell_edges)
            self._cells = driver.mem_alloc_d(
                (recorded, len(cell_edges[0]) - 1, len(cell_edges[1]) - 1),
                dtype)

        self._sky = None
        self._grids = ()
        self._moments = None
        self._spectra = None
        if sky:
            shares = [scipy.sparse.csr_array(grid.shares) for grid in sky]
            if any(s.shape[0] != grid.shape[0] * grid.shape[1]
                   for s, grid in zip(shares, sky)):
                raise ValueError(
                    "the shares of a sky grid need a row for each cell")
            first_cells = np.cumsum([0] + [len(s.indptr) for s in shares])
            first_entries = np.cumsum([0] + [s.nnz for s in shares])
            first_targets = np.cumsum([0] + [s.shape[1] for s in shares])
            ntargets = int(first_targets[-1])
            self._grids = tuple(np.asarray(grid.cells, float) for grid in sky)
            self._sky = dict(
                grid_sizes=device(
                    [grid.shape[::-1] for grid in sky], np.int32),
                grid_cells=device(first_cells[:-1], np.int64),
                offsets=device(np.concatenate([
                    s.indptr + first for s, first
                    in zip(shares, first_entries)]), np.int64),
                targets=device(np.concatenate([
                    s.indices + first for s, first
                    in zip(shares, first_targets)]), np.int32),
                shares=device(
                    np.concatenate([s.data for s in shares]), dtype))
            if order is not None:
                self._moments = driver.mem_alloc_d(
                    (recorded, ntargets, order + 2), dtype)
            if nchannels is not None:
                self._spectra = driver.mem_alloc_d(
                    (nchannels, ntargets), dtype)

        self._steps = driver.mem_alloc_d((norbits,), np.int32)
        self._energies = driver.mem_alloc_d((norbits, 2), dtype)

    def integrate(
            self,
            table: PotentialTable,
            starts: np.ndarray,
            turns: float,
            geometry: tuple[float, float],
            centres: np.ndarray | None = None,
            series: tuple[np.ndarray, np.ndarray] | None = None,
            light: np.ndarray | None = None,
            channels: tuple[float, float] | None = None
    ) -> OrbitRecords:
        """
        Integrate the orbits and record them.

        Parameters
        ----------
        table : PotentialTable
            The potential.
        starts : np.ndarray
            The start of each orbit, an array of shape (norbits, 6): x, y,
            z (arcsec) and vx, vy, vz (km/s), in the frame of the
            spheroid.
        turns : float
            How long each orbit is integrated, in periods of the circular
            orbit of its energy in the plane.
        geometry : tuple of float
            The position angle and the inclination of the spheroid
            (degrees).
        centres : np.ndarray, optional
            The centre of the spheroid in the sky of each grid (an array
            of shape (ngrids, 2): x, y), with a sky only.
        series : tuple of np.ndarray, optional
            The mean and the dispersion (km/s) of the Gauss-Hermite series
            of each target of the sky, with its moments only.
        light : np.ndarray, optional
            The light of each orbit (of each twin after its orbit), with
            spectra only.
        channels : tuple of float, optional
            The map of the velocities (km/s, relative to that of the
            centre) to the channels of the spectra, with spectra only: a
            velocity v is in the channel floor(channels[0] v +
            channels[1]).

        Returns
        -------
        OrbitRecords
            The records, on the host.
        """
        driver, dtype = self._driver, self._dtype
        if (centres is None) != (self._sky is None) \
                or (series is None) != (self._moments is None) \
                or (light is None or channels is None) \
                != (self._spectra is None):
            raise ValueError(
                "the centres of the grids go with a sky, the series of the "
                "targets with moments, and the light and the channels with "
                "spectra")
        if len(starts) != self._norbits:
            raise ValueError(
                f"the integrator has {self._norbits} orbits; it is given "
                f"{len(starts)} starts")

        def device(array: npt.ArrayLike) -> DeviceArray:
            return driver.mem_copy_h2d(np.ascontiguousarray(array, dtype))

        sky = dict.fromkeys((
            'grid_sizes', 'grids', 'grid_cells', 'offsets', 'targets',
            'shares', 'target_v', 'target_s', 'light'))
        if self._sky is not None:
            # (the maps of the grids from the centre of the spheroid)
            grids = [
                np.column_stack((
                    cells[:, :2], cells[:, 2] + cells[:, :2] @ centre))
                for cells, centre in zip(self._grids, centres)]
            sky |= self._sky | dict(
                grids=device([grid.ravel() for grid in grids]))
        if series is not None:
            sky |= dict(
                target_v=device(series[0]), target_s=device(series[1]))
        if light is not None:
            sky |= dict(light=device(light))
            driver.mem_fill(self._spectra, 0)
        cell_r, cell_b = self._cell_edges or (None, None)
        self._orbits.integrate(
            table=device(table.values),
            range=device([table.lnr_zero, table.lnr_step]),
            starts=device(starts), turns=turns,
            **self._options, posa=geometry[0], incl=geometry[1],
            cell_r=cell_r, cell_b=cell_b, cells=self._cells,
            **sky, order=self._order or 0, moments=self._moments,
            channel_scale=0 if channels is None else channels[0],
            channel_offset=0 if channels is None else channels[1],
            spectra=self._spectra,
            steps=self._steps, energies=self._energies)

        def host(array: DeviceArray | None) -> np.ndarray | None:
            return None if array is None else driver.mem_copy_d2h(array)
        return OrbitRecords(
            cells=host(self._cells), moments=host(self._moments),
            spectra=host(self._spectra), steps=host(self._steps),
            energies=host(self._energies))


def potential_at(
        driver: Driver,
        dtype: npt.DTypeLike,
        table: PotentialTable,
        points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the potential of a table ((km/s)^2) at points (an array of
    shape (n, 3): x, y, z, arcsec, in the frame of the spheroid), as the
    orbit kernels take it, and the forces there ((km/s)^2 / arcsec; an
    array of shape (n, 3)).
    """
    def device(array: npt.ArrayLike) -> DeviceArray:
        return driver.mem_copy_h2d(np.ascontiguousarray(array, dtype))
    values = driver.mem_alloc_d((len(points),), dtype)
    forces = driver.mem_alloc_d((len(points), 3), dtype)
    driver.native_class('Orbits', dtype).potential(
        device(table.values), device([table.lnr_zero, table.lnr_step]),
        device(points), values, forces)
    return driver.mem_copy_d2h(values), driver.mem_copy_d2h(forces)
