"""
Particles given in a table (e.g. the gas of a simulation), for mock data
of them (see Particles).
"""

import os.path
from typing import Any, Literal

import astropy.table
import numpy as np
import scipy.special

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamScalarDesc, ParamValues
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from ._detail import (
    Fluxes, disk_to_sky, dump_geometry, dump_name, geometry_desc)
from ._units import DataUnit
from .base import (
    Component, ComponentPlan, ComponentState, FieldsLayout, FrameParams,
    NativeGrid)
from .geometries import Geometry
from .lines import Band, LineCatalogue, LinePlacement
from .points import band_light, line_light
from .roles import BandEmission, Emission, LineEmission, emission_parser


__all__ = [
    'Particles'
]


# The columns of the table of particles, and the optional one
_COLUMNS = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'flux')
_SIGMA = 'sigma'

# The particles evaluated at once (to bound the memory of their spectra)
_CHUNK = 2 ** 16

# The half-width of the window of channels of the line of a particle, in
# its dispersions: the light beyond it (6e-7 of it) is left out
_LINE_WINDOW = 5


def read_particles(file: str) -> dict[str, np.ndarray]:
    """
    Read a table of particles (see Particles): a FITS table, an ECSV file
    or a numpy file (npz) of its columns, by name.

    Raises
    ------
    ConfigError
        If the file cannot be read, a column is missing, the columns are
        not of one length or have no particles, a value is not finite, or
        a dispersion is negative.
    """
    try:
        if os.path.splitext(file)[1].lower() == '.npz':
            with np.load(file) as arrays:
                table = {key: np.asarray(arrays[key]) for key in arrays}
        else:
            read = astropy.table.Table.read(file)
            table = {key: np.asarray(read[key]) for key in read.colnames}
    except (OSError, ValueError) as e:
        raise ConfigError(f"cannot read the particles of {file}: {e}") from e
    if missing := [key for key in _COLUMNS if key not in table]:
        raise ConfigError(
            f"the particles of {file} have no columns {missing} (see the "
            f"component particles)")
    names = _COLUMNS + ((_SIGMA,) if _SIGMA in table else ())
    columns = {key: np.asarray(table[key], dtype=float) for key in names}
    if len({len(column) for column in columns.values()}) > 1:
        raise ConfigError(f"the columns of the particles of {file} differ "
                          f"in length")
    if not len(columns['x']):
        raise ConfigError(f"the table of {file} has no particles")
    if not all(np.all(np.isfinite(column)) for column in columns.values()):
        raise ConfigError(f"the particles of {file} have values that are "
                          f"not finite")
    if np.any(columns.get(_SIGMA, 0) < 0):
        raise ConfigError(f"the particles of {file} have negative "
                          f"dispersions")
    return columns


def _sky(
        particles: dict[str, np.ndarray], params: ParamValues
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the positions of particles on the sky (x and y), their depths
    along the line of sight (away from the viewer) and their velocities
    along it (without the systemic velocity), seen as a disk of their
    frame is (see transform_cpos_posa_incl_inverse and vp_disk_project).
    """
    x, y, depth = disk_to_sky(
        particles['x'], particles['y'], particles['z'], params['xpos'],
        params['ypos'], params['posa'], params['incl'])
    incl = np.radians(params['incl'])
    velocity = particles['vy'] * np.sin(incl) + particles['vz'] * np.cos(incl)
    return x, y, depth, velocity


def _pixels(
        x: np.ndarray, y: np.ndarray, grid: NativeGrid
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the pixels that hold positions on the sky: the points that are
    on the grid, and the x and y of their pixels. (Not shared by the four
    pixels around, as a point is: that would smooth their light by a
    pixel.)
    """
    rota = np.radians(grid['spat_rota'])
    gx = x * np.cos(rota) + y * np.sin(rota)
    gy = -x * np.sin(rota) + y * np.cos(rota)
    px = np.floor((gx - grid['spat_zero'][0]) / grid['spat_step'][0] + 0.5)
    py = np.floor((gy - grid['spat_zero'][1]) / grid['spat_step'][1] + 0.5)
    size_x, size_y = grid['spat_size'][:2]
    points = np.flatnonzero(
        (px >= 0) & (px < size_x) & (py >= 0) & (py < size_y))
    return points, px[points].astype(int), py[points].astype(int)


def _optical_depths(
        odata: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        depths: np.ndarray,
        grid: NativeGrid
) -> np.ndarray:
    """
    Return the optical depth in each cube of opacity from the viewer to
    the given depths on the lines of sight of the given pixels (ncubes,
    n), from that from the viewer to the far side of each voxel (odata,
    of shape (ncubes, nz, ny, nx)), with the absorbers of each voxel
    spread along it (as the point component, see points._optical_depths).
    """
    size, step, zero = (
        grid[key][2] for key in ('spat_size', 'spat_step', 'spat_zero'))
    p = np.clip((depths - zero) / step + 0.5, 0, size)
    z = np.minimum(p.astype(int), size - 1)
    front = np.where(
        z > 0, odata[:, np.maximum(z - 1, 0), ys, xs], 0)
    return front + (p - z) * (odata[:, z, ys, xs] - front)


class ParticlesPlan(ComponentPlan):
    """
    The evaluation of a Particles component: of each particle in the pixel
    that holds it, at its depth and velocity (see Particles), in numpy on
    the host, then added to the outputs on the device.
    """

    def __init__(
            self,
            driver: Driver,
            dtype: np.dtype,
            particles: dict[str, np.ndarray],
            opacity_ratios: np.ndarray | None,
            fluxes: Fluxes,
            lines: LinePlacement | None = None
    ):
        self._driver = driver
        self._dtype = np.dtype(dtype)
        self._particles = particles
        self._opacity_ratios = opacity_ratios
        self._fluxes = fluxes
        self._lines = lines

    def line_values(self, params: ParamValues) -> np.ndarray:
        if self._lines is None:
            return np.array([[0.0, 0.0, self._fluxes(params)[0]]])
        rows = np.zeros((len(self._lines), 3))
        rows[:, :2] = self._lines(params['vsys'])
        rows[:, 2] = self._fluxes(params)
        return rows

    def _spectra(
            self,
            centres: np.ndarray,
            sigmas: np.ndarray,
            grid: NativeGrid
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the light of lines in the channels about their centres, per
        unit flux, each the mean of a Gaussian of its dispersion over the
        channel (a channel holds all the light of a line of none), and
        those channels (both of shape (n, nwindow); -1 for those off the
        axis): a window of _LINE_WINDOW dispersions on each side.
        """
        size, step, zero = (
            grid['spec_size'], grid['spec_step'], grid['spec_zero'])
        half = int(np.ceil(_LINE_WINDOW * sigmas.max(initial=0) / step)) + 1
        nearest = np.rint((centres - zero) / step).astype(int)
        channels = nearest[:, None] + np.arange(-half, half + 1)
        lower = zero + (channels - 0.5) * step
        with np.errstate(divide='ignore', invalid='ignore'):
            scaled = np.sqrt(2) * sigmas[:, None]
            cdf = 0.5 * (1 + scipy.special.erf(np.stack(
                [lower - centres[:, None],
                 lower + step - centres[:, None]]) / scaled))
        # (a line of no dispersion: a step at its centre, in the channel
        # that holds it)
        cdf = np.where(
            sigmas[:, None] > 0, cdf,
            np.stack([lower, lower + step]) > centres[:, None])
        light = (cdf[1] - cdf[0]) / step
        off = (channels < 0) | (channels >= size)
        return np.where(off, 0, light), np.where(off, -1, channels)

    def evaluate_emission(
            self,
            params: ParamValues,
            grid: NativeGrid,
            outputs: dict[str, DeviceArray | None],
            out_extra: dict[str, Any] | None,
            state: ComponentState | None
    ) -> None:
        driver = self._driver
        image = outputs.get('image')
        scube = outputs.get('scube')
        odata = outputs.get('odata')
        if odata is not None:
            odata = driver.mem_copy_d2h(odata)
        size_x, size_y = grid['spat_size'][:2]
        area = grid['spat_step'][0] * grid['spat_step'][1]
        rows = self.line_values(params)
        npixels = size_x * size_y
        h_image = np.zeros(npixels) if image is not None else None
        h_scube = np.zeros(grid['spec_size'] * npixels) \
            if scube is not None else None
        n = len(self._particles['flux'])
        for start in range(0, n, _CHUNK):
            chunk = {
                key: column[start:start + _CHUNK]
                for key, column in self._particles.items()}
            x, y, depth, velocity = _sky(chunk, params)
            points, xs, ys = _pixels(x, y, grid)
            # The surface brightness of each particle on the grid, and the
            # fraction of the light of each row that the opacity in front
            # of it lets through (npoints, nrows)
            light = chunk['flux'][points] / area
            transmission = np.ones((len(points), len(rows))) \
                if odata is None else np.exp(-(
                    self._opacity_ratios @ _optical_depths(
                        odata, xs, ys, depth[points], grid)).T)
            pixels = ys * size_x + xs
            if h_image is not None:
                h_image += np.bincount(
                    pixels, light * rows[0, 2] * transmission[:, 0],
                    minlength=npixels)
            if h_scube is None:
                continue
            sigmas = chunk.get(_SIGMA, np.zeros(len(chunk['flux'])))
            for row, (offset, scale, flux) in enumerate(rows):
                spectra, channels = self._spectra(
                    offset + scale * (params['vsys'] + velocity),
                    scale * sigmas, grid)
                values = (light * flux * transmission[:, row])[:, None] \
                    * spectra[points]
                channels = channels[points]
                kept = channels >= 0
                voxels = channels * npixels + pixels[:, None]
                h_scube += np.bincount(
                    voxels[kept], values[kept], minlength=h_scube.size)
        if image is not None:
            target = image.reshape(-1, *image.shape[-2:])[0]
            driver.math_add(target, driver.mem_copy_h2d(
                h_image.reshape(size_y, size_x).astype(self._dtype)),
                out=target)
        if scube is not None:
            driver.math_add(scube, driver.mem_copy_h2d(
                h_scube.reshape(scube.shape).astype(self._dtype)), out=scube)

    def fields_layout(self) -> FieldsLayout:
        return FieldsLayout(layer=False, nslots=3)

    def evaluate_fields(
            self,
            params: ParamValues,
            grid: NativeGrid,
            fields: DeviceArray,
            out_extra: dict[str, Any] | None,
            state: ComponentState | None
    ) -> None:
        # The particles of each voxel as one line, of their moments: their
        # light j, its mean velocity v and its dispersion s (that of their
        # velocities and their own; as the clouds of a Monte Carlo disk)
        size_x, size_y, size_z = grid['spat_size']
        nvoxels = size_x * size_y * size_z
        sums = np.zeros((4, nvoxels))
        area = grid['spat_step'][0] * grid['spat_step'][1]
        spectral = self._lines is not None
        for start in range(0, len(self._particles['flux']), _CHUNK):
            chunk = {
                key: column[start:start + _CHUNK]
                for key, column in self._particles.items()}
            x, y, depth, velocity = _sky(chunk, params)
            points, xs, ys = _pixels(x, y, grid)
            # (the plane of each in the slab of the grid, from its plane
            # slab_zero of the depth axis; see NativeGrid)
            zs = np.floor((depth[points] - grid['spat_zero'][2])
                          / grid['spat_step'][2] + 0.5).astype(int) \
                - grid['slab_zero']
            kept = (zs >= 0) & (zs < size_z)
            index = (zs[kept] * size_y + ys[kept]) * size_x + xs[kept]
            light = chunk['flux'][points][kept] / area
            v = (params['vsys'] + velocity[points])[kept] if spectral \
                else np.zeros(len(index))
            s2 = chunk.get(_SIGMA, np.zeros(len(chunk['flux'])))[
                points][kept] ** 2 if spectral else np.zeros(len(index))
            for i, values in enumerate((light, light * v, light * v ** 2,
                                        light * s2)):
                sums[i] += np.bincount(index, values, minlength=nvoxels)
        j, jv, jv2, js2 = sums
        with np.errstate(divide='ignore', invalid='ignore'):
            v = np.where(j > 0, jv / j, 0)
            s = np.where(
                j > 0, np.sqrt(np.maximum(jv2 / j - v ** 2 + js2 / j, 0)), 0)
        layout = self.fields_layout()
        self._driver.mem_copy_h2d(
            np.stack([j, v, s]).reshape(layout.shape(grid)).astype(
                self._dtype), fields)


class Particles(Component):
    """
    Particles given in a table (e.g. the gas of a simulation), each the
    light of a point at its position, moving at its velocity: for mock
    data of them (see GroupPlan.mock), which other models then fit.
    Each is in the pixel that holds it (so that a pixel has the light of
    the particles in it), as surface brightness (as a point component, see
    PointPlan), with its lines (emission lines) each a Gaussian of its
    dispersion; on a grid with a depth axis, the opacity in front of its
    depth dims it. In the march of radiative transfer, the particles
    of each voxel are one line, of their moments (as the clouds of a
    Monte Carlo disk).

    The table has the columns x, y, z (arcsec), vx, vy, vz (km/s) and
    flux (in the unit of its emission; of its first line, or of its
    band), and optionally sigma (km/s, the dispersion of the lines of
    each particle; 0 without). They are in the frame of the galaxy, z
    along its spin axis, seen as a disk of that frame is (see the disks):
    at the centre (xpos, ypos), the position angle posa (of the +x axis
    of the frame) and the inclination incl, with the velocity along the
    line of sight vsys + vy sin(incl) + vz cos(incl). Its parameters:
    xpos, ypos, posa, incl, and with emission lines vsys, and the flux
    ratios of its lines (see Lines).

    (The positions are in arcsec, as those of every component: the
    distance of a galaxy reaches its components only through its mass
    (see GalaxyStage). Positions in kpc would let one table be seen at any
    distance, with the length of an arcsec passed by the galaxy to the
    components that ask for it, as it passes the circular velocity to
    those it moves; not built.)

    Parameters
    ----------
    file : str
        The table of the particles: a FITS table, an ECSV file or a
        numpy file (npz) of its columns.
    emission : Emission
        The light it emits.
    geometry : Geometry, 'galaxy' or 'own', optional
        The geometry whose centre, position angle, inclination and systemic
        velocity it shares: that of its galaxy (by default), a named one
        (not a loose or tilted one), or none ('own'; see Component).
    name : str, optional
        Its name (see Component).

    Raises
    ------
    ConfigError
        If the table is invalid (see read_particles), the geometry is
        loose or tilted, or the lines are not valid.
    """

    @staticmethod
    def type() -> str:
        return 'particles'

    @classmethod
    def load(
            cls,
            info: dict[str, Any],
            catalogue: LineCatalogue | None = None
    ) -> 'Particles':
        parseutils.load_option_and_update_info(
            emission_parser, info, 'emission', required=True,
            catalogue=catalogue)
        if 'file' in info:
            info['file'] = parseutils.resolve_path(info['file'])
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        return (
            dict(type=self.type())
            | dump_name(self)
            | dump_geometry(self)
            | dict(file=self._file,
                   emission=emission_parser.dump(self.emission())))

    def __init__(
            self,
            file: str,
            emission: Emission,
            geometry: Geometry | Literal['galaxy', 'own'] = 'galaxy',
            name: str | None = None
    ):
        if isinstance(geometry, Geometry) and (
                geometry.loose() or geometry.tilted()):
            raise ConfigError(
                f"particles have one centre and orientation, but "
                f"{geometry_desc(geometry)} is loose or tilted; give them "
                f"geometry 'own'")
        super().__init__(name, geometry, emission, None)
        self._file = file
        self._particles = read_particles(file)
        self._lines = None
        self._bands = None
        names = ('xpos', 'ypos', 'posa', 'incl')
        if isinstance(emission, LineEmission):
            self._lines = emission.lines()
            names += ('vsys',)
            light = self._lines
        else:
            self._bands = emission.bands()
            light = self._bands
        pdescs = {name: ParamScalarDesc() for name in names}
        if repeated := sorted(set(light.pdescs()) & set(pdescs)):
            raise ConfigError(
                f"the parameters of the flux ratios have the names of other "
                f"parameters: {repeated}; rename the lines or bands")
        self._pdescs = pdescs | light.pdescs()

    def file(self) -> str:
        """Return the file of its table."""
        return self._file

    def thick(self, seen_by: 'type[Emission]') -> bool:
        return False

    def frame_params(self) -> FrameParams:
        vsys = 'vsys' if self._lines is not None else None
        return FrameParams('xpos', 'ypos', vsys, 'posa', 'incl')

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return self._pdescs

    def bounding_radius(self) -> float:
        return float(np.sqrt(np.max(
            self._particles['x'] ** 2 + self._particles['y'] ** 2
            + self._particles['z'] ** 2)))

    def plan(
            self,
            driver: Driver,
            spectral: gridutils.Grid,
            dtype: np.dtype,
            lines: tuple[str, ...] | None,
            band: Band | None,
            emits: bool,
            opacity_ratios: np.ndarray | None,
            data_unit: DataUnit | None,
            redshift: float
    ) -> ParticlesPlan:
        unit = self.emission().unit()
        if self._lines is None:
            return ParticlesPlan(
                driver, dtype, self._particles, opacity_ratios,
                band_light(self._bands, unit, band, data_unit, redshift))
        return ParticlesPlan(
            driver, dtype, self._particles, opacity_ratios, *line_light(
                spectral, self._lines, unit, lines, data_unit, redshift))
