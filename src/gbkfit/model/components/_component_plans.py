"""
The plans of the components of a galaxy on the grid of an observation,
which the evaluations of the galaxy share (see ComponentSetPlan and
TransferPlan): the plans of the components it evaluates, the cubes of
opacity of its absorbers, and the parameters of each component.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamValues
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError
from ._units import DataUnit
from .base import DepthAxis, NativeGrid, OpacityCubes
from .disks._disk import Disk, DiskPlan
from .extinction import ExtinctionLaw
from .lines import Band
from .roles import DustOpacity

if TYPE_CHECKING:
    from ..galaxy import Galaxy
    from ._component_set import ComponentSet
    from .base import Component


__all__ = [
    'ComponentPlans',
    'opacity_cubes'
]


_log = logging.getLogger(__name__)

# The share of the profile of a density height trait beyond its zmax
# above which a warning says that zmax shapes the density, and the
# azimuths where the displacements of vertical distortion traits are
# checked against their zmax (see ComponentPlans.warn_truncation)
_MAX_CUT = 0.01
_CUT_AZIMUTHS = 64

# The voxels whose light the resolution of the light checks: those with at
# least this share of the light of the brightest voxel of their component
# (see ComponentPlans.light_resolution)
_BRIGHT = 0.05


class ComponentPlans:
    """
    The plans of the components of a galaxy on the grid of an
    observation: the x and y axes of its data, and a depth axis along the
    line of sight (or none: the plane of the sky), with the spectral axis
    of its data (or that of an image). It plans the given emitters, whose
    light is evaluated, and the components with dust (the absorbers: all
    of them, since they absorb whatever the observation sees; the line
    opacity of an emitter is that of its own lines, see LineOpacity). Each
    component has its parameters, and those that the gravity of its galaxy
    gives it (see GravityPlan).

    The absorbers add the optical depth of each voxel to the cubes of
    opacity, as the given OpacityCubes hold the dust (see opacity_cubes).

    Parameters
    ----------
    component_set : ComponentSet
        The components.
    driver : Driver
        The driver of the evaluation.
    grid : Grid
        The grid of the x and y axes.
    spectral : Grid
        The spectral axis (one axis).
    depth : DepthAxis, optional
        The depth axis, if any.
    dtype : np.dtype
        The floating type of the evaluation.
    emitters : Sequence of int
        The indices of the components whose light is evaluated.
    cubes : OpacityCubes
        How the cubes of opacity hold the dust of the absorbers, and that
        of other galaxies, which dims the light of the emitters (see
        opacity_cubes).
    lines : tuple of str, optional
        The names of the emission lines evaluated; all if None.
    band : Band, optional
        The band of an image, if known.
    data_unit : DataUnit, optional
        The unit of the data of the observation, which the light of the
        emitters takes (see Component.plan); None without data.
    redshift : float
        The redshift of the galaxy, which places the lines of the
        emitters (see Component.plan).

    Attributes
    ----------
    driver : Driver
        The driver.
    dtype : np.dtype
        The floating type.
    native_grid : NativeGrid
        The grid of the native evaluation functions.
    emitters : tuple of int
        The indices of the emitters.
    absorbers : tuple of int
        The indices of the absorbers.
    plans : dict of int to ComponentPlan
        The plan of each emitter and absorber, by index.
    ratios : dict of int to np.ndarray
        For each emitter, the ratios of the cubes of opacity at the
        wavelength of each of its lines or its band (an array of shape
        (nrows, ncubes)), or None without dust.
    ncubes : int
        The number of cubes of opacity (0 without dust).

    Raises
    ------
    ConfigError
        If the light of an emitter has a unit that cannot become that of
        the data (see Component.plan).
    ValueError
        If there is no depth axis and components have opacity (the
        observations check that first, see LineOfSight).
    """

    def __init__(
            self,
            component_set: 'ComponentSet',
            driver: Driver,
            grid: gridutils.Grid,
            spectral: gridutils.Grid,
            depth: DepthAxis | None,
            dtype: np.dtype,
            emitters: Sequence[int],
            cubes: OpacityCubes,
            lines: tuple[str, ...] | None,
            band: Band | None,
            data_unit: DataUnit | None,
            redshift: float
    ):
        components = component_set.components()
        absorbers = tuple(cubes.deposits)
        if absorbers and depth is None:
            raise ValueError(
                "components with opacity need a grid with a depth axis")
        self._component_set = component_set
        self._grid = grid
        self._spectral = spectral
        self._depth = depth
        self.driver = driver
        self.dtype = dtype
        self.native_grid = _native_grid(grid, spectral, depth)
        self.emitters = tuple(emitters)
        self.absorbers = absorbers
        self.ratios = cubes.ratios
        self.ncubes = cubes.ncubes
        self._deposits = cubes.deposits
        # The plans of the emitters, and of the absorbers that do not emit
        # here
        self.plans = {}
        for i in emitters:
            try:
                self.plans[i] = components[i].plan(
                    driver, spectral, dtype, lines, band, True,
                    self.ratios[i], data_unit, redshift)
            except ConfigError as e:
                raise ConfigError(f"{_label(components, i)}: {e}") from e
        self.plans |= {
            i: components[i].plan(
                driver, spectral, dtype, lines, band, False, None, None,
                redshift)
            for i in absorbers if i not in self.plans}

    def params(
            self, params: ParamValues, gravity: dict[int, ParamValues]
    ) -> dict[int, ParamValues]:
        """
        Return the values of the parameters of each component it evaluates
        (by its own names), by index, from those of the galaxy, with those
        that the gravity of the galaxy gives the components it moves
        (gravity, see GravityPlan.evaluate).
        """
        components = self._component_set.components()
        mappings = self._component_set.mappings()
        return {
            i: {p: params[mappings[i][p]] for p in components[i].pdescs()}
            | gravity.get(i, {})
            for i in sorted(self.plans)}

    def evaluate_opacity(
            self,
            params: dict[int, ParamValues],
            grid: NativeGrid,
            odata: DeviceArray,
            extras: dict[int, dict[str, Any]] | None
    ) -> None:
        """
        Add the optical depth of each voxel of the grid of the absorbers
        to the cubes of opacity odata (of shape (ncubes, nz, ny, nx)), each
        with the parameters of its index in params. With extras, the extra
        outputs of each absorber go to extras[index].
        """
        for i in self.absorbers:
            cube, scale = self._deposits[i]
            extra = {} if extras is not None else None
            self.plans[i].evaluate_opacity(
                params[i], grid, odata[cube], scale, extra)
            if extras is not None:
                extras.setdefault(i, {}).update(extra)

    def warn_truncation(self, params: dict[int, ParamValues]) -> None:
        """
        Log a warning for each density height trait of the disks it
        evaluates, with the parameters of each by index (see params),
        whose zmax cuts more than 1% of its profile at a ring, and for
        each vertical distortion trait whose zmax limits its displacement
        somewhere (at a ring, at azimuths around it): zmax then shapes the
        disk, rather than holding it (see RHTrait and ZPTrait).
        """
        components = self._component_set.components()
        for i, plan in self.plans.items():
            disk = components[i].density_disk()
            if disk is None or not disk.traits('rht'):
                continue
            label = _label(components, i)
            for index, cut, radius in _height_cuts(disk, params[i]):
                if cut > _MAX_CUT:
                    trait = disk.traits('rht')[index]
                    _log.warning(
                        f"{label}: the zmax ({trait.zmax():g} arcsec) of its "
                        f"density height trait {index} ({trait.type()}) cuts "
                        f"{cut:.1%} of its profile at the radius {radius:g}: "
                        f"zmax shapes the density there; raise it")
            for index, radius in _limited_displacements(
                    disk, plan.disk_plan(), params[i]):
                trait = disk.traits('zpt')[index]
                _log.warning(
                    f"{label}: the zmax ({trait.zmax():g} arcsec) of its "
                    f"vertical distortion trait {index} ({trait.type()}) "
                    f"limits its displacement at the radius {radius:g}: "
                    f"zmax shapes the disk there; raise it")

    def light_resolution(
            self,
            values: dict[int, ParamValues],
            extras: dict[int, dict[str, Any]],
            smoothing: float
    ) -> float:
        """
        Return how finely the depth axis resolves the velocity of the light
        of the emitters, with the values of their parameters and their
        extra outputs (extras, by index: the light, velocity and dispersion
        of each voxel, 'bdata', 'vdata' and 'ddata'): the largest change of
        the velocity between adjacent voxels along a line of sight, where
        both are bright (see _BRIGHT), over the width of the lines there,
        that of their dispersion (the smaller of the two) and the given
        spectral smoothing of the instrument (km/s) in quadrature, on the
        spectral axis (see ComponentPlan.line_values; the widest scale of
        the lines of each emitter). Lines of no width are left out; 0
        without adjacent bright voxels (e.g. without a depth axis).
        """
        worst = 0.0
        for i in self.emitters:
            extra = extras.get(i, {})
            if 'vdata' not in extra or 'ddata' not in extra:
                continue
            j = np.abs(extra['bdata'])
            v, s = extra['vdata'], extra['ddata']
            scale = np.abs(self.plans[i].line_values(values[i])[:, 1]).max()
            bright = j >= _BRIGHT * j.max(initial=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                width = np.hypot(
                    scale * np.minimum(s[1:], s[:-1]), smoothing)
                ratio = scale * np.abs(v[1:] - v[:-1]) / width
            ratio = ratio[bright[1:] & bright[:-1] & (width > 0)]
            worst = max(worst, float(ratio.max(initial=0)))
        return worst

    def add_extras(
            self,
            out_extra: dict[str, Any],
            extras: dict[int, dict[str, Any]],
            totals: dict[str, np.ndarray],
            depth_zero: float | None = None
    ) -> None:
        """
        Add the extra outputs of each component (extras, by index) to
        out_extra, named after its name, or its position if it has none
        (e.g. 'disk_bdata' or 'component0_bdata'), and the totals
        ('total_*'): the given ones and the optical depth at V of all the
        absorbers ('total_odata'); with the given position of the first
        voxel of the depth axis, if not that of the axis (see extra).
        """
        components = self._component_set.components()
        for index, extra in extras.items():
            label = _label(components, index)
            for key, value in extra.items():
                out_extra[f'{label}_{key}'] = self.extra(value, depth_zero)
        if self.absorbers:
            totals = dict(odata=sum(
                extras[i]['odata'] for i in self.absorbers)) | totals
        for name, total in totals.items():
            out_extra[f'total_{name}'] = self.extra(total, depth_zero)

    def extra(
            self, data: np.ndarray, depth_zero: float | None = None
    ) -> gridutils.GridData:
        """
        Return an extra output on the grid: an image without a depth axis,
        a cube with one (along the line of sight, from z = 0 at the plane
        of the galaxy), whose first voxel is at the given position (arcsec;
        by default, that of the depth axis; it differs for the galaxies
        that a march evaluates together, see TransferPlan).
        """
        grid = self._grid
        if self._depth is None:
            return gridutils.GridData(
                data.reshape(data.shape[-2:]), grid.coords, None)
        depth = self._depth
        zero = depth.zero if depth_zero is None else depth_zero
        coords = gridutils.Coords(
            tuple(grid.coords.step) + (depth.step,),
            tuple(grid.coords.rpix) + (-zero / depth.step,),
            tuple(grid.coords.rval) + (0.0,), grid.coords.rota)
        return gridutils.GridData(data, coords, None)

    def output_extra(self, data: np.ndarray) -> gridutils.GridData:
        """
        Return an extra output of the shape of the output on the grid: an
        image (2d data), or a cube along the spectral axis (3d data).
        """
        coords = self._grid.coords
        if data.ndim == 2:
            return gridutils.GridData(data, coords, None)
        spectral = self._spectral.coords
        return gridutils.GridData(data, coords._replace(
            step=tuple(coords.step) + tuple(spectral.step),
            rpix=tuple(coords.rpix) + tuple(spectral.rpix),
            rval=tuple(coords.rval) + tuple(spectral.rval),
            rest=spectral.rest, medium=spectral.medium), 2)


def _label(components: Sequence['Component'], index: int) -> str:
    """
    Return the label of a component of the given index: its name, or its
    position if it has none (e.g. 'component0').
    """
    name = components[index].name()
    return name if name is not None else f'component{index}'


def _native_grid(
        grid: gridutils.Grid,
        spectral: gridutils.Grid,
        depth: DepthAxis | None
) -> NativeGrid:
    """
    Return the grid of the native evaluation functions: the x and y axes
    of the grid, the depth axis (size 1 and step 0 without one) and the
    spectral axis.
    """
    if depth is None:
        depth = DepthAxis(1, 0, 0)
    return NativeGrid(
        spat_size=tuple(grid.size[:2]) + (depth.size,),
        spat_step=tuple(grid.coords.step[:2]) + (depth.step,),
        spat_zero=tuple(grid.zero()[:2]) + (depth.zero,),
        spat_rota=grid.coords.rota,
        slab_zero=0,
        spec_size=spectral.size[0],
        spec_step=spectral.coords.step[0],
        spec_zero=spectral.zero()[0])


# The most cubes of opacity that the native kernels read (OPACITY_NUM_MAX
# in constants.hpp)
_MAX_OPACITY_CUBES = 4


class _Light(NamedTuple):
    """
    Light of a rest wavelength (um; None if not known) emitted at a
    redshift.
    """
    wavelength: float | None
    redshift: float

    def observed(self) -> float | None:
        """Return its observed wavelength (um), if known."""
        return None if self.wavelength is None \
            else self.wavelength * (1 + self.redshift)


class _Dust(NamedTuple):
    """
    Dust of an extinction law (None: grey dust, whose optical depth is
    that at V at every wavelength) at a redshift (0 for grey dust).
    """
    law: ExtinctionLaw | None
    redshift: float

    def ratio(self, light: _Light) -> float:
        """
        Return the ratio of its optical depth at the wavelength of the
        light in its rest frame to that at V. Raise ConfigError if it has a
        law and the wavelength is not known.
        """
        if self.law is None:
            return 1.0
        if light.wavelength is None:
            raise ConfigError(
                "dust of an extinction law dims light at its rest "
                "wavelength: the emission of the components it dims must "
                "name its lines (e.g. lines: [halpha]), and an image needs "
                "its band")
        # (a factor of 1 for light of the galaxy of the dust)
        return self.law.ratio(
            light.wavelength * ((1 + light.redshift) / (1 + self.redshift)))


# Dust whose optical depth is the same at every wavelength
_GREY = _Dust(None, 0.0)


def opacity_cubes(
        galaxies: Sequence['Galaxy'],
        emitters: Sequence[Sequence[int]],
        lines: tuple[str, ...] | None,
        band: Band | None,
        dtype: np.dtype
) -> list[OpacityCubes]:
    """
    Return how the plan of each of the given galaxies holds, in cubes of
    opacity, the dust of all of them, which dims the light of the given
    emitters of each (by index): of each of their lines of the given
    names (all if None), or in the given band of an image (if known). The
    components with dust are the absorbers. The dust of each is that of
    its extinction law at the redshift of its galaxy, which dims light of
    another galaxy at the wavelength of that light in its rest frame (see
    _opacity_cubes).

    Raises
    ------
    ConfigError
        If dust of an extinction law dims light of a wavelength that is
        not known, or there are too many kinds of dust (see
        _opacity_cubes).
    """
    absorbers = [
        [i for i, cmp in enumerate(galaxy.components())
         if isinstance(cmp.opacity(), DustOpacity)]
        for galaxy in galaxies]
    if not any(absorbers):
        return [OpacityCubes(0, {}, dict.fromkeys(indices))
                for indices in emitters]
    # The dust of each absorber, and the light of each line (or band) of
    # each emitter of each galaxy
    dust = []
    light = []
    for galaxy, galaxy_absorbers, galaxy_emitters in zip(
            galaxies, absorbers, emitters):
        components = galaxy.components()
        redshift = galaxy.redshift()
        for i in galaxy_absorbers:
            law = components[i].opacity().law()
            dust.append(_Dust(law, redshift) if law is not None else _GREY)
        light.append({
            i: [_Light(wavelength, redshift) for wavelength in
                components[i].emission().wavelengths(lines, band, redshift)]
            for i in galaxy_emitters})
    kinds, cubes, scales = _opacity_cubes(dust, [
        row for galaxy_light in light for rows in galaxy_light.values()
        for row in rows])
    deposits = iter(zip(cubes, scales))
    result = []
    for galaxy_absorbers, galaxy_light in zip(absorbers, light):
        result.append(OpacityCubes(
            len(kinds),
            {i: next(deposits) for i in galaxy_absorbers},
            {i: np.array([[kind.ratio(row) for kind in kinds] for row in rows],
                         dtype)
             for i, rows in galaxy_light.items()}))
    return result


def _opacity_cubes(
        dust: Sequence[_Dust], light: Sequence[_Light]
) -> tuple[tuple[_Dust, ...], tuple[int, ...], tuple[float, ...]]:
    """
    Return how the optical depth of absorbers of the given kinds of dust
    is held for an observation of light of the given rows (see _Light):
    the kind of dust of each cube of opacity, and the cube of each
    absorber and the factor of its optical depth at V there. With one
    kind, one cube of the optical depth at V; with several kinds and one
    observed wavelength, one cube of the optical depth at that wavelength
    (of grey dust); with several kinds and wavelengths, one cube of the
    optical depth at V of each kind.

    Raises
    ------
    ConfigError
        If dust of a law dims light of a wavelength that is not known, or
        with several wavelengths, there are more kinds of dust than the
        kernels take.
    """
    distinct = tuple(dict.fromkeys(dust))
    if len(distinct) == 1:
        return distinct, (0,) * len(dust), (1.0,) * len(dust)
    if all(_same_wavelength(row.observed(), light[0].observed())
           for row in light):
        return (
            (_GREY,), (0,) * len(dust),
            tuple(kind.ratio(light[0]) for kind in dust))
    if len(distinct) > _MAX_OPACITY_CUBES:
        raise ConfigError(
            f"the dust that an observation of several wavelengths sees can "
            f"have at most {_MAX_OPACITY_CUBES} extinction laws (a law of "
            f"galaxies of different redshifts counts once for each); it has "
            f"{len(distinct)}")
    return (
        distinct, tuple(distinct.index(kind) for kind in dust),
        (1.0,) * len(dust))


def _same_wavelength(a: float | None, b: float | None) -> bool:
    """
    Check whether two observed wavelengths (um; None if not known) are
    one: to the precision of their arithmetic (the rest wavelength of a
    band, times 1 + the redshift, is its wavelength only to rounding).
    """
    if a is None or b is None:
        return a is b
    return bool(np.isclose(a, b, rtol=1e-9, atol=0))


def _height_cuts(
        disk: Disk, params: ParamValues
) -> list[tuple[int, float, float]]:
    """
    Return, for each density height trait of a disk (with its
    parameters), the largest share of its profile that its zmax cuts (see
    RHTrait), over its rings, and the radius where it does: (the index of
    the trait, the share, the radius).
    """
    radii = np.asarray(disk.rnodes())
    trait_params = disk.trait_params('rht')
    cuts = []
    for index, (trait, names) in enumerate(zip(
            disk.traits('rht'), trait_params.pnames)):
        values = [
            _at_radii(disk, params[names[name]], radii)
            for name in ('s', 'b') if name in names]
        zmax = trait.zmax()
        cut = 1 - (trait.cdf(zmax, *values) - trait.cdf(-zmax, *values))
        ring = int(np.argmax(cut))
        cuts.append((index, float(cut[ring]), float(radii[ring])))
    return cuts


def _limited_displacements(
        disk: Disk, plan: DiskPlan, params: ParamValues
) -> list[tuple[int, float]]:
    """
    Return, for each vertical distortion trait of a disk (with its plan
    and parameters) whose zmax limits its displacement somewhere (see
    ZPTrait), over its rings and azimuths around them, the index of the
    trait and the first radius where it does.
    """
    if not disk.traits('zpt'):
        return []
    radii = np.asarray(disk.rnodes())
    azimuths = np.linspace(0, 360, _CUT_AZIMUTHS, endpoint=False)
    displacements = plan.evaluate_polar_traits(
        params, 'zpt', radii[:, None], azimuths)
    limited = []
    for index, (trait, values) in enumerate(zip(
            disk.traits('zpt'), displacements)):
        # (a limited displacement is zmax, in the precision of the kernels)
        zmax = values.dtype.type(trait.zmax())
        rings = np.flatnonzero((np.abs(values) >= zmax).any(axis=1))
        if rings.size:
            limited.append((index, float(radii[rings[0]])))
    return limited


def _at_radii(
        disk: Disk, values: float | np.ndarray, radii: np.ndarray
) -> np.ndarray:
    """
    Return the values of a parameter of a trait of a disk at the given
    radii: one value, or the node-wise values at the nodes or the
    subnodes of the disk, interpolated linearly.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return np.full(len(radii), float(values))
    nodes = disk.rnodes() if len(values) == len(disk.rnodes()) \
        else disk.subrnodes()
    return np.interp(radii, nodes, values)
