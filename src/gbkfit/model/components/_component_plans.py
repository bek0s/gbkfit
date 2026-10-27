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
from ..extras import Extras
from ._units import DataUnit
from .base import ComponentState, DepthAxis, NativeGrid, OpacityCubes
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

# The share of the light of a component beyond its extent above which a
# warning says that the extent shapes it (see
# ComponentPlans.warn_truncation)
_MAX_CUT = 0.01

# The voxels whose light the resolution of the light checks: those with at
# least this share of the light of the brightest voxel of their component
# (see ComponentPlans.light_resolution)
_BRIGHT = 0.05

# Where the grid of the extra outputs of the components is (see
# ComponentPlans.extra)
_ON_THE_GRID = (
    "On the pixels of the observation (finer by its option oversampling), "
    "along the line of sight if it has a depth axis (z = 0 at the plane of "
    "the galaxy, away from the viewer), or else in an image.")

# The titles and the descriptions of the extra outputs of each component,
# by name, and of their totals over the components of a galaxy (see
# ComponentPlans.add_extras)
_COMPONENT_EXTRAS = dict(
    bdata=(
        "Brightness",
        "The light that the component emits in each voxel, before any "
        "opacity."),
    odata=(
        "Optical depth of dust at V",
        "The optical depth at V of the dust of the component in each "
        "voxel."),
    vdata=(
        "Velocity",
        "The line-of-sight velocity of the component in each voxel (km/s; "
        "the mean weighted by its density there)."),
    ddata=(
        "Velocity dispersion",
        "The velocity dispersion of the component in each voxel (km/s; "
        "the mean weighted by its density there)."),
    obdata=(
        "Dimmed brightness",
        "The light of the component in each voxel after the opacity that "
        "dims it there."))
_TOTAL_EXTRAS = dict(
    total_bdata=(
        "Total brightness",
        "The light that the components of the galaxy emit in each voxel, "
        "before any opacity."),
    total_obdata=(
        "Total dimmed brightness",
        "The light of the components of the galaxy in each voxel after the "
        "opacity that dims it there."),
    total_odata=(
        "Total optical depth of dust at V",
        "The optical depth at V of the dust of the components of the galaxy "
        "in each voxel."))


class ComponentPlans:
    """
    The plans of the components of a galaxy on the grid of an
    observation: the x and y axes of its data, and a depth axis along the
    line of sight (or none: the plane of the sky), with the spectral axis
    of its data (or that of an image). It plans the given emitters, whose
    light is evaluated, and the components with dust (the absorbers: all
    of them, since they absorb whatever the observation sees; the line
    opacity of an emitter is that of its own lines, see LineOpacity). Each
    component has its parameters, and those that its galaxy gives it (see
    GalaxyStage).

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

    def params(self, params: ParamValues) -> dict[int, ParamValues]:
        """
        Return the values of the parameters of each component it evaluates
        (by its own names), by index, from those of the galaxy.
        """
        components = self._component_set.components()
        mappings = self._component_set.mappings()
        return {
            i: {p: params[mappings[i][p]] for p in components[i].pdescs()}
            for i in sorted(self.plans)}

    def evaluate_opacity(
            self,
            params: dict[int, ParamValues],
            states: dict[int, ComponentState],
            grid: NativeGrid,
            odata: DeviceArray,
            extras: dict[int, dict[str, Any]] | None
    ) -> None:
        """
        Add the optical depth of each voxel of the grid of the absorbers
        to the cubes of opacity odata (of shape (ncubes, nz, ny, nx)), each
        with the parameters of its index in params and what its galaxy
        gives it, its state, if any (see GalaxyState). With extras, the
        extra outputs of each absorber go to extras[index].
        """
        for i in self.absorbers:
            cube, scale = self._deposits[i]
            extra = {} if extras is not None else None
            self.plans[i].evaluate_opacity(
                params[i], grid, odata[cube], scale, extra, states.get(i))
            if extras is not None:
                extras.setdefault(i, {}).update(extra)

    def warn_truncation(self, params: dict[int, ParamValues]) -> None:
        """
        Log a warning for each component it evaluates, with the parameters
        of each by index (see params), whose extent cuts more than 1% of
        its light on a grid with a depth axis (see Component.extent_cut),
        and where an option of a component shapes it rather than holds it
        (see ComponentPlan.warn_truncation).
        """
        components = self._component_set.components()
        for i, plan in self.plans.items():
            extent = components[i].extent_cut(params[i]) \
                if self._depth is not None else None
            if extent is not None and extent[1] > _MAX_CUT:
                option, cut = extent
                _log.warning(
                    f"{_label(components, i)}: its {option} cuts "
                    f"{cut:.1%} of its light: {option} shapes it; raise it")
            plan.warn_truncation(params[i], _label(components, i))

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
            out_extra: Extras,
            extras: dict[int, dict[str, Any]],
            totals: dict[str, np.ndarray],
            depth_zero: float | None = None
    ) -> None:
        """
        Add the extra outputs of each component (extras, by index: e.g.
        'bdata', see _COMPONENT_EXTRAS) to out_extra, with the id
        'component': its name, or its position if it has none (e.g.
        'component0'); and their totals ('total_*', see _TOTAL_EXTRAS):
        the given ones and the optical depth at V of all the absorbers
        ('total_odata'); with the given position of the first voxel of the
        depth axis, if not that of the axis (see extra).
        """
        components = self._component_set.components()
        for index, extra in extras.items():
            label = _label(components, index)
            for key, value in extra.items():
                title, description = _COMPONENT_EXTRAS[key]
                out_extra.add(
                    key, self.extra(value, depth_zero), title,
                    f"{description} {_ON_THE_GRID}", component=label)
        if self.absorbers:
            totals = dict(odata=sum(
                extras[i]['odata'] for i in self.absorbers)) | totals
        for name, total in totals.items():
            title, description = _TOTAL_EXTRAS[f'total_{name}']
            out_extra.add(
                f'total_{name}', self.extra(total, depth_zero), title,
                f"{description} {_ON_THE_GRID}")

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
            rest=spectral.rest, medium=spectral.medium,
            convention=spectral.convention), 2)


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
