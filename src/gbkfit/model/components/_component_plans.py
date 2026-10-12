"""
The plans of the components of a galaxy on the grid of an observation,
which the evaluations of the galaxy share (see ComponentSetPlan and
TransferPlan): the plans of the components it evaluates, the cubes of
opacity of its absorbers, and the parameters of each component.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from gbkfit.driver import Driver
from gbkfit.params import ParamValues
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError
from ..distance import Distance
from ..mass import MassContext, MassModel
from .base import Component, ComponentPlan, DepthAxis, NativeGrid
from .disks._disk import DiskPlan
from .disks._profiles import DiskDensity, pressure_support
from .extinction import ExtinctionLaw
from .lines import Band

if TYPE_CHECKING:
    from ._component_set import ComponentSet


__all__ = [
    'ComponentPlans'
]


class ComponentPlans:
    """
    The plans of the components of a galaxy on the grid of an
    observation: the x and y axes of its data, and a depth axis along the
    line of sight (or none: the plane of the sky), with the spectral axis
    of its data (or that of an image). It plans the given emitters, whose
    light is evaluated, and the components with opacity (the absorbers:
    all of them, since they absorb whatever the observation sees). Each
    component has its parameters, and those its galaxy gives it (the
    circular velocity of its mass, evaluated once for all of them).

    The absorbers add the optical depth of each voxel to cubes of opacity
    (see _opacity_cubes), whose ratios at the rest wavelength of each line
    of each emitter (or of its band) are those of their extinction laws.

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
    lines : tuple of str, optional
        The names of the emission lines evaluated; all if None.
    band : Band, optional
        The band of an image, if known.
    redshift : float
        The redshift of the galaxy, which makes the rest wavelengths of
        bands.

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
        (nrows, ncubes)), or None without absorbers.
    odata : DeviceArray
        The cubes of opacity (of shape (ncubes, nz, ny, nx)), or None
        without absorbers.

    Raises
    ------
    ValueError
        If there is no depth axis and components have opacity (the
        observations check that first, see LineOfSight).
    ConfigError
        If dust of an extinction law dims light of an unknown wavelength,
        or there are too many laws (see _opacity_cubes).
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
            lines: tuple[str, ...] | None,
            band: Band | None,
            redshift: float
    ):
        components = component_set.components()
        absorbers = tuple(
            i for i, cmp in enumerate(components) if cmp.opacity() is not None)
        if absorbers and depth is None:
            raise ValueError(
                "components with opacity need a grid with a depth axis")
        self._component_set = component_set
        self._grid = grid
        self._depth = depth
        self.driver = driver
        self.dtype = dtype
        self.native_grid = _native_grid(grid, spectral, depth)
        self.emitters = tuple(emitters)
        self.absorbers = absorbers
        # The cubes of opacity: the law of each, the cube of each absorber
        # and the factor of its optical depth there, and the ratios of the
        # cubes at the wavelength of each line (or band) of each emitter
        self.ratios = dict.fromkeys(emitters)
        self.odata = None
        self._deposits = {}
        if absorbers:
            wavelengths = {
                i: components[i].emission().wavelengths(
                    lines, band, redshift)
                for i in emitters}
            cube_laws, cubes, scales = _opacity_cubes(
                [components[i].opacity().law() for i in absorbers],
                [w for i in emitters for w in wavelengths[i]])
            self._deposits = dict(zip(absorbers, zip(cubes, scales)))
            self.ratios = {
                i: np.array([[_ratio(law, w) for law in cube_laws]
                             for w in wavelengths[i]], dtype)
                for i in emitters}
            self.odata = driver.mem_alloc_d(
                (len(cube_laws),) + self.native_grid['spat_size'][::-1],
                dtype)
        # The plans of the emitters, and of the absorbers that do not emit
        # here
        self.plans = {
            i: components[i].plan(
                driver, spectral, dtype, lines, band, True, self.ratios[i])
            for i in emitters}
        self.plans |= {
            i: components[i].plan(
                driver, spectral, dtype, lines, band, False, None)
            for i in absorbers if i not in self.plans}
        self._followed = _followed_densities(
            component_set, self.plans, driver, dtype)

    def params(self, params: ParamValues) -> dict[int, ParamValues]:
        """
        Return the values of the parameters of each component it evaluates
        (by its own names), by index, from those of the galaxy, with those
        its galaxy gives it.
        """
        components = self._component_set.components()
        mappings = self._component_set.mappings()
        evaluated = sorted(self.plans)
        given = _circular_velocities(
            [components[i] for i in evaluated],
            [self.plans[i].disk_plan() for i in evaluated],
            [mappings[i] for i in evaluated], self._followed,
            self._component_set.distance(), self._component_set.mass_model(),
            params)
        return {
            i: {p: params[mappings[i][p]] for p in components[i].pdescs()}
            | cmp_given
            for i, cmp_given in zip(evaluated, given)}

    def evaluate_opacity(
            self,
            params: dict[int, ParamValues],
            out_extra: dict[str, Any] | None
    ) -> np.ndarray | None:
        """
        Write the optical depth of each voxel of the absorbers to the cubes
        of opacity, each with the parameters of its index in params. With
        out_extra, add the extra outputs of each absorber to it (see
        add_extra), and return the optical depth at V of each voxel of
        all of them.
        """
        if self.odata is None:
            return None
        self.driver.mem_fill(self.odata, 0)
        total = None
        for i in self.absorbers:
            cube, scale = self._deposits[i]
            extra = {} if out_extra is not None else None
            self.plans[i].evaluate_opacity(
                params[i], self.native_grid, self.odata[cube], scale, extra)
            if extra is not None:
                tau = extra['odata']
                total = tau if total is None else total + tau
            self.add_extra(out_extra, i, extra)
        return total

    def add_extra(
            self,
            out_extra: dict[str, Any] | None,
            index: int,
            extra: dict[str, Any] | None
    ) -> None:
        """
        Add the extra outputs of the component of the given index to
        out_extra, named after its name, or its position if it has none
        (e.g. 'disk_bdata' or 'component0_bdata').
        """
        if out_extra is None:
            return
        name = self._component_set.components()[index].name()
        label = name if name is not None else f'component{index}'
        for key, value in extra.items():
            out_extra[f'{label}_{key}'] = self.extra(value)

    def extra(self, data: np.ndarray) -> gridutils.GridData:
        """
        Return an extra output on the grid: an image without a depth axis,
        a cube with one (along the line of sight, from z = 0).
        """
        grid = self._grid
        if self._depth is None:
            return gridutils.GridData(
                data.reshape(data.shape[-2:]), grid.coords, None)
        depth = self._depth
        coords = gridutils.Coords(
            tuple(grid.coords.step) + (depth.step,),
            tuple(grid.coords.rpix) + (-depth.zero / depth.step,),
            tuple(grid.coords.rval) + (0.0,), grid.coords.rota)
        return gridutils.GridData(data, coords, None)


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
        spec_size=spectral.size[0],
        spec_step=spectral.coords.step[0],
        spec_zero=spectral.zero()[0])


# The most cubes of opacity that the native kernels read (OPACITY_NUM_MAX
# in constants.hpp)
_MAX_OPACITY_CUBES = 4


def _ratio(law: ExtinctionLaw | None, wavelength: float | None) -> float:
    """
    Return the ratio of the optical depth of dust of an extinction law
    (None: grey dust, 1 everywhere) at a rest wavelength (um) to that at V.
    Raise ConfigError if the law needs a wavelength that is not known.
    """
    if law is None:
        return 1.0
    if wavelength is None:
        raise ConfigError(
            "dust of an extinction law dims light at its rest wavelength: "
            "the emission of the components it dims must name its lines "
            "(e.g. lines: [halpha]), and an image needs its band")
    return law.ratio(wavelength)


def _opacity_cubes(
        laws: Sequence[ExtinctionLaw | None],
        wavelengths: Sequence[float | None]
) -> tuple[
        tuple[ExtinctionLaw | None, ...], tuple[int, ...], tuple[float, ...]]:
    """
    Return how the optical depth of absorbers of the given extinction laws
    (None: grey dust) is held for an observation of light of the given
    rest wavelengths (um; None where not known): the law of each cube of
    opacity (None: the ratio is 1 at every wavelength), and the cube of
    each absorber and the factor of its optical depth at V there. With one
    law, one cube of the optical depth at V; with several laws and one
    wavelength, one cube of the optical depth at that wavelength; with
    several laws and wavelengths, one cube of the optical depth at V of
    each law.

    Raises
    ------
    ConfigError
        If a law needs a wavelength that is not known, or with several
        wavelengths, there are more laws than the kernels take.
    """
    distinct = tuple(dict.fromkeys(laws))
    if len(distinct) == 1:
        return distinct, (0,) * len(laws), (1.0,) * len(laws)
    if len(set(wavelengths)) == 1:
        return (
            (None,), (0,) * len(laws),
            tuple(_ratio(law, wavelengths[0]) for law in laws))
    if len(distinct) > _MAX_OPACITY_CUBES:
        raise ConfigError(
            f"the dust that an observation of several wavelengths sees can "
            f"have at most {_MAX_OPACITY_CUBES} extinction laws; it has "
            f"{len(distinct)}")
    return (
        distinct, tuple(distinct.index(law) for law in laws),
        (1.0,) * len(laws))


def _followed_densities(
        component_set: 'ComponentSet',
        plans: dict[int, ComponentPlan],
        driver: Driver,
        dtype: np.dtype
) -> tuple[tuple[Component, DiskPlan, dict[str, str]], ...]:
    """
    Return the components of a galaxy that the components of its mass
    follow (see MassComponent.followed_components), with the plans of
    their disks and the names of their parameters: those of the given
    plans, or new ones for the other components (their mass counts,
    though they are not evaluated).
    """
    mass_model = component_set.mass_model()
    if mass_model is None:
        return ()
    followed = {name for mass_component in mass_model.components()
                for name in mass_component.followed_components()}
    result = []
    for i, (cmp, mapping) in enumerate(zip(
            component_set.components(), component_set.mappings())):
        if cmp.name() not in followed:
            continue
        disk_plan = plans[i].disk_plan() if i in plans \
            else cmp.density_disk().plan(driver, 0, dtype)
        result.append((cmp, disk_plan, mapping))
    return tuple(result)


def _circular_velocities(
        components: Sequence[Component],
        disk_plans: Sequence[DiskPlan | None],
        mappings: Sequence[dict[str, str]],
        followed: Sequence[tuple[Component, DiskPlan, dict[str, str]]],
        distance: Distance | None,
        mass_model: MassModel | None,
        params: ParamValues
) -> list[dict[str, np.ndarray]]:
    """
    Return the values of the parameters of each component that are the
    circular velocity of the mass of the galaxy (see
    Component.circular_velocity_params), from the parameters of the
    distance and the mass in params, and the densities of the components
    it follows (see _followed_densities and DiskDensity), with the
    pressure support of their component where its traits ask for it (see
    Disk.pressure_supported_params): NaN where their square is negative or
    infinite (there is no circular orbit). The mass is evaluated once, at
    all their radii.
    """
    radii = [cmp.circular_velocity_params() for cmp in components]
    if mass_model is None or not any(radii):
        return [{} for _ in components]
    kpc_per_arcsec = distance.kpc_per_arcsec(params)
    densities = {
        cmp.name(): DiskDensity(
            cmp.density_disk(), disk_plan,
            {p: params[mapping[p]] for p in cmp.pdescs()}, kpc_per_arcsec)
        for cmp, disk_plan, mapping in followed}
    context = MassContext(
        distance.critical_density(params), distance.omega_m(params),
        densities)
    every = [r for cmp in radii for r in cmp.values()]
    vcirc2 = mass_model.vcirc2(
        np.concatenate(every) * kpc_per_arcsec, params, context)
    ends = np.cumsum([len(r) for r in every])
    pieces = iter(vcirc2[end - len(r):end] for r, end in zip(every, ends))
    values = []
    for cmp, disk_plan, mapping, cmp_radii in zip(
            components, disk_plans, mappings, radii, strict=True):
        values.append({})
        for name, r in cmp_radii.items():
            v2 = next(pieces)
            disk = cmp.density_disk()
            if name in disk.pressure_supported_params():
                v2 = v2 + pressure_support(
                    disk, disk_plan,
                    {p: params[mapping[p]] for p in cmp.pdescs()}, r)
            values[-1][name] = np.sqrt(
                np.where(np.isfinite(v2) & (v2 >= 0), v2, np.nan))
    return values
