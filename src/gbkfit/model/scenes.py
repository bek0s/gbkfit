"""
Scenes: galaxies drawn together on one grid (e.g. those an observation
sees; see gbkfit.observation), and their evaluation on the grid of an
image or of a spectral cube: their light added up (AdditiveScenePlan), on
the plane of the sky or on a depth axis, or marched through, front to
back, by radiative transfer (TransferScenePlan). Which of them a scene
needs follows from its components (see Scene.evaluation).
"""

import abc
import enum
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.grid import Coords, Grid
from gbkfit.params import ParamValues
from ._galaxy_plan import GalaxyPlan
from ._opacity import opacity_cubes
from ._transfer import (
    MAX_SUBCHANNELS, LineResolution, TransferMember, TransferPlan,
    check_emitters, warn_resolution)
from ._galaxy_stage import GalaxyState
from .components._units import DataUnit
from .components.base import (
    ComponentState, DepthAxis, FrameParams, SceneTarget)
from .components.roles import View
from .extras import Extras

if TYPE_CHECKING:
    from gbkfit.instrument import PSF

    from .components.base import Component
    from .galaxy import Galaxy


__all__ = [
    'IMAGE_SPECTRAL_AXIS',
    'MAX_SUBCHANNELS',
    'AdditiveScenePlan',
    'DepthUnit',
    'Scene',
    'SceneEvaluation',
    'ScenePlan',
    'TransferScenePlan',
    'named_components'
]


# The spectral axis of an image: one channel, without a width
IMAGE_SPECTRAL_AXIS = Grid(
    (1,), Coords((0,), (0,), (0,), 0), 0)


class DepthUnit(NamedTuple):
    """
    Galaxies of a scene that radiative transfer marches together (see
    TransferScenePlan): one galaxy, or a group of galaxies that can pass
    through each other (see ObservationGroup: depth_order), by index, with
    the name of the group, and the parameter of the position of each along
    the line of sight, from the first of its group (its separation; None
    for the first, at 0).
    """
    galaxies: tuple[int, ...]
    name: str | None = None
    separations: tuple[str | None, ...] = (None,)


class SceneEvaluation(enum.Enum):
    """
    How a scene is evaluated along the line of sight, which its components
    decide (see Scene.evaluation):

    - PLANE: on the plane of the sky (AdditiveScenePlan without a depth
      axis), for components that are thin and have no opacity;
    - DEPTH: on a depth axis (AdditiveScenePlan with one), each galaxy
      dimmed by its own dust: thick components fill its voxels, and thin
      ones are layers at their depths;
    - TRANSFER: by radiative transfer along each line of sight
      (TransferScenePlan), front to back, for line opacity and for
      galaxies that dim one another.

    Where several could evaluate a scene, they give the same model: the
    first of them that can is the one.
    """
    PLANE = 'plane'
    DEPTH = 'depth'
    TRANSFER = 'transfer'


class Scene:
    """
    Galaxies drawn together on one grid (e.g. those an observation sees;
    see gbkfit.observation): the galaxies, the indices of the components
    of each whose light is drawn (its emitters; the opacity of all
    components applies), what it sees of their light (see View: a band,
    or a spectral axis with its lines and continuum), the order of the
    galaxies along the
    line of sight, front to back (in units: galaxies or groups of them;
    those it leaves out after them, in their order), and the unit of the
    data, which the
    light of the emitters takes (see Component.plan; None without data).
    The parameters of each galaxy have the names of the mapping of the
    galaxy among those of its group of galaxies (e.g. ObservationGroup),
    and its extra outputs its label as the id 'galaxy' (those of a group
    of them, its name as the id 'group').
    """

    def __init__(
            self,
            galaxies: Sequence['Galaxy'],
            mappings: Sequence[dict[str, str]],
            labels: Sequence[str],
            emitters: Sequence[tuple[int, ...]],
            view: View,
            order: Sequence[DepthUnit] | None = None,
            data_unit: DataUnit | None = None
    ):
        self._galaxies = tuple(galaxies)
        self._mappings = tuple(mappings)
        self._labels = tuple(labels)
        self._emitters = tuple(emitters)
        self._view = view
        self._data_unit = data_unit
        order = tuple(order or ())
        ordered = {g for unit in order for g in unit.galaxies}
        self._order = order + tuple(
            DepthUnit((g,)) for g in range(len(self._galaxies))
            if g not in ordered)

    def data_unit(self) -> DataUnit | None:
        """
        Return the unit of the data, which the light of the emitters takes
        (None without data).
        """
        return self._data_unit

    def view(self) -> View:
        """Return what it sees of the light of its components (see View)."""
        return self._view

    def emitters(self) -> tuple[tuple[int, ...], ...]:
        """
        Return the indices of the components of each galaxy whose light is
        drawn.
        """
        return self._emitters

    def order(self) -> tuple[DepthUnit, ...]:
        """
        Return the order of its galaxies along the line of sight, front to
        back, in units: those given, then the others, in their order.
        """
        return self._order

    def evaluation(self) -> SceneEvaluation:
        """
        Return how it is evaluated along the line of sight, which the
        components it draws (its emitters, and the absorbers of its
        galaxies, see Galaxy.absorbers) decide (see SceneEvaluation):

        - by radiative transfer if one has line opacity (see LineOpacity),
          or there are several galaxies and one has opacity (its light,
          or that of the others behind it, is then dimmed by the dust in
          front of it, in their order);
        - otherwise on a depth axis if one is thick in the emission it is
          seen in (see Component.thick) or has opacity;
        - otherwise on the plane of the sky.
        """
        drawn = [
            [galaxy.components()[i]
             for i in sorted(set(emitters) | set(galaxy.absorbers()))]
            for galaxy, emitters in zip(self._galaxies, self._emitters)]
        opacities = [
            [c.opacity() for c in components if c.opacity() is not None]
            for components in drawn]
        if any(not o.dims_others() for galaxy in opacities for o in galaxy) \
                or (len(drawn) > 1 and any(opacities)):
            return SceneEvaluation.TRANSFER
        if any(opacities) or any(
                c.needs_depth(self._view)
                for components in drawn for c in components):
            return SceneEvaluation.DEPTH
        return SceneEvaluation.PLANE

    def target(
            self,
            driver: Driver,
            grid: Grid,
            dtype: np.dtype,
            smoothing: float | None
    ) -> SceneTarget:
        """
        Return what it asks of each galaxy that it draws, and of their
        components, on a driver, a grid (of an image, without a spectral
        axis: one channel, IMAGE_SPECTRAL_AXIS), in a floating type, of the
        spectral smoothing of the instrument (km/s; None for an image):
        see SceneTarget.
        """
        image = grid.spectral_axis is None
        return SceneTarget(
            driver, grid if image else grid.spatial(),
            IMAGE_SPECTRAL_AXIS if image else grid.spectral(), dtype,
            self._view, self._data_unit, smoothing)

    def galaxy_label(self, galaxy: int) -> str:
        """
        Return the label of a galaxy, by its index: the id 'galaxy' of its
        extra outputs.
        """
        return self._labels[galaxy]

    def component(self, galaxy: int, component: int) -> 'Component':
        """Return a component of a galaxy, by their indices."""
        return self._galaxies[galaxy].components()[component]

    def component_names(self, galaxy: int, component: int) -> dict[str, str]:
        """
        Return the names of the parameters of a component of a galaxy (by
        their indices) in the group of the galaxies, by their own names.
        """
        return self._galaxies[galaxy].component_param_names(
            component, self._mappings[galaxy])

    def orbit_emitters(self) -> tuple[tuple[int, int], ...]:
        """
        Return the components whose light is drawn and whose stars move on
        orbits whose weights are solved (see Component.orbit_weights), in
        a spectral view of their stars (its lines name them; an image sees
        their light, as that of any component), each as the index of its
        galaxy (in the scene) and its own index.
        """
        if not self._view.continuum:
            return ()
        return tuple(
            (g, i)
            for g, (galaxy, emitters) in enumerate(
                zip(self._galaxies, self._emitters))
            for i in emitters
            if galaxy.components()[i].orbit_weights() is not None)

    def has_dust(self) -> bool:
        """Check whether a component of its galaxies has dust."""
        return any(galaxy.absorbers() for galaxy in self._galaxies)

    def bounding_radii(self) -> tuple[float, ...]:
        """
        Return the bounding radius (arcsec) of each galaxy: that of its
        largest component that is drawn (see _bounding_radius); 0 without
        extent.
        """
        return tuple(
            _bounding_radius(galaxy, emitters)
            for galaxy, emitters in zip(self._galaxies, self._emitters))

    def check_psf(self, psf: 'PSF | None') -> None:
        """
        Check that the components whose light is drawn can be seen through
        the PSF of an observation (None without one; see
        Component.check_psf).
        """
        for galaxy, emitters in zip(self._galaxies, self._emitters):
            for i in emitters:
                galaxy.components()[i].check_psf(psf)

    def frame_params(self, name: str) -> FrameParams:
        """
        Return the parameters that place the component of the given name
        (see named_components) in a frame (see Component.frame_params), by
        their names among those of the group of galaxies. Raise ValueError
        if no component has the name.
        """
        g, i = named_components(self._galaxies, [name])[name]
        return self._galaxies[g].component_frame_params(i, self._mappings[g])


class ScenePlan(abc.ABC):
    """
    The evaluation of a scene on a grid (e.g. along the line of sight of
    an observation, as it needs, see Scene.evaluation and
    gbkfit.observation.plan_scene): the plan of each galaxy, whose light
    adds up in the image or spectral cube.
    """

    def __init__(self, scene: Scene, grid: Grid, target: SceneTarget):
        self._scene = scene
        self._key = 'image' if grid.spectral_axis is None else 'scube'
        self._target = target
        self._spatial = target.spatial
        self._spectral = target.spectral

    def evaluate(
            self,
            params: ParamValues,
            galaxies: Sequence[GalaxyState],
            data: DeviceArray,
            out_extra: Extras | None
    ) -> None:
        """
        Add the light of the galaxies to the data (the image or spectral
        cube), with the parameters of the observation group (params) and
        what each galaxy gives its components in the evaluation (see
        GalaxyState). The extra outputs of each galaxy have its label as
        the id 'galaxy', and those of a group of galaxies its name as the
        id 'group'.
        """
        scene = self._scene
        values = [
            {name: params[full] for name, full in mapping.items()}
            for mapping in scene._mappings]
        extras = None if out_extra is None else [
            out_extra.scoped(galaxy=label) for label in scene._labels]
        self._evaluate(
            params, values, [galaxy.states for galaxy in galaxies],
            {self._key: data}, extras, out_extra)

    @abc.abstractmethod
    def _evaluate(
            self,
            group_params: ParamValues,
            params: list[ParamValues],
            states: Sequence[dict[int, ComponentState]],
            outputs: dict[str, DeviceArray],
            extras: list[Extras] | None,
            group_extras: Extras | None
    ) -> None:
        """
        Add the light of the galaxies to the output array (see evaluate),
        with the parameters of the observation group (group_params: e.g.
        the separations of the galaxies of a group, see DepthUnit), those
        of each galaxy (by its own names), and what it gives its
        components, their states (states, see GalaxyState); with extras,
        the extra outputs of each galaxy go to its item, and those of each
        group of galaxies to group_extras, with its name as the id
        'group'.
        """
        pass


class AdditiveScenePlan(ScenePlan):
    """
    The evaluation of a scene whose galaxies add their light, each dimmed
    by its own opacity only (see SceneEvaluation, PLANE and DEPTH): on the
    given depth axis, if any (for the largest disk of the scene). With
    extra outputs, it logs a warning if the depth axis resolves the
    velocity of the light too coarsely (see warn_resolution).
    """

    def __init__(
            self,
            scene: Scene,
            driver: Driver,
            grid: Grid,
            depth: DepthAxis | None,
            dtype: np.dtype,
            smoothing: float | None
    ):
        target = scene.target(driver, grid, dtype, smoothing)
        super().__init__(scene, grid, target)
        # (the dust of each galaxy alone, see opacity_cubes)
        cubes = [
            opacity_cubes(
                [galaxy], [emitters], target.view, dtype)[0]
            for galaxy, emitters in zip(scene._galaxies, scene._emitters)]
        self._plans: tuple[GalaxyPlan, ...] = tuple(
            galaxy.plan(target, emitters, galaxy_cubes, depth)
            for galaxy, emitters, galaxy_cubes in zip(
                scene._galaxies, scene._emitters, cubes))
        self._dust = [galaxy_cubes.model for galaxy_cubes in cubes]
        self._depth = depth

    @staticmethod
    def check(scene: Scene) -> None:
        """
        Check, without a driver, that a scene can be evaluated so: its
        galaxies each dimmed by its own dust (see opacity_cubes).

        Raises
        ------
        ValueError
            If it cannot.
        """
        for galaxy, emitters in zip(scene._galaxies, scene._emitters):
            opacity_cubes(
                [galaxy], [emitters], scene.view(), np.float64)

    def _evaluate(
            self,
            group_params: ParamValues,
            params: list[ParamValues],
            states: Sequence[dict[int, ComponentState]],
            outputs: dict[str, DeviceArray],
            extras: list[Extras] | None,
            group_extras: Extras | None
    ) -> None:
        # (the dust of each galaxy, of its systemic velocities)
        for g, dust in enumerate(self._dust):
            if dust is not None:
                dust.update([params[g]])
        resolutions = [
            plan.evaluate(
                params[g], states[g], outputs,
                extras[g] if extras is not None else None)
            for g, plan in enumerate(self._plans)]
        if extras is not None and self._depth is not None:
            warn_resolution(
                LineResolution(*np.max(resolutions, axis=0)),
                self._spectral.coords.step[0], 1, self._depth.step)


class TransferScenePlan(ScenePlan):
    """
    The evaluation of a scene by radiative transfer along each line of
    sight (see SceneEvaluation, TRANSFER), unit by unit, front to back,
    in the order of the scene (see DepthUnit): a galaxy on its depth axis
    (the given one of each galaxy, for its largest disk), or a group of
    galaxies on one axis, each at its position (see TransferPlan), from
    the opacity of the units in front of it. One set
    of cubes of opacity holds the dust of all of them (see opacity_cubes).
    (The light of galaxies without opacity just adds up, in any order.)

    With extra outputs, it logs a warning if the march resolves line
    opacity, or the depth axis the velocity of the light, too coarsely
    (see warn_resolution).
    """

    @staticmethod
    def check(scene: Scene) -> None:
        """
        Check, without a driver, that a scene can be evaluated so: the
        dust of all its galaxies in one set of cubes of opacity (see
        opacity_cubes), and the components that emit in each unit (see
        _transfer.check_emitters).

        Raises
        ------
        ValueError
            If it cannot.
        """
        opacity_cubes(
            scene._galaxies, scene._emitters, scene.view(),
            np.float64)
        for unit in scene._order:
            check_emitters(
                sum(len(scene._emitters[g]) for g in unit.galaxies))

    def __init__(
            self,
            scene: Scene,
            driver: Driver,
            grid: Grid,
            depths: Sequence[DepthAxis],
            dtype: np.dtype,
            slab: int | None,
            subchannels: int,
            smoothing: float | None
    ):
        target = scene.target(driver, grid, dtype, smoothing)
        super().__init__(scene, grid, target)
        spatial = self._spatial
        cubes = opacity_cubes(
            scene._galaxies, scene._emitters, target.view,
            dtype)
        # (the dust of all the galaxies, which dims the light of those
        # behind it: one model, see opacity_cubes)
        self._dust = cubes[0].model
        members = [
            TransferMember(galaxy, tuple(emitters), galaxy_cubes, depth)
            for galaxy, emitters, galaxy_cubes, depth in zip(
                scene._galaxies, scene._emitters, cubes, depths)]
        self._plans: tuple[TransferPlan, ...] = tuple(
            TransferPlan(
                [members[g] for g in unit.galaxies], target, slab,
                subchannels)
            for unit in scene._order)
        self._subchannels = subchannels
        # (that of every galaxy)
        self._depth_step = members[0].depth.step
        # The state of the march from one unit to the next: the optical
        # depth of the dust in front, in each cube of opacity, and with
        # line opacity, that of the lines at each sub-channel
        self._driver = driver
        shape = (spatial.size[1], spatial.size[0])
        self._opacity_front = None
        if cubes[0].ncubes:
            self._opacity_front = driver.mem_alloc_d(
                (cubes[0].ncubes,) + shape, dtype)
        self._tau = None
        if any(plan.has_line_opacity() for plan in self._plans):
            self._tau = driver.mem_alloc_d(
                (self._spectral.size[0] * subchannels,) + shape, dtype)

    def _evaluate(
            self,
            group_params: ParamValues,
            params: list[ParamValues],
            states: Sequence[dict[int, ComponentState]],
            outputs: dict[str, DeviceArray],
            extras: list[Extras] | None,
            group_extras: Extras | None
    ) -> None:
        for state in (self._opacity_front, self._tau):
            if state is not None:
                self._driver.mem_fill(state, 0)
        if self._dust is not None:
            self._dust.update(params)
        # Each unit marches from the dust and the line opacity in front of
        # it, if any; with extras, its march tells how finely it resolves
        # its lines, and the worst gets a warning. The extras of the march
        # of a galaxy are its own; those of a group, the group's.
        dust = line_opacity = False
        resolutions = []
        for unit, plan in zip(self._scene._order, self._plans):
            dust = dust or plan.has_dust()
            line_opacity = line_opacity or plan.has_line_opacity()
            unit_extras = march_extra = None
            if extras is not None:
                unit_extras = [extras[g] for g in unit.galaxies]
                march_extra = unit_extras[0] if unit.name is None \
                    else group_extras.scoped(group=unit.name)
            positions = [
                0.0 if name is None else float(group_params[name])
                for name in unit.separations]
            resolution = plan.evaluate(
                [params[g] for g in unit.galaxies],
                [states[g] for g in unit.galaxies], outputs, unit_extras,
                march_extra, self._opacity_front if dust else None,
                self._tau if line_opacity else None, positions)
            if resolution is not None:
                resolutions.append(resolution)
        if resolutions:
            warn_resolution(
                LineResolution(*np.max(resolutions, axis=0)),
                self._spectral.coords.step[0], self._subchannels,
                self._depth_step)


def _bounding_radius(galaxy: 'Galaxy', emitters: Sequence[int]) -> float:
    """
    Return the radius (arcsec) of the sphere around the centre of a galaxy
    that holds all the density of the components that an observation
    evaluates (of the given emitters, and the components with dust) that
    have one, at any inclination (see Component.bounding_radius); 0
    without.
    """
    radii = [
        cmp.bounding_radius()
        for i, cmp in enumerate(galaxy.components())
        if i in emitters or i in galaxy.absorbers()]
    return max(radii, default=0.0)


def named_components(
        galaxies: Sequence['Galaxy'], names: Sequence[str]
) -> dict[str, tuple[int, int]]:
    """
    Return the indices of the galaxy and of the component of each named
    component of galaxies drawn together, by its name ('galaxy.component'
    when there are several galaxies).

    Raises
    ------
    ValueError
        If a given name is not among them.
    """
    known = {}
    for g, galaxy in enumerate(galaxies):
        for i, cmp in enumerate(galaxy.components()):
            if cmp.name() is None:
                continue
            key = cmp.name() if len(galaxies) == 1 \
                else f'{galaxy.name()}.{cmp.name()}'
            known[key] = (g, i)
    if unknown := [name for name in names if name not in known]:
        raise ValueError(
            f"there are no components named {unknown}; the named components "
            f"are {list(known)}")
    return known
