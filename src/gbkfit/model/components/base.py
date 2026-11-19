"""
The components of a galaxy (see Component): their base, their stages
and plans, and what the plans evaluate on (the grid and its depth axis,
the cubes of opacity, the fields of radiative transfer).
"""

import abc
import dataclasses
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING, Any, ClassVar, NamedTuple, Protocol, TypedDict,
    runtime_checkable)

import numpy as np

from gbkfit.config import TypedSerializable, check_name
from gbkfit.driver import DeviceArray, Driver
from gbkfit.grid import Grid
from gbkfit.params import ParamDesc, ParamValues
from .roles import (
    Emission, Opacity, View)

if TYPE_CHECKING:
    from gbkfit.instrument import PSF

    from ..extras import Extras
    from ..gravity import GravityField
    from ..mass import ComponentDensity, MassContext, MassModel
    from ..mass.sky import RingPlacement, SpheroidAxis
    from ._needs import Need
    from ._profiles import ComponentProfiles
    from ._units import DataUnit


__all__ = [
    'Balanced',
    'BalancedLayer',
    'Component',
    'ComponentLinks',
    'ComponentPart',
    'ComponentPlan',
    'ComponentStage',
    'ComponentState',
    'DepthAxis',
    'FieldsLayout',
    'Frame',
    'FrameParams',
    'GalaxyMass',
    'NativeGrid',
    'OpacityCubes',
    'Placed',
    'Ringed',
    'SceneTarget',
    'SkyPlacements'
]


class DepthAxis(NamedTuple):
    """
    An axis along the line of sight (z): its number of voxels, their size
    and the position of the first (arcsec).
    """
    size: int
    step: float
    zero: float


class OpacityCubes(NamedTuple):
    """
    How the plan of a galaxy holds the dust that dims its light, in the
    cubes of opacity of its observation (see opacity_cubes): their number
    (0 without dust), the cube of each component of the galaxy with dust
    (an absorber, by index) and the factor of its optical depth at V
    there (an array of one value), and for each component whose light is
    evaluated (an emitter, by index), the ratios of the optical depths of
    the cubes to theirs at the wavelength of each of its lines or its band
    (an array of shape (nrows, ncubes)), or None without dust. The factors
    and the ratios follow the systemic velocities of the components: the
    model of the dust (None without dust) changes the arrays in place at
    each evaluation (its method update, of the values of the parameters
    of the galaxies; see _opacity._DustModel), which the plans
    read then.
    """
    ncubes: int
    deposits: dict[int, tuple[int, np.ndarray]]
    ratios: dict[int, np.ndarray | None]
    model: Any = None


class NativeGrid(TypedDict):
    """
    The grid that component plans evaluate on, as the native evaluation
    functions take it: the 3d spatial grid (each in x, y, z order; the
    rotation in degrees) and the spectral axis. Without a depth axis, the
    z axis has size 1 (and step 0), and so does the spectral axis of an
    image. The grid can be a slab of planes of the depth axis: its planes
    from the plane slab_zero (0 for the whole axis) of an axis whose
    first plane is at spat_zero[2].
    """
    spat_size: tuple[int, int, int]
    spat_step: tuple[float, float, float]
    spat_zero: tuple[float, float, float]
    spat_rota: float
    slab_zero: int
    spec_size: int
    spec_step: float
    spec_zero: float


class FieldsLayout(NamedTuple):
    """
    How an emitter holds its fields for the march of radiative transfer
    (see the native FieldsArgs::fields): nslots values in each voxel of the
    3d grid (a volume, e.g. a thick disk: j, v, s, and with line opacity,
    the light times the coefficient of each line), or in each spaxel (a
    layer at a depth along each line of sight: j, v, s and the position
    along the depth axis, in voxels; e.g. a thin disk or a point).
    """
    layer: bool
    nslots: int
    line_opacity: bool = False

    def shape(self, grid: NativeGrid) -> tuple[int, ...]:
        """Return the shape of the fields on a grid."""
        size_x, size_y, size_z = grid['spat_size']
        planes = (size_y, size_x) if self.layer else (size_z, size_y, size_x)
        return (self.nslots,) + planes


class FrameParams(NamedTuple):
    """
    The parameters of a component that place it in a frame, by their
    names among its own (see Component.pdescs; None for those it does not
    have): its centre (x, y), its systemic velocity (vsys), and the
    orientation of its plane or axis (posa, incl).
    """
    x: str | None
    y: str | None
    vsys: str | None
    posa: str | None
    incl: str | None


@dataclasses.dataclass(frozen=True)
class SceneTarget:
    """
    What one scene of an observation asks of each galaxy that it
    evaluates, and of their components (see Scene.target): the driver of
    the evaluation, the grid of its x and y axes (spatial) and its
    spectral axis (one axis), the floating type, what the observation
    sees of the light (see View: its band, or its lines and continuum),
    the unit of the data of the observation, which the light
    takes (None without data; see Component.plan), the dispersion
    (km/s) by which the instrument smooths the spectra of the model (None
    for an image; see LineResolution), and whether the grid is the source
    plane of a gravitational lens (lensed), whose image the lens draws by
    sampling it at the rays of the image (see gbkfit.lensing): light drawn
    on it must be a surface brightness that the grid resolves.
    """
    driver: Driver
    spatial: Grid
    spectral: Grid
    dtype: np.dtype
    view: View
    data_unit: 'DataUnit | None'
    smoothing: float | None
    lensed: bool = False


@dataclasses.dataclass(frozen=True)
class ComponentLinks:
    """
    The links of a component to the mass of its galaxy (see
    MassLinks): the names of the mass components that follow its density
    (followers: by its surface density or its layer), whether its galaxy
    has mass (mass), and whether a mass disk follows its layer, its
    vertical structure (layer).
    """
    followers: tuple[str, ...] = ()
    mass: bool = False
    layer: bool = False

    @property
    def followed(self) -> bool:
        """Whether the mass follows its density."""
        return bool(self.followers)


class ComponentPart(NamedTuple):
    """
    A part of a type of component (see Component.parts): one of the
    options of its configurations, whose value is an item of a family or
    a list of them (e.g. its velocity traits), or its radial nodes (see
    Nodes); with the option of the height traits paired with its traits,
    if any (e.g. those of its velocity traits): one for each trait, or
    one that they all share.
    """
    option: str
    heights: str | None = None


class Component(TypedSerializable, abc.ABC):
    """
    A component of a galaxy: a shape (the subclass) with roles, the light
    it emits (emission) and what it removes from the light behind it
    (opacity). It needs at least one.

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters in its galaxy (required
        when its galaxy has several components).
    emission : Emission, optional
        The light it emits.
    opacity : Opacity, optional
        What it removes from the light behind it.

    Raises
    ------
    ValueError
        If the name is not valid (see check_name), or it has neither
        emission nor opacity.
    """

    # (the titles of the options of its types, those that several share)
    config_titles = dict(
        name='Name', rnodes='Rings', rptraits='Density',
        rhtraits='Density heights', vptraits='Velocity',
        vhtraits='Velocity heights', dptraits='Dispersion',
        dhtraits='Dispersion heights', zptraits='Warps', sptraits='Selections',
        emission='Light', opacity='Opacity', rstep='Ring width',
        interp='Interpolation')
    config_start = dict(emission=dict(type='band'))

    # Its parts, in the order people read them (e.g. its density, then its
    # velocity): the options that it is made of; its other options are its
    # settings (e.g. how its parameters are interpolated)
    parts: ClassVar[tuple[ComponentPart, ...]] = (ComponentPart('emission'),)

    def __init__(
            self,
            name: str | None,
            emission: Emission | None,
            opacity: Opacity | None
    ):
        check_name(name)
        if emission is None and opacity is None:
            raise ValueError(
                "a component needs emission, opacity or both")
        self._name = name
        self._emission = emission
        self._opacity = opacity

    def name(self) -> str | None:
        """Return its name, if any."""
        return self._name

    @abc.abstractmethod
    def frame_params(self) -> FrameParams:
        """Return its parameters that place it in a frame."""

    def emission(self) -> Emission | None:
        """Return the light it emits, if any."""
        return self._emission

    def opacity(self) -> Opacity | None:
        """Return what it removes from the light behind it, if anything."""
        return self._opacity

    @abc.abstractmethod
    def needs_depth(self, view: View) -> bool:
        """
        Check whether an observation of the given view (see View) needs a
        line of sight with a depth axis to see it (e.g. a thick disk; a
        spheroid in a spectral view, whose velocities change along the
        line of sight).
        """

    def check_psf(self, psf: 'PSF | None') -> None:
        """
        Check that it can be seen through the PSF of an observation (None
        without one), warning of what it sees badly (see warn; e.g. the
        light of an image already blurred by a PSF of its own, see
        traits.RPTraitImage.check_psf); nothing here.
        """

    @abc.abstractmethod
    def moves(self) -> bool:
        """
        Check whether it has velocities that a spectral axis sees (its
        kinematics, or the systemic velocity of a point of lines): a
        spectral view of the continuum sees the light in bands of the
        components that move (see Emission.seen_in).
        """

    @abc.abstractmethod
    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name."""
        pass

    def param_owners(self) -> dict[str, tuple[str | int, ...]]:
        """
        Return the option of its configuration that makes each of its
        parameters (see pdescs), by name, as a path of option names and
        list indices: e.g. ('vptraits', 1) for those of its
        second velocity polar trait, ('emission', 'lines', 2) for the flux
        ratio of its third line, and () for those of the component itself
        (e.g. its centre). For those who show the parameters with the
        options that make them (e.g. Studio). Here, the ratios of its lines
        or bands, and () for the others.
        """
        owners: dict[str, tuple[str | int, ...]] = dict.fromkeys(
            self.pdescs(), ())
        emission = self.emission()
        if emission is not None:
            light = emission.light()
            for index, name in enumerate(light.names()[1:], 1):
                if f'{name}_ratio' in owners:
                    owners[f'{name}_ratio'] = (
                        'emission', light.option, index)
        return owners

    def constants(self) -> dict[str, Any]:
        """
        Return the values that parameter expressions can use (e.g. the
        radial nodes of a disk), by name; none here.
        """
        return {}

    def needs(
            self, links: ComponentLinks = ComponentLinks()
    ) -> tuple['Need', ...]:
        """
        Return what it needs from its galaxy (e.g. its mass), of its links
        to the mass of the galaxy (e.g. its fluid; none before the galaxy
        links it), each of which checks itself against the galaxy when the
        galaxy is made (see Need); nothing here.
        """
        return ()

    def light_amplitudes(self) -> dict[str, str | None]:
        """
        Return its parameters whose elements are amplitudes of its light,
        by name: its light is the sum of parts, each times one element
        (e.g. the Gaussians of an MGE), so that models of it can be linear
        in them, and solve them (see params.Solve). Each with None if its
        models are linear in it, or why not (e.g. kinematics that take its
        light). None by default.
        """
        return {}

    def orbit_weights(self) -> str | None:
        """
        Return its parameter of the weights of the orbits of its stars, if
        they move on orbits whose weights are solved to fit the data (e.g.
        a spheroid of Schwarzschild kinematics; see objective.Objective,
        which solves them); None by default. Its part in the stage of its
        galaxy does not read them (see ComponentStage): only the
        observations of the orbits weigh the orbits by them (see
        observation.orbits.OrbitRecording).
        """
        return None

    def bounding_radius(self) -> float:
        """
        Return the radius (arcsec) of the sphere around its centre that
        holds all its density, at any inclination, if it has one: the
        depth axis of a grid holds it (see Depth.axis); 0 here, and for
        disks and cones, which do not end (see Disk; the depth axis holds
        what the field sees of them).
        """
        return 0.0

    def extent_cut(self, params: ParamValues) -> tuple[str, float] | None:
        """
        Return the option that bounds its extent on a grid with a depth
        axis (e.g. 'rmax'), if one does, and the share of its light that it
        cuts at the values of its parameters; None here.
        """
        return None

    def profile_families(self) -> dict[str, tuple[float, float]]:
        """
        Return the families of its radial profiles (see ComponentProfiles:
        'rotation', 'speed', 'dispersion', 'density', 'orientation'), each
        with the range of its radii (arcsec), without planning them; none
        here.
        """
        return {}

    def profiles(self, driver: Driver) -> 'ComponentProfiles | None':
        """
        Return its radial profiles (see profile_families), planned on the
        given driver in float64, if it has any; none here.
        """
        return None

    def density_kind(self) -> 'type[ComponentDensity] | None':
        """
        Return the kind of the density that it gives the mass of its
        galaxy to follow (e.g. mass.DiskDensity; see
        mass.MassComponent.followed_kind), if any; none here.
        """
        return None

    def density_has_height(self) -> bool:
        """
        Check whether the density that it gives the mass of its galaxy to
        follow has a height (that of a thick disk, see
        mass.DiskDensity.thick); not here.
        """
        return False

    def plan_stage(
            self,
            driver: Driver,
            sky_origin: tuple[float, float] | None,
            links: 'ComponentLinks',
            label: str
    ) -> 'ComponentStage | None':
        """
        Plan its part in the stage of its galaxy (see GalaxyStage), if it
        takes anything from the galaxy or the mass of the galaxy follows it;
        None here (it takes nothing, and offers no density).

        Parameters
        ----------
        driver : Driver
            The driver where its part is evaluated.
        sky_origin : tuple of float, optional
            The RA and Dec (degrees) of the origin of the frame of the
            parameters of the galaxy (see ObservationGroup).
        links : ComponentLinks
            Its links to the mass of the galaxy: e.g. whether the mass
            follows its density.
        label : str
            Its name, or its position (e.g. 'component0'), for messages.

        Returns
        -------
        ComponentStage or None
            Its part.
        """
        return None

    @abc.abstractmethod
    def plan(
            self,
            target: 'SceneTarget',
            emits: bool,
            opacity_ratios: np.ndarray | None,
            redshift: float
    ) -> 'ComponentPlan':
        """
        Plan the evaluation of the component; the plan owns the memory
        it needs.

        Parameters
        ----------
        target : SceneTarget
            What the scene asks of it: the driver, the spectral axis of the
            outputs, the floating type, the emission lines to evaluate
            (all if None), the band of an image, if known, and the unit of
            the data of the observation, which its light is turned into
            from the unit of its emission, if it has one (see
            _units.LineFactors and BandFactors; None without data: its
            light stays in the unit of its emission). A plan that does not
            emit takes no unit.
        emits : bool
            Whether its light is evaluated (an observation sees it), or
            only its opacity.
        opacity_ratios : np.ndarray, optional
            If opacity dims its light: the ratio of the optical depth of
            each cube of opacity to that of the cube at the rest wavelength
            of each of its lines that it evaluates, in their order, or of
            its band (an array of shape (nrows, ncubes); see
            GalaxyPlan).
        redshift : float
            The redshift of its galaxy, which places its named lines on
            the spectral axis (see Lines.factors) and, with a unit, dims its
            light (see _units.line_factor).

        Returns
        -------
        ComponentPlan
            The plan.

        Raises
        ------
        ValueError
            If its light has a unit that cannot become that of the data
            (see _units.LineFactors and BandFactors).
        """
        pass


class ComponentState:
    """
    What the stage of a galaxy computes for one of its components in an
    evaluation, which the plans of the component read (see
    ComponentStage.state): the base of such states (e.g. what a disk takes
    from its galaxy, or the tables of the Jeans equations of a
    spheroid).
    """


@dataclasses.dataclass(frozen=True)
class GalaxyMass:
    """
    The mass of a galaxy in one evaluation: its model, the values of the
    parameters of the mass and the gravity of the galaxy, the context of
    the mass (with the length of an arcsec in the galaxy, kpc; None
    without a distance, which only what takes it needs, see
    Capability.DISTANCE), the field of the mass in the gravity of the
    galaxy (see Gravity; None while the mass is built, see
    ComponentStage.thickness), and the values of the parameters of the
    layers that the field sets and the mass follows, by the names of
    their components (see Balanced; none without them).
    """
    model: 'MassModel'
    params: ParamValues
    context: 'MassContext'
    field: 'GravityField'
    layers: Mapping[str, ParamValues] = dataclasses.field(
        default_factory=dict)


class ComponentStage(abc.ABC):
    """
    The part of a component in the stage of its galaxy (see GalaxyStage):
    what the galaxy computes for it once per evaluation, for all the
    observations of the galaxy, from the values of its parameters and the
    mass of the galaxy, in three steps, each of the values of the
    parameters it declares (see GalaxyStage): its light (see values), the
    groundwork of its state (see groundwork) and its state (see state).
    """

    @abc.abstractmethod
    def light_params(self) -> tuple[str, ...]:
        """
        Return the names of its parameters (its own names) that its light
        reads (see values and density).
        """
        pass

    @abc.abstractmethod
    def values(self, own: ParamValues) -> ParamValues:
        """
        Return the values of the parameters of its light, from those of
        its parameters that its light reads (see light_params; by its own
        names), with those it makes of them (e.g. the light of image
        traits).
        """
        pass

    def density(
            self, values: ParamValues, kpc_per_arcsec: float
    ) -> 'ComponentDensity | None':
        """
        Return its density at the values of the parameters of its light
        (see values), which the mass of its galaxy can follow (see
        mass.MassContext), in a galaxy of the given length of an arcsec
        (kpc); None here.
        """
        return None

    def thickness(
            self, values: ParamValues, mass: 'GalaxyMass'
    ) -> ParamValues:
        """
        Return the values of the parameters of its thickness that the mass
        of its galaxy sets, by name (e.g. the scale heights of the
        vertical equilibrium of a disk), at the values of the parameters
        of its light (see values), in the mass of its galaxy as it is
        built (its model, parameters, context and length of an arcsec,
        without its field): its density of them is what the mass that
        follows it takes (see density); none here.
        """
        return {}

    def state_params(self) -> tuple[str, ...]:
        """
        Return the names of its parameters (its own names) that its state
        reads besides those of its light (see state); none here.
        """
        return ()

    def groundwork(
            self, values: ParamValues, mass: 'GalaxyMass | None'
    ) -> object:
        """
        Return the groundwork of its state: what the values of the
        parameters of its light (see values) and the mass of its galaxy
        (None without mass) alone set of it (e.g. the forces in the Jeans
        equations of a spheroid), so that a change of the other parameters
        of its state does not compute it again; none here.
        """
        return None

    def state(
            self,
            values: ParamValues,
            groundwork: object,
            mass: 'GalaxyMass | None',
            out_extra: 'Extras | None'
    ) -> ComponentState | None:
        """
        Return what its galaxy gives it at the values of the parameters of
        its light (see values) and of those its state reads (see
        state_params), of its groundwork (see groundwork), in the mass of
        the galaxy (None without mass): its state, which its plans read
        (see ComponentPlan.evaluate_emission); none here. Its extra
        outputs, if wanted, go to out_extra (with the id of the component).
        """
        return None


class BalancedLayer(Protocol):
    """
    The layer of a component in one evaluation, whose thickness is its
    vertical equilibrium in the field of the mass of its galaxy, while
    the mass follows it (see Balanced): of its height potential
    Phi(R, z) - Phi(R, 0) at the points of a grid (see grid), through a
    dimensionless unknown, e.g. that divided by the square of the vertical
    dispersion (see unknown), which a sum of such unknowns keeps a layer.
    """

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the radii and the heights (kpc, increasing from 0) of the
        grid of its height potential.
        """

    def unknown(self, potential: np.ndarray) -> np.ndarray:
        """
        Return its unknown of its height potential ((km/s)^2, an array of
        the shape of its grid): finite everywhere, the same as of a
        uniform layer where it has no light.

        Raises
        ------
        ModelEvaluationError
            If the potential gives no layer where it has light.
        """

    def change(self, unknown: np.ndarray, new: np.ndarray) -> tuple[
            float, float]:
        """
        Return the change of its layer from an unknown to a new one: the
        largest change of the pdf of its height at a radius, relative to
        its peak there, and that radius (arcsec).
        """

    def values(self, unknown: np.ndarray) -> ParamValues:
        """
        Return the values of the parameters of its thickness, of an
        unknown, by name: those that its density takes (see
        ComponentStage.density) and its state draws.
        """


@runtime_checkable
class Balanced(Protocol):
    """
    The part of a component in the stage of its galaxy (see
    ComponentStage) whose thickness can be its vertical equilibrium in
    the field of the mass of the galaxy while the mass follows it (e.g.
    the vertical equilibrium 'galaxy' of a disk that a mass disk
    follows): the galaxy solves the two together (see GalaxyStage).
    """

    def balanced(self) -> bool:
        """
        Check whether its thickness is its vertical equilibrium in the
        field of the mass while the mass follows it.
        """

    def layer(
            self, values: ParamValues, kpc_per_arcsec: float
    ) -> BalancedLayer:
        """
        Return its layer (see BalancedLayer) at the values of the
        parameters of its light (see values), in a galaxy of the given
        length of an arcsec (kpc).
        """


@runtime_checkable
class Placed(Protocol):
    """
    The part of a component in the stage of its galaxy (see
    ComponentStage) that says where the parts of the component are: of a
    component that is where the mass of the galaxy is (one that the mass
    moves, holds up or follows, see Galaxy.placed), which the gravity
    takes at one centre (that of the mass on the sky, where it is laid
    out there), and on which the mass components that follow it lie on
    the sky, for the lenses of the mass (see GalaxyStage).
    """

    def placement_params(self) -> tuple[str, ...]:
        """
        Return the names of the parameters of its component (its own
        names) that say where its parts are (see frames): those of its
        geometry (e.g. its centre and orientation, at each ring of a
        disk) and of its displacements, if any.
        """

    def frames(self, values: ParamValues) -> list['Frame']:
        """
        Return where each of its parts is in the frame of the mass of its
        galaxy (see Frame), at the values of the parameters of its
        component (by its own names; those of placement_params among
        them): its centre and the plane (or axis) of each.
        """

    def place(
            self,
            label: str,
            values: ParamValues,
            kpc_per_arcsec: float,
            placement: 'SkyPlacements'
    ) -> None:
        """
        Add where its component is on the sky to the placement of the mass
        of its galaxy (see SkyPlacements), for the mass components that
        follow it: a disk its rings, a spheroid its axis.

        Parameters
        ----------
        label : str
            The name of its component, by which the mass components that
            follow it ask for it (or its position, e.g. 'component0').
        values : ParamValues
            The values of all the parameters of its component (by its own
            names).
        kpc_per_arcsec : float
            The length of an arcsec in its galaxy (kpc).
        placement : SkyPlacements
            The placement.
        """


class SkyPlacements(Protocol):
    """
    The placement of the mass of a galaxy on the sky, as the components
    where the mass is add themselves to it (see Placed.place), by name:
    the rings of each disk and the axis of each spheroid (see
    mass.SkyPlacement).
    """

    def add_rings(self, name: str, rings: 'RingPlacement') -> None:
        """Add the rings of a disk (see mass.RingPlacement), by name."""

    def add_axis(self, name: str, axis: 'SpheroidAxis') -> None:
        """Add the centre and axis of a spheroid, by name."""


@runtime_checkable
class Ringed(Protocol):
    """
    A component of rings (e.g. a disk), whose rings a mass that follows
    its density lays out on the sky (see mass.RingGeometry).
    """

    def rings_gap(self) -> str | None:
        """
        Return why its rings are not ellipses on the sky, if they are not
        (e.g. vertical distortions of an order of 2 or more); None if
        they are.
        """


class Frame(NamedTuple):
    """
    Where a part of a component is in the frame of the mass of its galaxy
    (see Placed.frames): its radius (arcsec; 0 for the whole of a
    component without rings), its centre (x, y; arcsec), and the
    orientation of its plane or axis (posa, incl; degrees), None for a
    part without one (e.g. a sphere).
    """
    radius: float
    x: float
    y: float
    posa: float | None
    incl: float | None


class ComponentPlan(abc.ABC):
    """
    The evaluation of a component on a driver and dtype, in two stages:
    the opacity of all the components of a galaxy, then the light of
    those an observation sees, dimmed by that opacity; or, for the march
    of radiative transfer, the fields of the light instead of the light.
    """

    def warn_truncation(self, params: ParamValues, label: str) -> None:
        """
        Log a warning where an option of the component shapes it rather
        than holds it, at the values of its parameters (e.g. the zmax of a
        height trait of a disk), with its label (its name, or its
        position) in the message; none here.
        """

    @abc.abstractmethod
    def evaluate_emission(
            self,
            params: ParamValues,
            grid: NativeGrid,
            outputs: dict[str, DeviceArray | None],
            out_extra: dict[str, Any] | None,
            state: ComponentState | None
    ) -> None:
        """
        Add the light of the component to the outputs.

        Parameters
        ----------
        params : dict
            The values of its parameters, by name.
        grid : NativeGrid
            The grid of the evaluation.
        outputs : dict
            The arrays the component adds to, which can be left out
            (None): the 'image' or 'scube', and on the 3d grid its
            brightness ('bdata') and brightness after the opacity
            ('obdata'); it reads the optical depth from the viewer to
            the far side of each voxel of each cube of opacity ('odata',
            of shape (ncubes, nz, ny, nx)), which it scales with its
            opacity_ratios (see Component.plan).
        out_extra : dict, optional
            Where its extra outputs (on the host) go, if wanted.
        state : ComponentState, optional
            Its state in the evaluation (see ComponentStage.state).
        """
        pass

    @abc.abstractmethod
    def fields_layout(self) -> FieldsLayout:
        """Return how it holds its fields (see evaluate_fields)."""
        pass

    @abc.abstractmethod
    def evaluate_fields(
            self,
            params: ParamValues,
            grid: NativeGrid,
            fields: DeviceArray,
            out_extra: dict[str, Any] | None,
            state: ComponentState | None
    ) -> None:
        """
        Write the fields of its light that the march of radiative transfer
        reads (see FieldsLayout), instead of its light: its light without
        the flux of its lines (or the flux ratio of its band), its velocity
        and its dispersion, in each voxel or spaxel of the grid (0 where
        there is none).

        Parameters
        ----------
        params : dict
            The values of its parameters, by name.
        grid : NativeGrid
            The grid of the evaluation.
        fields : DeviceArray
            The fields, of the shape of its layout on the grid (see
            FieldsLayout.shape).
        out_extra : dict, optional
            Where its extra outputs (on the host) go, if wanted.
        state : ComponentState, optional
            Its state in the evaluation (see ComponentStage.state).
        """
        pass

    @abc.abstractmethod
    def line_values(self, params: ParamValues) -> np.ndarray:
        """
        Return how the march of radiative transfer spreads the light of its
        fields (see evaluate_fields) along the spectral axis: the offset,
        scale and flux of each of its lines that it evaluates (an array of
        shape (nlines, 3); see the native GridArgs::lines), with the flux
        ratios of the given parameters; in an image, one row of the flux
        of its band (see Bands), whose offset and scale are 0.
        """
        pass

    def evaluate_opacity(
            self,
            params: ParamValues,
            grid: NativeGrid,
            odata: DeviceArray,
            scale: float,
            out_extra: dict[str, Any] | None,
            state: ComponentState | None
    ) -> None:
        """
        Add the optical depth of the component times scale to that of
        each voxel of a cube of opacity (odata), with what its galaxy
        gives it, its state, if any (see ComponentStage.state); its extra
        output 'odata' has its optical depth (at V). Only components with
        opacity have it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no opacity to evaluate")
