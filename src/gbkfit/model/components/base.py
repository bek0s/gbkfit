import abc
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypedDict

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .geometries import Geometry
from .lines import Band
from .roles import Emission, LineEmission, Opacity

if TYPE_CHECKING:
    from ..extras import Extras
    from ..mass import ComponentDensity
    from ._needs import Need
    from ._stage import GalaxyMass
    from ._units import DataUnit
    from .disks._disk import NodalDisk, DiskPlan


__all__ = [
    'Component',
    'ComponentPlan',
    'ComponentStage',
    'DepthAxis',
    'FieldsLayout',
    'FrameParams',
    'NativeGrid',
    'OpacityCubes'
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
    there, and for each component whose light is evaluated (an emitter, by
    index), the ratios of the optical depths of the cubes to theirs at the
    wavelength of each of its lines or its band (an array of shape (nrows,
    ncubes)), or None without dust.
    """
    ncubes: int
    deposits: dict[int, tuple[int, float]]
    ratios: dict[int, np.ndarray | None]


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


class Component(parseutils.TypedSerializable, abc.ABC):
    """
    A component of a galaxy: a shape (the subclass) with roles, the light
    it emits (emission) and what it removes from the light behind it
    (opacity). It needs at least one.

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters in its galaxy (required
        when its galaxy has several components).
    geometry : Geometry, 'galaxy' or 'own'
        The geometry that it shares with other components: that of its
        galaxy ('galaxy'; or the Geometry itself, which a warped one must
        be, see Galaxy), a named one, or none ('own': its geometric
        parameters are its own).
    emission : Emission, optional
        The light it emits.
    opacity : Opacity, optional
        What it removes from the light behind it.

    Raises
    ------
    ConfigError
        If the name is not valid (see parseutils.check_name), the geometry
        is not one of the above, or it has neither emission nor opacity.

    Notes
    -----
    In configurations, a component is loaded with the catalogue of the
    lines its emission can name: load(info, catalogue) (see
    Emission.load).
    """

    def __init__(
            self,
            name: str | None,
            geometry: Geometry | Literal['galaxy', 'own'],
            emission: Emission | None,
            opacity: Opacity | None
    ):
        parseutils.check_name(name)
        if not isinstance(geometry, Geometry) \
                and geometry not in ('galaxy', 'own'):
            raise ConfigError(
                f"geometry must be 'galaxy', 'own' or a geometry; it is "
                f"{geometry!r}")
        if emission is None and opacity is None:
            raise ConfigError(
                "a component needs emission, opacity or both")
        self._name = name
        self._geometry = geometry
        self._emission = emission
        self._opacity = opacity

    def name(self) -> str | None:
        """Return its name, if any."""
        return self._name

    def geometry(self) -> Geometry | Literal['galaxy', 'own']:
        """Return the geometry it shares, 'galaxy' or 'own'."""
        return self._geometry

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
    def thick(self) -> bool:
        """
        Check whether it is thick: whether it needs a line of sight with a
        depth axis.
        """
        pass

    @abc.abstractmethod
    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name."""
        pass

    def constants(self) -> dict[str, Any]:
        """
        Return the values that parameter expressions can use (e.g. the
        radial nodes of a disk), by name; none here.
        """
        return {}

    def line_names(self) -> tuple[str, ...]:
        """Return the names of its emission lines, if any."""
        if isinstance(self._emission, LineEmission):
            return self._emission.lines().names()
        return ()

    def needs(self) -> tuple['Need', ...]:
        """
        Return what it needs from its galaxy (e.g. its mass), each of which
        checks itself against the galaxy when the galaxy is made (see
        Need); nothing here.
        """
        return ()

    def uses_mass(self) -> bool:
        """
        Check whether it takes anything from the mass of its galaxy, when
        there is one (e.g. its rotation in the gravity of the mass); False
        here.
        """
        return False

    def bounding_radius(self) -> float:
        """
        Return the radius (arcsec) of the sphere around its centre that
        holds all its density, at any inclination: the depth axis of a
        grid holds it (see Depth.axis); 0 here.
        """
        return 0.0

    def extent_cut(self, params: ParamValues) -> tuple[str, float] | None:
        """
        Return the option that bounds its extent on a grid with a depth
        axis (e.g. 'rmax'), if one does, and the share of its light that it
        cuts at the values of its parameters; None here.
        """
        return None

    def density_disk(self) -> 'NodalDisk | None':
        """
        Return its disk, if it is made of one: the mass of its galaxy can
        follow the density of the disk (see mass.SurfaceComponent and
        mass.VerticalComponent). None here.
        """
        return None

    def density_kind(self) -> 'type[ComponentDensity] | None':
        """
        Return the kind of the density that it gives the mass of its
        galaxy to follow (e.g. mass.DiskDensity; see
        mass.MassComponent.followed_kind), if any; none here.
        """
        return None

    def plan_stage(
            self,
            driver: Driver,
            sky_origin: tuple[float, float] | None,
            followed: bool,
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
        followed : bool
            Whether the mass of the galaxy follows its density.
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
            driver: Driver,
            spectral: gridutils.Grid,
            dtype: np.dtype,
            lines: tuple[str, ...] | None,
            band: Band | None,
            emits: bool,
            opacity_ratios: np.ndarray | None,
            data_unit: 'DataUnit | None',
            redshift: float
    ) -> 'ComponentPlan':
        """
        Plan the evaluation of the component; the plan owns the memory
        it needs.

        Parameters
        ----------
        driver : Driver
            The driver it is evaluated on.
        spectral : Grid
            The spectral axis of the outputs (one axis).
        dtype : np.dtype
            The floating type of the evaluation.
        lines : tuple of str, optional
            The names of the emission lines to evaluate; all if None.
        band : Band, optional
            The band of the image it is evaluated in, if known.
        emits : bool
            Whether its light is evaluated (an observation sees it), or
            only its opacity.
        opacity_ratios : np.ndarray, optional
            If opacity dims its light: the ratio of the optical depth of
            each cube of opacity to that of the cube at the rest wavelength
            of each of its lines that it evaluates, in their order, or of
            its band (an array of shape (nrows, ncubes); see
            ComponentSetPlan).
        data_unit : DataUnit, optional
            The unit of the data of the observation, which its light is
            turned into from the unit of its emission, if it has one (see
            _units.LineFactors and BandFactors); None without data: its
            light stays in the unit of its emission.
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
        ConfigError
            If its light has a unit that cannot become that of the data
            (see _units.LineFactors and BandFactors).
        """
        pass


class ComponentStage(abc.ABC):
    """
    The part of a component in the stage of its galaxy (see GalaxyStage):
    what the galaxy computes for it once per evaluation, for all the
    observations of the galaxy, from the values of its parameters and the
    mass of the galaxy.
    """

    @abc.abstractmethod
    def values(self, own: ParamValues) -> ParamValues:
        """
        Return the values of its parameters, from those of its own
        parameters (by its own names), with those it makes before the mass
        of the galaxy (e.g. the light of image traits): those that all its
        part reads.
        """
        pass

    def density(
            self, values: ParamValues, kpc_per_arcsec: float
    ) -> 'ComponentDensity | None':
        """
        Return its density at the values of its parameters (see values),
        which the mass of its galaxy can follow (see mass.MassContext), in
        a galaxy of the given length of an arcsec (kpc); None here.
        """
        return None

    @abc.abstractmethod
    def given(
            self,
            values: ParamValues,
            mass: 'GalaxyMass | None',
            out_extra: 'Extras | None'
    ) -> ParamValues:
        """
        Return what its galaxy gives it at the values of its parameters
        (see values), in the mass of the galaxy (None without mass): the
        values of the parameters that it takes from the galaxy, by their
        names. Its extra outputs, if wanted, go to out_extra (with the id
        of the component).
        """
        pass

    def rings(self, values: ParamValues) -> list[tuple[float, ...]]:
        """
        Return its rings that the gravity of its galaxy takes as concentric
        and flat (see GalaxyStage), at the values of its parameters: the
        radius, the centre and the plane of each (radius, xpos, ypos, posa,
        incl; arcsec and degrees); none here.
        """
        return []


class ComponentPlan(abc.ABC):
    """
    The evaluation of a component on a driver and dtype, in two stages:
    the opacity of all the components of a galaxy, then the light of
    those an observation sees, dimmed by that opacity; or, for the march
    of radiative transfer, the fields of the light instead of the light.
    """

    def disk_plan(self) -> 'DiskPlan | None':
        """Return the plan of its disk, if it is made of one; none here."""
        return None

    @abc.abstractmethod
    def evaluate_emission(
            self,
            params: ParamValues,
            grid: NativeGrid,
            outputs: dict[str, DeviceArray | None],
            out_extra: dict[str, Any] | None
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
            out_extra: dict[str, Any] | None
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
            out_extra: dict[str, Any] | None
    ) -> None:
        """
        Add the optical depth of the component times scale to that of
        each voxel of a cube of opacity (odata); its extra output 'odata'
        has its optical depth (at V). Only components with opacity have
        it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no opacity to evaluate")
