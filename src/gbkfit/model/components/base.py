import abc
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .geometries import Geometry
from .lines import Band
from .roles import Emission, LineEmission, Opacity

if TYPE_CHECKING:
    from .disks._disk import Disk, DiskPlan


__all__ = [
    'Component',
    'ComponentPlan',
    'DepthAxis',
    'NativeGrid'
]


class DepthAxis(NamedTuple):
    """
    An axis along the line of sight (z): its number of voxels, their size
    and the position of the first (arcsec).
    """
    size: int
    step: float
    zero: float


class NativeGrid(TypedDict):
    """
    The grid that component plans evaluate on, as the native evaluation
    functions take it: the 3d spatial grid (each in x, y, z order; the
    rotation in degrees) and the spectral axis. Without a depth axis, the
    z axis has size 1 (and step 0), and so does the spectral axis of an
    image.
    """
    spat_size: tuple[int, int, int]
    spat_step: tuple[float, float, float]
    spat_zero: tuple[float, float, float]
    spat_rota: float
    spec_size: int
    spec_step: float
    spec_zero: float


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
    geometry : Geometry, optional
        A geometry that it shares with other components.
    emission : Emission, optional
        The light it emits.
    opacity : Opacity, optional
        What it removes from the light behind it.

    Raises
    ------
    ConfigError
        If the name is not valid (see parseutils.check_name), or it has
        neither emission nor opacity.

    Notes
    -----
    In configurations, a component is loaded with the catalogue of the
    lines its emission can name: load(info, catalogue) (see
    Emission.load).
    """

    def __init__(
            self,
            name: str | None,
            geometry: Geometry | None,
            emission: Emission | None,
            opacity: Opacity | None
    ):
        parseutils.check_name(name)
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

    def geometry(self) -> Geometry | None:
        """Return the geometry it shares, if any."""
        return self._geometry

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

    def circular_velocity_params(self) -> dict[str, tuple[float, ...]]:
        """
        Return the parameters that its galaxy gives it: the circular
        velocity of the mass of the galaxy at the given radii (arcsec), by
        name (see traits.VPTraitMass). They are not in pdescs, but its
        plans need them with the others. None here.
        """
        return {}

    def density_disk(self) -> 'Disk | None':
        """
        Return its disk, if it is made of one: the mass of its galaxy can
        follow the density of the disk (see mass.MassTraitDisk). None
        here.
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
            opacity_ratios: np.ndarray | None
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

        Returns
        -------
        ComponentPlan
            The plan.
        """
        pass


class ComponentPlan(abc.ABC):
    """
    The evaluation of a component on a driver and dtype, in two stages:
    the opacity of all the components of a galaxy, then the light of
    those an observation sees, dimmed by that opacity.
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
