
import abc

from gbkfit.utils import parseutils
from .geometries import Geometry


__all__ = [
    'Component',
    'ComponentPlan',
    'BrightnessComponent2D',
    'BrightnessComponent3D',
    'SpectralComponent2D',
    'SpectralComponent3D',
    'OpacityComponent3D'
]


class Component(parseutils.TypedSerializable, abc.ABC):
    """
    A component of a model. The kinds of components below differ only in
    the models that accept them, and in the outputs they get. A component
    can have a name, which then prefixes its parameters in its model
    instead of its position (see parseutils.item_prefixes), and a
    geometry that it shares with other components (see Geometry).
    """

    def __init__(self, name: str | None, geometry: Geometry | None = None):
        parseutils.check_name(name)
        self._name = name
        self._geometry = geometry

    def name(self) -> str | None:
        return self._name

    def geometry(self) -> Geometry | None:
        """Return the geometry it shares, if any."""
        return self._geometry

    @abc.abstractmethod
    def pdescs(self):
        pass

    def has_weights(self):
        return False

    def constants(self):
        """
        Values that parameter expressions can use (e.g. the radial nodes
        of a disk), by name.
        """
        return {}

    def line_names(self) -> tuple[str, ...]:
        """The names of the emission lines of the component (none here)."""
        return ()

    def circular_velocity_params(self) -> dict[str, tuple[float, ...]]:
        """
        The parameters of the component that are not in pdescs, because
        its model gives them: the circular velocity of the mass model of
        the model at the given radii (arcsec), by name (see
        traits.VPTraitMass). Its plans need them with the others. None
        here.
        """
        return {}

    @abc.abstractmethod
    def plan(self, driver, spectral, dtype, lines):
        """
        The evaluation of the component on the given driver and dtype (a
        ComponentPlan), which owns the memory it needs. spectral is the
        spectral axis of the outputs (a gridutils.Grid of one axis), and
        lines the names of the emission lines to evaluate (all if None).
        """
        pass


class ComponentPlan(abc.ABC):
    """The evaluation of a component on a driver and dtype."""

    @abc.abstractmethod
    def evaluate(self, params, grid, outputs, out_extra):
        """
        Add the component to the outputs.

        grid has the 3d spatial grid and the spectral axis: spat_size,
        spat_step, spat_zero (each in x, y, z order), spat_rota (degrees)
        and spec_size, spec_step, spec_zero. The z axis of 2d models and
        the spectral axis of image models have size 1 (and step 0).

        outputs has the (device) arrays the component adds to, which a
        model may leave out: the 'image' or 'scube', the 3d spatial
        weights ('wdata'), brightness ('bdata'), opacity ('odata') and
        brightness after the opacity ('obdata'). Opacity components add
        to odata, and the other components read it.

        out_extra is a dict for the extra outputs (on the host) of the
        component, or None.
        """
        pass


class BrightnessComponent2D(Component, abc.ABC):
    """A component of intensity_2d. Its outputs: image, wdata, bdata."""


class BrightnessComponent3D(Component, abc.ABC):
    """
    A component of intensity_3d. Its outputs: image, wdata, bdata, odata,
    obdata.
    """


class SpectralComponent2D(Component, abc.ABC):
    """A component of kinematics_2d. Its outputs: scube, wdata, bdata."""


class SpectralComponent3D(Component, abc.ABC):
    """
    A component of kinematics_3d. Its outputs: scube, wdata, bdata, odata,
    obdata.
    """


class OpacityComponent3D(Component, abc.ABC):
    """
    An opacity component of intensity_3d and kinematics_3d. Its output:
    odata, the optical depth of each voxel. Its polar traits give the
    optical depth of the disk seen face-on, and its height traits (pdfs)
    distribute it along z.
    """
