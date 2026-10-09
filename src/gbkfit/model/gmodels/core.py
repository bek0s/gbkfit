
import abc

from gbkfit.utils import parseutils


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
    A component of a gmodel. The kinds of components below differ only in
    the gmodels that accept them, and in the outputs they get. A component
    can have a name, which then prefixes its parameters in its gmodel
    instead of its position (see parseutils.item_prefixes).
    """

    def __init__(self, name: str | None):
        parseutils.check_name(name)
        self._name = name

    def name(self) -> str | None:
        return self._name

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

    @abc.abstractmethod
    def plan(self, driver, dtype):
        """
        The evaluation of the component on the given driver and dtype (a
        ComponentPlan), which owns the memory it needs.
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
        and spec_size, spec_step, spec_zero. The z axis of 2d gmodels and
        the spectral axis of image gmodels have size 1 (and step 0).

        outputs has the (device) arrays the component adds to, which a
        gmodel may leave out: the 'image' or 'scube', the 3d spatial
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
