
import abc

from gbkfit.utils import parseutils


__all__ = [
    'GModel',
    'GModelImage',
    'GModelPlan',
    'GModelSCube',
    'gmodel_parser'
]


class GModel(parseutils.TypedSerializable, abc.ABC):
    """
    A model of a galaxy: what is on the sky. It can have a name, which
    then prefixes its parameters instead of its position when there are
    several gmodels (see ObservationGroup).
    """

    def __init__(self, name: str | None):
        parseutils.check_name(name)
        self._name = name

    def name(self) -> str | None:
        return self._name

    @abc.abstractmethod
    def pdescs(self):
        pass

    @abc.abstractmethod
    def has_weights(self):
        pass

    def constants(self):
        """
        Values that parameter expressions can use (e.g. the radial nodes
        of a disk), by name.
        """
        return {}

    @abc.abstractmethod
    def plan(self, driver, grid, has_weights, dtype, components=None):
        """
        The evaluation of the gmodel on the given driver, grid of its data
        (fitsutils.Grid: x and y, and the spectral axis of spectral cubes)
        and dtype, with spatial weights if has_weights (a GModelPlan), of
        its components of the given names (all if None; opacity components
        always absorb).
        """
        pass


class GModelPlan(abc.ABC):
    """The evaluation of a gmodel on a driver, grid and dtype."""

    @abc.abstractmethod
    def evaluate(self, params, data, weights, out_extra):
        """
        Add the gmodel to data (the image or spectral cube on the grid of
        the plan), and weight the data weights (or None) with its spatial
        weights.
        """
        pass


class GModelImage(GModel, abc.ABC):
    """A gmodel evaluated into images."""


class GModelSCube(GModel, abc.ABC):
    """A gmodel evaluated into spectral cubes."""


gmodel_parser = parseutils.TypedParser(GModel)
