
import abc
import typing

from gbkfit.utils import parseutils


__all__ = [
    'Model',
    'ModelImage',
    'ModelPlan',
    'ModelSCube',
    'Selection',
    'model_parser'
]


class Selection(typing.NamedTuple):
    """
    What of a model an observation sees: its components and the emission
    lines of its components of the given names (all if None). Opacity
    components always absorb; the components without lines have one,
    which is always seen.
    """
    components: tuple[str, ...] | None = None
    lines: tuple[str, ...] | None = None


class Model(parseutils.TypedSerializable, abc.ABC):
    """
    A model of a galaxy: what is on the sky. It can have a name, which
    then prefixes its parameters instead of its position when there are
    several models (see ObservationGroup).
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
    def plan(self, driver, grid, has_weights, dtype, selection=Selection()):
        """
        The evaluation of the model on the given driver, grid of its data
        (gridutils.Grid: x and y, and the spectral axis of spectral cubes)
        and dtype, with spatial weights if has_weights (a ModelPlan), of
        what the selection has (see Selection).
        """
        pass


class ModelPlan(abc.ABC):
    """The evaluation of a model on a driver, grid and dtype."""

    @abc.abstractmethod
    def evaluate(self, params, data, weights, out_extra):
        """
        Add the model to data (the image or spectral cube on the grid of
        the plan), and weight the data weights (or None) with its spatial
        weights.
        """
        pass


class ModelImage(Model, abc.ABC):
    """A model evaluated into images."""


class ModelSCube(Model, abc.ABC):
    """A model evaluated into spectral cubes."""


model_parser = parseutils.TypedParser(Model)
