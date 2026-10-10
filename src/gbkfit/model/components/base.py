import abc
from typing import Any, TypedDict

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamDesc
from gbkfit.utils import gridutils, parseutils
from .geometries import Geometry


__all__ = [
    'Component',
    'ComponentPlan',
    'NativeGrid',
    'BrightnessComponent2D',
    'BrightnessComponent3D',
    'SpectralComponent2D',
    'SpectralComponent3D',
    'OpacityComponent3D'
]


class NativeGrid(TypedDict):
    """
    The grid that component plans evaluate on, as the native evaluation
    functions take it: the 3d spatial grid (each in x, y, z order; the
    rotation in degrees) and the spectral axis. The z axis of 2d models
    and the spectral axis of image models have size 1 (and step 0).
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
    A component of a model. The kinds of components below differ only in
    the models that accept them, and in the outputs they get.

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters in its model instead of
        its position (see parseutils.item_prefixes).
    geometry : Geometry, optional
        A geometry that it shares with other components.

    Raises
    ------
    ConfigError
        If the name is not valid (see parseutils.check_name).
    """

    def __init__(self, name: str | None, geometry: Geometry | None = None):
        parseutils.check_name(name)
        self._name = name
        self._geometry = geometry

    def name(self) -> str | None:
        """Return its name, if any."""
        return self._name

    def geometry(self) -> Geometry | None:
        """Return the geometry it shares, if any."""
        return self._geometry

    @abc.abstractmethod
    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name."""
        pass

    def has_weights(self) -> bool:
        """Check whether it has spatial weights; not here."""
        return False

    def constants(self) -> dict[str, Any]:
        """
        Return the values that parameter expressions can use (e.g. the
        radial nodes of a disk), by name; none here.
        """
        return {}

    def line_names(self) -> tuple[str, ...]:
        """Return the names of its emission lines; none here."""
        return ()

    def circular_velocity_params(self) -> dict[str, tuple[float, ...]]:
        """
        Return the parameters that its model gives it: the circular
        velocity of the mass model of its model at the given radii
        (arcsec), by name (see traits.VPTraitMass). They are not in
        pdescs, but its plans need them with the others. None here.
        """
        return {}

    @abc.abstractmethod
    def plan(
            self,
            driver: Driver,
            spectral: gridutils.Grid,
            dtype: np.dtype,
            lines: tuple[str, ...] | None
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

        Returns
        -------
        ComponentPlan
            The plan.
        """
        pass


class ComponentPlan(abc.ABC):
    """The evaluation of a component on a driver and dtype."""

    @abc.abstractmethod
    def evaluate(
            self,
            params: dict[str, float | np.ndarray],
            grid: NativeGrid,
            outputs: dict[str, DeviceArray | None],
            out_extra: dict[str, Any] | None
    ) -> None:
        """
        Add the component to the outputs.

        Parameters
        ----------
        params : dict
            The values of its parameters, by name.
        grid : NativeGrid
            The grid of the evaluation.
        outputs : dict
            The arrays the component adds to, which a model may leave out
            (None): the 'image' or 'scube', the 3d spatial weights
            ('wdata'), brightness ('bdata'), opacity ('odata') and
            brightness after the opacity ('obdata'). Opacity components
            add to odata, and the other components read it.
        out_extra : dict, optional
            Where its extra outputs (on the host) go, if wanted.
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
