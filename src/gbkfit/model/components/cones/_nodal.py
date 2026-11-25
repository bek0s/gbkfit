"""
The cones: shells around an apex, with traits (NodalCone), evaluated
smoothly (SmoothCone).
"""

import abc
from collections.abc import Callable, Sequence
from typing import Any

import astropy.units
import numpy as np
import numpy.typing as npt

from gbkfit.driver import DeviceArray, Driver
from gbkfit.math.interpolation import Interpolator
from gbkfit.params import ParamDesc, ParamScalarDesc, ParamValues
from .._native import NativePlan, ParamBuffer, evaluate_native
from .._nodal import NodalComponent, evaluate_native_traits, native_traits
from ..base import (
    ComponentState, FieldsLayout, NativeGrid)
from ...geometries import GEOMETRY_UNITS, geometry_pdesc
from ..traits import Trait


__all__ = [
    'SIDES',
    'TRAIT_KINDS',
    'NodalCone',
    'SmoothCone',
    'SmoothConePlan'
]


# The kinds of traits of a cone: density (r), velocity (v) and dispersion
# (d) polar (p) and height (h) traits, and selection (s) polar traits
TRAIT_KINDS = ('rpt', 'rht', 'vpt', 'vht', 'dpt', 'dht', 'spt')

# The sides that a cone can have: both, or the positive one (along the
# positive side of its axis) or the negative one only
SIDES = ('both', 'positive', 'negative')

# The geometric parameters of a cone: its systemic velocity (only with
# velocity traits), its apex, its axis and its wall
GEOMETRY_PDESCS = {
    name: geometry_pdesc(name) for name in GEOMETRY_UNITS
} | dict(
    wall=ParamScalarDesc(
        title='Wall angle',
        desc="the angle of the mid-surface of the walls from the axis",
        min=0, max=90, unit='deg', default=30))


class NodalCone(NodalComponent):
    """
    A cone: its sides, its radial nodes (shells, spheres around its apex)
    and its traits (see traits), whose parameters are those of its traits
    and of its geometry: its apex (xpos, ypos), its axis, the spin axis of
    a disk of position angle posa and inclination incl, the angle of the
    mid-surface of its walls from the axis (wall), and with velocity
    traits, its systemic velocity (vsys).
    """

    def __init__(
            self,
            sides: str,
            rnodes: Sequence[float],
            rstep: float,
            interp: type[Interpolator],
            traits_: dict[str, Sequence[Trait]],
            prefixes: dict[str, str],
            density_units: dict[str, astropy.units.UnitBase | None]
    ):
        """
        sides is one of SIDES; the others are as those of NodalComponent.
        """
        super().__init__(
            TRAIT_KINDS, rnodes, rstep, interp, traits_, prefixes,
            density_units)
        self._sides = sides
        self._geometry_pdescs = {
            name: pdesc for name, pdesc in GEOMETRY_PDESCS.items()
            if name != 'vsys' or self.traits('vpt')}
        self._pdescs = self._geometry_pdescs | self._traits.pdescs()

    def sides(self) -> str:
        """Return its sides (see SIDES)."""
        return self._sides

    def geometry_pdescs(self) -> dict[str, ParamDesc]:
        return self._geometry_pdescs

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._pdescs

    @abc.abstractmethod
    def plan(self, driver: Driver, nlines: int, dtype: np.dtype) -> NativePlan:
        """
        Plan the evaluation of the cone on the given driver and dtype,
        which owns the memory it needs. nlines is the number of light
        lines it adds to spectral cubes (0 without them).
        """
        pass


class SmoothCone(NodalCone):
    """A cone evaluated smoothly: on the voxels of its grid."""

    def plan(
            self, driver: Driver, nlines: int, dtype: np.dtype
    ) -> 'SmoothConePlan':
        return SmoothConePlan(self, driver, nlines, dtype)


class SmoothConePlan(NativePlan):
    """
    The evaluation of a SmoothCone on a driver and dtype: the native
    description of the cone, and its parameter values (geometry and traits,
    and the values of its emission lines) in one buffer (see ParamBuffer).
    """

    def __init__(
            self, cone: SmoothCone, driver: Driver, nlines: int,
            dtype: np.dtype
    ):
        # The number of values of each group of parameters: one for each
        # geometric parameter, and the values of all the traits of each
        # kind, and the offset, scale and flux of each emission line
        sizes = dict.fromkeys(cone.geometry_pdescs(), 1) \
            | cone.nodal_traits().sizes() | dict(lines=3 * nlines)
        buffer = ParamBuffer(driver, dtype, sizes)

        # Describe the cone to the native module: its sides, its
        # (sub)nodes, the device memory of its geometric parameters and its
        # traits
        self._native_cone = driver.native_class('Cone', dtype)(
            positive=cone.sides() != 'negative',
            negative=cone.sides() != 'positive',
            rnodes=driver.mem_copy_h2d(np.array(cone.subrnodes(), dtype)),
            **{name: buffer.device(name) for name in cone.geometry_pdescs()},
            **native_traits(driver, dtype, cone.nodal_traits(), buffer))

        self._cone = cone
        self._driver = driver
        self._dtype = dtype
        self._buffer = buffer
        self._lines_h = buffer.host('lines').reshape(nlines, 3)
        self._lines_d = buffer.device('lines').reshape(nlines, 3) \
            if nlines else None
        self._backend = driver.native_class('GModel', dtype)()

    def evaluate(
            self,
            params: ParamValues,
            grid: NativeGrid,
            outputs: dict[str, DeviceArray | None],
            out_extra: dict[str, Any] | None,
            density_key: str,
            lines: np.ndarray | None = None,
            rdata_scale: float = 1.0,
            image_scale: float = 1.0,
            state: ComponentState | None = None
    ) -> None:
        self._write_params(params, lines)
        evaluate_native(
            lambda kwargs: self._backend.smcone_evaluate(
                self._native_cone, **kwargs),
            self._driver, self._dtype, grid, outputs, out_extra,
            density_key, self._lines_d, rdata_scale, image_scale,
            velocity=bool(self._cone.traits('vpt')),
            dispersion=bool(self._cone.traits('dpt')))

    def fields_layout(self) -> FieldsLayout:
        # (a cone fills the voxels with j, v and s)
        return FieldsLayout(layer=False, nslots=3)

    def evaluate_polar_traits(
            self,
            params: ParamValues,
            kind: str,
            r: npt.ArrayLike,
            theta: npt.ArrayLike
    ) -> np.ndarray:
        """
        Evaluate each polar trait of one kind ('rpt', 'vpt', 'dpt' or
        'spt') at points of the mid-surface of the walls of the cone, as
        the cone is evaluated: r the distance (arcsec) from its apex, and
        theta the azimuth (degrees) around its axis; the velocity traits
        give their speeds, each along its direction. A row for each trait,
        of the shape of the points (see profiles.TraitsPlan).
        """
        r, theta = np.broadcast_arrays(r, theta)
        return self._evaluate_traits(
            self._backend.cone_ptraits_evaluate, params, kind,
            dict(r=r, theta=theta), wall=float(params['wall']))

    def evaluate_height_traits(
            self,
            params: ParamValues,
            kind: str,
            r: npt.ArrayLike,
            z: npt.ArrayLike
    ) -> np.ndarray:
        """
        Evaluate each height trait of one kind ('rht', 'vht' or 'dht') at
        points of the cone, as the cone is evaluated: r the distance
        (arcsec) from its apex, and z the distance (arcsec) from the
        mid-surface of its walls. A row for each trait, of the shape of the
        points.
        """
        r, z = np.broadcast_arrays(r, z)
        return self._evaluate_traits(
            self._backend.cone_htraits_evaluate, params, kind,
            dict(r=r, z=z))

    def _evaluate_traits(
            self,
            function: Callable[..., None],
            params: ParamValues,
            kind: str,
            points: dict[str, np.ndarray],
            **options: Any
    ) -> np.ndarray:
        """
        Evaluate the traits of one kind with a native function of traits
        at points (see _nodal.evaluate_native_traits), of the values of
        the parameters of the cone.
        """
        # (an unknown kind has no rows: the native function says so)
        ntraits = len(self._cone.traits(kind)) if kind in TRAIT_KINDS else 0
        self._write_params(params, None)
        return evaluate_native_traits(
            self._driver, self._dtype, function, self._native_cone, ntraits,
            points, kind=kind, **options)

    def _write_params(
            self, params: ParamValues, lines: np.ndarray | None
    ) -> None:
        """
        Write the parameter values into the host memory, with the node-wise
        parameters at the subnodes (interpolated if given at the nodes),
        and the values of the lines if given, and copy them to the device
        at once.
        """
        buffer = self._buffer
        params = dict(params)
        for name in self._cone.geometry_pdescs():
            buffer.host(name)[:] = params[name]
        self._cone.nodal_traits().fill(buffer, params)
        if lines is not None:
            self._lines_h[:] = lines
        buffer.copy_to_device()
