"""
The cones: shells around an apex, with traits (NodalCone), evaluated
smoothly (SmoothCone).
"""

import abc
from collections.abc import Sequence
from typing import Any

import astropy.units
import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.math.interpolation import Interpolator
from gbkfit.params import ParamDesc, ParamScalarDesc, ParamValues
from .._native import NativePlan, ParamBuffer, evaluate_native
from .._nodal import NodalTraits, native_traits, subring_radii
from ..base import (
    ComponentState, FieldsLayout, FrameParams, NativeGrid)
from ..geometries import GEOMETRY_UNITS
from ..traits import Trait


__all__ = [
    'NodalCone',
    'SIDES',
    'SmoothCone',
    'SmoothConePlan',
    'TRAIT_KINDS'
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
    name: ParamScalarDesc(unit=unit) for name, unit in GEOMETRY_UNITS.items()
} | dict(
    wall=ParamScalarDesc(
        desc="the angle of the mid-surface of the walls from the axis",
        min=0, max=90, unit='deg'))


class NodalCone(abc.ABC):
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
        traits_ has the traits of each kind, keyed as in TRAIT_KINDS, and
        can leave kinds out (no traits); prefixes has the prefix of the
        parameters of the traits of each kind in traits_ (e.g. 'pt' for the
        density polar traits); density_units has the unit of the density
        of the density polar traits, if known (see NodalTraits).
        """
        subrnodes = subring_radii(rnodes, rstep)
        self._sides = sides
        self._rnodes = rnodes
        self._rstep = rstep
        self._subrnodes = subrnodes
        self._interp = interp
        self._traits = NodalTraits(
            TRAIT_KINDS, traits_, prefixes, density_units, rnodes,
            subrnodes, interp)
        self._geometry_pdescs = {
            name: pdesc for name, pdesc in GEOMETRY_PDESCS.items()
            if name != 'vsys' or self.traits('vpt')}
        self._pdescs = self._geometry_pdescs | self._traits.pdescs()

    def sides(self) -> str:
        """Return its sides (see SIDES)."""
        return self._sides

    def rnodes(self) -> tuple[float, ...]:
        return self._rnodes

    def rstep(self) -> float:
        return self._rstep

    def subrnodes(self) -> tuple[float, ...]:
        return self._subrnodes

    def interp(self) -> type[Interpolator]:
        return self._interp

    def traits(self, kind: str) -> tuple[Trait, ...]:
        return self._traits.traits(kind)

    def frame_params(self) -> FrameParams:
        """Return its parameters that place it in a frame."""
        vsys = 'vsys' if 'vsys' in self._geometry_pdescs else None
        return FrameParams('xpos', 'ypos', vsys, 'posa', 'incl')

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._pdescs

    def bounding_radius(self) -> float:
        """
        Return the radius (arcsec) of the sphere around its apex that holds
        all its density: that of its last shell.
        """
        return float(self._rnodes[-1])

    @abc.abstractmethod
    def plan(self, driver: Driver, nlines: int, dtype: np.dtype) -> NativePlan:
        """
        Plan the evaluation of the cone on the given driver and dtype,
        which owns the memory it needs. nlines is the number of emission
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
        sizes = dict.fromkeys(cone._geometry_pdescs, 1) \
            | cone._traits.sizes() | dict(lines=3 * nlines)
        buffer = ParamBuffer(driver, dtype, sizes)

        # Describe the cone to the native module: its sides, its
        # (sub)nodes, the device memory of its geometric parameters and its
        # traits
        self._native_cone = driver.native_class('Cone', dtype)(
            positive=cone.sides() != 'negative',
            negative=cone.sides() != 'positive',
            rnodes=driver.mem_copy_h2d(np.array(cone.subrnodes(), dtype)),
            **{name: buffer.device(name) for name in cone._geometry_pdescs},
            **native_traits(driver, dtype, cone._traits, buffer))

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
        for name in self._cone._geometry_pdescs:
            buffer.host(name)[:] = params[name]
        self._cone._traits.fill(buffer, params)
        if lines is not None:
            self._lines_h[:] = lines
        buffer.copy_to_device()
