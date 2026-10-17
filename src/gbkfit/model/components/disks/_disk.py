
import abc
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from gbkfit.driver import DeviceArray, Driver
from gbkfit.math.interpolation import Interpolator
from gbkfit.params import (
    ParamDesc, ParamScalarDesc, ParamValues, ParamVectorDesc)
from .._nodal import (
    ExtraMaps, NodalPlan, NodalTraits, ParamBuffer, TraitParams,
    fill_param_values, native_traits, subring_radii)
from ..base import NativeGrid
from ..geometries import GEOMETRY_PARAMS
from ..traits import PressureSupport, Trait


# The kinds of traits of a disk: density (r), velocity (v) and dispersion
# (d) polar (p) and height (h) traits, vertical distortion (z) and
# selection (s) polar traits, and the polar traits of the coefficients of
# its line opacity (o)
TRAIT_KINDS = (
    'rpt', 'rht', 'vpt', 'vht', 'dpt', 'dht', 'zpt', 'spt', 'opt')


def _make_param_descs(key: str, nnodes: int, nw: bool) -> dict[str, ParamDesc]:
    """Return the parameter of the given name, node-wise if nw."""
    return {key: ParamVectorDesc(nnodes) if nw else ParamScalarDesc()}


class Disk(abc.ABC):

    def __init__(
            self,
            loose: bool,
            tilted: bool,
            rnodes: Sequence[float],
            rstep: float,
            interp: type[Interpolator],
            traits_: dict[str, Sequence[Trait]],
            prefixes: dict[str, str | Sequence[str]],
            opacity_lines: Sequence[int] = ()
    ):
        """
        traits_ has the traits of each kind, keyed as in TRAIT_KINDS, and
        can leave kinds out (no traits). prefixes has the prefix of the
        parameters of the traits of each kind in traits_ (e.g. 'pt' for
        the density polar traits), or of each of its traits (e.g.
        'op_halpha_'). opacity_lines has the line of each trait of line
        opacity ('opt'): its index among the lines of the emission.
        """

        subrnodes = subring_radii(rnodes, rstep)

        self._loose = loose
        self._tilted = tilted
        self._rnodes = rnodes
        self._rstep = rstep
        self._subrnodes = subrnodes
        self._interp = interp
        self._traits = NodalTraits(
            TRAIT_KINDS, traits_, prefixes, rnodes, subrnodes, interp)
        self._opacity_lines = tuple(opacity_lines)

        # Make descs for the geometric parameters. There is no systemic
        # velocity without velocity traits.
        switches = dict(loose=loose, tilted=tilted)
        self._geometry_isnw = {
            name: switches[switch] for name, switch in GEOMETRY_PARAMS.items()}
        self._geometry_pdescs = {
            name: _make_param_descs(
                name, len(rnodes), self._geometry_isnw[name])
            for name in GEOMETRY_PARAMS
            if name != 'vsys' or self.traits('vpt')}

        # Merge all parameter descs into the same dictionary, without those
        # that the model gives
        self._pdescs = {}
        for pdescs in self._geometry_pdescs.values():
            self._pdescs.update(pdescs)
        self._pdescs.update(self._traits.pdescs())

    def loose(self) -> bool:
        return self._loose

    def tilted(self) -> bool:
        return self._tilted

    def rnodes(self) -> tuple[float, ...]:
        return self._rnodes

    def bounding_radius(self) -> float:
        """
        Return the radius (arcsec) of the sphere around its centre that
        holds all its density, at any inclination: that of its last ring,
        R, or with density height traits, sqrt(R^2 + Z^2), where Z is the
        largest zmax of its density height traits (above its midplane) plus
        those of its vertical distortion traits (the largest displacement
        of its midplane; see RHTrait and ZPTrait).
        """
        height = max(
            (trait.zmax() for trait in self.traits('rht')), default=0.0)
        displacement = sum(trait.zmax() for trait in self.traits('zpt'))
        return float(np.hypot(self._rnodes[-1], height + displacement))

    def rstep(self) -> float:
        return self._rstep

    def subrnodes(self) -> tuple[float, ...]:
        return self._subrnodes

    def interp(self) -> type[Interpolator]:
        return self._interp

    def traits(self, kind: str) -> tuple[Trait, ...]:
        return self._traits.traits(kind)

    def opacity_lines(self) -> tuple[int, ...]:
        """
        Return the line of each trait of line opacity ('opt'): its index
        among the lines of the emission.
        """
        return self._opacity_lines

    def trait_params(self, kind: str) -> TraitParams:
        """Return the parameters of the traits of one kind (TraitParams)."""
        return self._traits.params(kind)

    def trait_values(
            self, kind: str, params: ParamValues
    ) -> tuple[dict[str, np.ndarray], ...]:
        """
        Return the values of the parameters of each trait of one kind, by
        their names in the trait, from the values of the parameters of the
        disk: the node-wise ones at the subnodes, as the disk evaluates
        them.
        """
        return self._traits.values(kind, params)

    def options(self) -> dict[str, Any]:
        """
        Return the options of this type of disk, besides those of all
        disks.
        """
        return {}

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._pdescs

    def circular_velocity_params(self) -> dict[str, tuple[float, ...]]:
        """
        Return the parameters that are not in pdescs, but whose values are
        the circular velocity of the mass of their galaxy at the
        given radii (arcsec), by name. The disk plans need them with the
        others.
        """
        return self._traits.circular_velocity_params()

    def pressure_supported_params(self) -> dict[str, PressureSupport]:
        """
        Return how those of its circular velocity parameters (see
        circular_velocity_params) whose values are corrected for its
        pressure are corrected, by their names (see
        traits.VPTraitMass).
        """
        return self._traits.pressure_supported_params()

    @abc.abstractmethod
    def plan(
            self,
            driver: Driver,
            nlines: int,
            dtype: np.dtype,
            opacity_rows: Sequence[int] | None = None
    ) -> 'DiskPlan':
        """
        Plan the evaluation of the disk on the given driver and dtype (a
        DiskPlan), which owns the memory it needs. nlines is the number of
        emission lines it adds to spectral cubes (0 without them), and
        opacity_rows has the row of the line of each trait of line opacity
        among them, or -1 for a line that is not evaluated (all -1 if
        None).
        """
        pass


class DiskPlan(NodalPlan):
    """
    The evaluation of a disk on a driver and dtype: the native description
    of the disk, and its parameter values (geometry and traits, and the
    values of its emission lines) in one buffer (see ParamBuffer).
    """

    def __init__(
            self,
            disk: Disk,
            driver: Driver,
            nlines: int,
            dtype: np.dtype,
            opacity_rows: Sequence[int] | None = None
    ):

        # The number of values of each group of parameters: one value for
        # each subnode for node-wise geometric parameters, and the values
        # of all the traits of each kind
        nsubrnodes = len(disk.subrnodes())
        sizes = {
            name: nsubrnodes if disk._geometry_isnw[name] else 1
            for name in disk._geometry_pdescs}
        sizes |= disk._traits.sizes()
        # The offset, scale and flux of each emission line
        sizes['lines'] = 3 * nlines
        # The sines and cosines of posa and incl, for the search of the
        # ring of each point of a loose or tilted disk
        sizes['posa_incl_sincos'] = 4 * sizes['posa']
        buffer = ParamBuffer(driver, dtype, sizes)

        def to_device(values, dtype_):
            return driver.mem_copy_h2d(np.array(values, dtype_))

        # Describe the disk to the native module: its (sub)nodes, the
        # device memory of its geometric parameters, its traits, and the
        # row of the line of each trait of line opacity
        if opacity_rows is None:
            opacity_rows = [-1] * len(disk.traits('opt'))
        self._native_disk = driver.native_class('Disk', dtype)(
            loose=disk.loose(),
            tilted=disk.tilted(),
            rnodes=to_device(disk.subrnodes(), dtype),
            **{name: buffer.device(name)
               if name in disk._geometry_pdescs else None
               for name in GEOMETRY_PARAMS},
            **native_traits(driver, dtype, disk._traits, buffer),
            opt_rows=to_device(opacity_rows, np.int32)
            if disk.traits('opt') else None,
            posa_incl_sincos=buffer.device('posa_incl_sincos'))

        self._disk = disk
        self._driver = driver
        self._dtype = dtype
        self._nlines = nlines
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
            image_scale: float = 1.0
    ) -> None:
        disk = self._disk
        params = self._write_params(params, lines)
        maps = None
        if out_extra is not None:
            maps = ExtraMaps(
                self._driver, self._dtype, grid,
                density=bool(disk.traits('rpt')),
                velocity=bool(disk.traits('vpt')),
                dispersion=bool(disk.traits('dpt')),
                opacity=outputs.get('opacity') is not None)

        # The keyword arguments of the native evaluation functions
        grid_and_outputs = grid | outputs \
            | (maps.native() if maps is not None else {}) \
            | dict(lines=self._lines_d, rdata_scale=rdata_scale,
                   image_scale=image_scale)

        self._impl_evaluate(params, grid_and_outputs, out_extra)

        if maps is not None:
            maps.collect(out_extra, density_key)

    def _fields_lines(self) -> int:
        """
        Return the number of fields of line opacity of the disk: one for
        each line it evaluates, if it has line opacity.
        """
        return self._nlines if self._disk.traits('opt') else 0

    def evaluate_polar_traits(
            self,
            params: ParamValues,
            kind: str,
            r: npt.ArrayLike,
            theta: npt.ArrayLike,
            incl: float | None = None
    ) -> np.ndarray:
        """
        Evaluate each polar trait of one kind at points of the plane of the
        disk, as the disk is evaluated: with the node-wise parameters at
        the subnodes, and NaN off the disk (radii below the first or above
        the last node; at the last node, the values that the disk tends to
        there).

        Parameters
        ----------
        params : ParamValues
            The parameters of the disk (see Disk.pdescs).
        kind : str
            The kind of the polar traits: 'rpt', 'vpt', 'dpt', 'zpt' or
            'spt' (see TRAIT_KINDS).
        r, theta : array_like
            The radius (arcsec) and azimuth (degrees) of each point in the
            plane of the disk (see traits), broadcast together.
        incl : float, optional
            The inclination (degrees) at which the velocity traits project
            their velocities on the line of sight: required for them
            ('vpt'), and only for them.

        Returns
        -------
        np.ndarray
            The values: one row for each trait, of the shape of the points.

        Raises
        ------
        ValueError
            If kind is not a kind of polar traits, the disk has no traits
            of the kind, or incl is missing for the velocity traits or
            given for others.
        """
        r, theta = np.broadcast_arrays(r, theta)
        return self._evaluate_traits(
            self._backend.disk_ptraits_evaluate, params, kind,
            dict(r=r, theta=theta), incl=incl)

    def evaluate_height_traits(
            self,
            params: ParamValues,
            kind: str,
            r: npt.ArrayLike,
            z: npt.ArrayLike
    ) -> np.ndarray:
        """
        Evaluate each height trait of one kind at points of the plane of
        the disk, as the disk is evaluated: with the node-wise parameters
        at the subnodes, and NaN off the disk (radii below the first or
        above the last node; at the last node, the values that the disk
        tends to there).

        Parameters
        ----------
        params : ParamValues
            The parameters of the disk (see Disk.pdescs).
        kind : str
            The kind of the height traits: 'rht', 'vht' or 'dht' (see
            TRAIT_KINDS).
        r, z : array_like
            The radius (arcsec) of each point, and its height (arcsec)
            above the midplane (see traits), broadcast together.

        Returns
        -------
        np.ndarray
            The values: one row for each trait, of the shape of the points.

        Raises
        ------
        ValueError
            If kind is not a kind of height traits, or the disk has no
            traits of the kind.
        """
        r, z = np.broadcast_arrays(r, z)
        return self._evaluate_traits(
            self._backend.disk_htraits_evaluate, params, kind,
            dict(r=r, z=z))

    def _write_params(
            self, params: ParamValues, lines: np.ndarray | None = None
    ) -> ParamValues:
        """
        Write the parameter values into the host memory, with the node-wise
        parameters at the subnodes (interpolated if given at the nodes),
        and the values of the lines if given, and copy them to the device
        at once. Return the parameters with the node-wise values at the
        subnodes (a new dict: the caller's arrays are not modified).
        """
        disk = self._disk
        buffer = self._buffer
        params = dict(params)
        for name, pdescs in disk._geometry_pdescs.items():
            sampling = 'rnodes' if disk._geometry_isnw[name] else None
            fill_param_values(
                buffer.host(name), params, pdescs,
                dict.fromkeys(pdescs, sampling), disk.rnodes(),
                disk.subrnodes(), disk.interp())
        disk._traits.fill(buffer, params)
        if lines is not None:
            self._lines_h[:] = lines
        posa, incl = (
            np.deg2rad(buffer.host(name).astype(np.float64))
            for name in ('posa', 'incl'))
        buffer.host('posa_incl_sincos').reshape(-1, 4)[...] = \
            np.column_stack(
                [np.sin(posa), np.cos(posa), np.sin(incl), np.cos(incl)])
        buffer.copy_to_device()
        return params

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
        (disk_ptraits_evaluate or disk_htraits_evaluate) at points: the
        arrays of their coordinates by name (r, and theta or z), of one
        shape. options are the other arguments of the function. The native
        function checks the arguments (e.g. that the disk has traits of
        the kind: with none, the output has no rows).
        """
        driver = self._driver
        dtype = self._dtype
        shape = points['r'].shape
        # (an unknown kind has no rows: the native function says so)
        ntraits = len(self._disk.traits(kind)) if kind in TRAIT_KINDS else 0

        def to_device(values: np.ndarray) -> DeviceArray:
            return driver.mem_copy_h2d(
                np.ascontiguousarray(values.ravel(), dtype))

        self._write_params(params)
        out = driver.mem_alloc_d((ntraits, points['r'].size), dtype)
        function(
            self._native_disk, kind, out=out,
            **{name: to_device(values) for name, values in points.items()},
            **options)
        return np.asarray(driver.mem_copy_d2h(out)).reshape(
            (ntraits,) + shape)

    @abc.abstractmethod
    def _impl_evaluate(
            self,
            params: ParamValues,
            grid_and_outputs: dict[str, Any],
            out_extra: dict[str, Any] | None
    ) -> None:
        """
        Evaluate the disk. params has the values of the node-wise
        parameters at the subnodes, and grid_and_outputs the keyword
        arguments of the grid and the outputs of the native evaluation
        functions.
        """
        pass
