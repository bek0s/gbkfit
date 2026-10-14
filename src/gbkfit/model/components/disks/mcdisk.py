from collections.abc import Sequence
from typing import Any, Self

import numpy as np

from gbkfit.driver import Driver
from gbkfit.math.interpolation import Interpolator
from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError
from .._units import DataUnit
from ..base import Component, ComponentPlan, FieldsLayout
from ..geometries import Geometry
from ..lines import Band, LineCatalogue
from ..roles import Emission, Opacity
from . import _detail, _disk, traits
from .parts import Density, Kinematics


__all__ = [
    'MCDisk'
]


# The most clouds the kernels draw: they count them in int32
_MAX_CLOUDS = np.iinfo(np.int32).max


class CloudDisk(_disk.Disk):
    """
    A disk made of clouds drawn at random from its traits, each of flux
    cflux, with the random numbers of the given seed (see Disk for the
    other options). Raise ConfigError if the seed is negative or cflux is
    not positive.
    """

    def __init__(
            self,
            cflux: float,
            seed: int,
            loose: bool,
            tilted: bool,
            rnodes: Sequence[float],
            rstep: float,
            interp: type[Interpolator],
            traits_: dict[str, Sequence[traits.Trait]],
            prefixes: dict[str, str | Sequence[str]],
            opacity_lines: Sequence[int] = ()
    ):
        super().__init__(
            loose, tilted, rnodes, rstep, interp, traits_, prefixes,
            opacity_lines)
        if seed < 0:
            raise ConfigError(f"seed must be >= 0; supplied value: {seed}")
        if not cflux > 0:
            raise ConfigError(f"cflux must be greater than 0; it is {cflux}")
        self._cflux = cflux
        self._seed = seed

    def cflux(self) -> float:
        """Return the flux of each cloud."""
        return self._cflux

    def seed(self) -> int:
        """Return the seed of the random numbers of the clouds."""
        return self._seed

    def options(self) -> dict[str, Any]:
        return dict(cflux=self._cflux, seed=self._seed)

    def plan(
            self,
            driver: Driver,
            nlines: int,
            dtype: np.dtype,
            opacity_rows: Sequence[int] | None = None
    ) -> _disk.DiskPlan:
        return CloudDiskPlan(self, driver, nlines, dtype, opacity_rows)


class CloudDiskPlan(_disk.DiskPlan):
    """The evaluation of a CloudDisk."""

    def __init__(
            self,
            disk: CloudDisk,
            driver: Driver,
            nlines: int,
            dtype: np.dtype,
            opacity_rows: Sequence[int] | None = None
    ):
        super().__init__(disk, driver, nlines, dtype, opacity_rows)
        # The clouds are made in pools: one for each density trait with
        # an analytical integral, and one for each ring of the others
        # (the rings are centred on the subnodes between the first and
        # the last, which are the edges of the disk). For each pool: the
        # cumulative number of clouds, and the (signed) flux of each of
        # its clouds. Also, for each density trait, whether it has an
        # analytical integral.
        rptraits = disk.traits('rpt')
        analytical = [t.has_analytical_integral() for t in rptraits]
        self._s_has_analytical_integral = driver.mem_alloc_s(
            len(rptraits), bool)
        host, device = self._s_has_analytical_integral
        host[:] = analytical
        driver.mem_copy_h2d(host, device)
        nrings = len(disk.subrnodes()) - 2
        npools = sum([1 if h else nrings for h in analytical])
        self._s_ncloudscsum = driver.mem_alloc_s(npools, np.int32)
        self._s_cloud_flux = driver.mem_alloc_s(npools, dtype)

    def fields_layout(self) -> FieldsLayout:
        # (j, v, s, its line opacity and the scratch of the moments of the
        # clouds)
        nk = self._fields_lines()
        return FieldsLayout(layer=False, nslots=4 + nk, line_opacity=nk > 0)

    def _impl_evaluate(
            self,
            params: ParamValues,
            grid_and_outputs: dict[str, Any],
            out_extra: dict[str, Any] | None
    ) -> None:

        disk = self._disk
        driver = self._driver

        # The flux of each pool
        pool_flux = []
        rpt_params = disk.trait_params('rpt')
        ring_centers = np.array(disk.subrnodes()[1:-1], self._dtype)
        for trait, pnames in zip(disk.traits('rpt'), rpt_params.pnames):
            # Make a parameter dict for the current trait
            # Use the original names and not the new/prefixed ones
            trait_params = {}
            for old_name, new_name in pnames.items():
                if rpt_params.sampling[new_name] is not None:
                    trait_params[old_name] = params[new_name][1:-1]
                else:
                    trait_params[old_name] = params[new_name]
            # The flux of the trait (one value if it has an analytical
            # integral), or of each of its rings
            pool_flux.extend(np.atleast_1d(
                trait.cloud_flux(trait_params, ring_centers)))
        pool_flux = np.asarray(pool_flux, np.float64)

        # Each pool has as many clouds as its flux needs at cflux each
        # (and at least one), which share its flux exactly: a negative
        # flux gives negative clouds
        nclouds = np.where(
            pool_flux != 0,
            np.maximum(np.rint(np.abs(pool_flux) / disk.cflux()), 1),
            0)
        if not np.sum(nclouds) <= _MAX_CLOUDS:
            raise RuntimeError(
                f"the Monte Carlo disk needs {np.sum(nclouds):.3g} clouds "
                f"of flux cflux = {disk.cflux()}, but the kernels draw at "
                f"most {_MAX_CLOUDS}; give it a larger cflux")
        nclouds = nclouds.astype(np.int32)
        cloud_flux = np.divide(
            pool_flux, nclouds, out=np.zeros_like(pool_flux),
            where=nclouds > 0)

        self._s_ncloudscsum[0][:] = np.cumsum(nclouds)
        self._s_cloud_flux[0][:] = cloud_flux
        driver.mem_copy_h2d(self._s_ncloudscsum[0], self._s_ncloudscsum[1])
        driver.mem_copy_h2d(self._s_cloud_flux[0], self._s_cloud_flux[1])

        self._backend.mcdisk_evaluate(
            self._native_disk,
            cloud_flux=self._s_cloud_flux[1],
            seed=disk.seed(),
            nclouds=int(nclouds.sum()),
            ncloudscsum=self._s_ncloudscsum[1],
            has_analytical_integral=self._s_has_analytical_integral[1],
            **grid_and_outputs)


class MCDisk(Component):
    """
    A disk made of clouds drawn at random from its density, each of flux
    cflux, with the random numbers of the given seed: a sample of the
    smooth disk of the same parts, to which it converges. It is thick
    (density height traits), and evaluated on a grid with a depth axis.

    Parameters
    ----------
    density : Density
        Its density (with height traits).
    cflux : float
        The flux (density) of each cloud.
    seed : int, optional
        The seed of the random numbers of its clouds.
    other parameters
        As SMDisk.

    Raises
    ------
    ConfigError
        As SMDisk, if it is thin, if the seed is negative, or if cflux is
        not positive.
    """

    @staticmethod
    def type() -> str:
        return 'mcdisk'

    @classmethod
    def load(
            cls,
            info: dict[str, Any],
            catalogue: LineCatalogue | None = None
    ) -> Self:
        return cls(**_detail.load_disk_component(cls, info, catalogue))

    def dump(self) -> dict[str, Any]:
        return _detail.dump_disk_component(
            self, self._disk, self._density, self._kinematics)

    def __init__(
            self,
            density: Density,
            cflux: float,
            kinematics: Kinematics | None = None,
            emission: Emission | None = None,
            opacity: Opacity | None = None,
            zptraits: traits.ZPTrait | Sequence[traits.ZPTrait] | None = None,
            sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None = None,
            loose: bool | None = None,
            tilted: bool | None = None,
            rnmin: float | None = None,
            rnmax: float | None = None,
            rnsep: float | None = None,
            rnlen: int | None = None,
            rnodes: Sequence[float] | None = None,
            rstep: float | None = None,
            interp: str = 'linear',
            seed: int = 0,
            geometry: Geometry | None = None,
            name: str | None = None
    ):
        super().__init__(name, geometry, emission, opacity)
        if not density.thick():
            raise ConfigError(
                "a Monte Carlo disk needs density height traits (it is "
                "thick)")
        self._density = density
        self._kinematics = kinematics
        self._disk = _detail.make_disk(
            CloudDisk, density, kinematics, emission, opacity, zptraits,
            sptraits, geometry=geometry, loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            cflux=cflux, seed=seed)
        self._lines = _detail.make_lines(self._disk, emission)
        self._bands = _detail.make_bands(self._disk, emission)

    def density(self) -> Density:
        """Return its density."""
        return self._density

    def kinematics(self) -> Kinematics | None:
        """Return its kinematics, if any."""
        return self._kinematics

    def thick(self) -> bool:
        return True

    def pdescs(self) -> dict[str, ParamDesc]:
        lines = self._lines.pdescs() if self._lines is not None else {}
        bands = self._bands.pdescs() if self._bands is not None else {}
        return self._disk.pdescs() | lines | bands

    def constants(self) -> dict[str, Any]:
        return dict(
            rnodes=self._disk.rnodes(), subrnodes=self._disk.subrnodes())

    def circular_velocity_params(self) -> dict[str, tuple[float, ...]]:
        return self._disk.circular_velocity_params()

    def density_disk(self) -> _disk.Disk:
        return self._disk

    def plan(
            self,
            driver: Driver,
            spectral: gridutils.Grid,
            dtype: np.dtype,
            lines: tuple[str, ...] | None,
            band: Band | None,
            emits: bool,
            opacity_ratios: np.ndarray | None,
            data_unit: DataUnit | None
    ) -> ComponentPlan:
        emission = self.emission()
        return _detail.disk_plan(
            self._disk, self._lines, self._bands,
            emission.unit() if emission is not None else None, driver,
            spectral, dtype, lines, band, emits, opacity_ratios, data_unit)
