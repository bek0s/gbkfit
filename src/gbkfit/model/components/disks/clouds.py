"""
The disks made of clouds (see Clouds).
"""

from collections.abc import Sequence
from typing import Any, Self

import astropy.units
import numpy as np

from gbkfit.driver import Driver
from gbkfit.math.interpolation import Interpolator
from gbkfit.params import ModelEvaluationError, ParamValues
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from .. import traits
from ..base import FieldsLayout
from . import _disk
from ..parts import Equilibrium


__all__ = [
    'CloudDisk',
    'CloudDiskPlan',
    'Clouds'
]


# The most clouds the kernels draw: they count them in int32
_MAX_CLOUDS = np.iinfo(np.int32).max


class Clouds(parseutils.Serializable):
    """
    The clouds of a disk made of clouds (see Disk): drawn at random from
    its density, each of the flux cflux, with the random numbers of the
    given seed: a sample of the smooth disk of the same parts. The clouds
    of the density polar traits that the disk draws as a whole (see
    RPTrait.has_analytical_integral; e.g. the uniform one and the mixtures
    of blobs) follow them exactly; those of the others are drawn ring by
    ring (subrings of width rstep): each ring has the flux of the trait
    over it (its integral, see RPTrait.cloud_flux), and its clouds are at
    radii drawn from the trait across it, and at azimuths drawn from the
    trait at their radii. A disk of clouds converges to the smooth disk
    as its clouds get more (smaller cflux).

    Parameters
    ----------
    cflux : float
        The flux (density) of each cloud.
    seed : int, optional
        The seed of the random numbers of the clouds.

    Raises
    ------
    ConfigError
        If cflux is not positive, or the seed is negative.
    """

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        return cls(**parseutils.parse_options_for_callable(info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        return dict(cflux=self._cflux, seed=self._seed)

    def __init__(self, cflux: float, seed: int = 0):
        if not cflux > 0:
            raise ConfigError(f"cflux must be greater than 0; it is {cflux}")
        if seed < 0:
            raise ConfigError(f"seed must be >= 0; it is {seed}")
        self._cflux = float(cflux)
        self._seed = seed

    def cflux(self) -> float:
        """Return the flux of each cloud."""
        return self._cflux

    def seed(self) -> int:
        """Return the seed of the random numbers of the clouds."""
        return self._seed


class CloudDisk(_disk.NodalDisk):
    """
    A disk made of clouds (see Clouds; see NodalDisk for the other
    options).
    """

    def __init__(
            self,
            clouds: Clouds,
            loose: bool,
            tilted: bool,
            rnodes: Sequence[float],
            rstep: float,
            interp: type[Interpolator],
            traits_: dict[str, Sequence[traits.Trait]],
            prefixes: dict[str, str | Sequence[str]],
            density_units: dict[str, astropy.units.UnitBase | None],
            opacity_lines: Sequence[int] = (),
            equilibrium: Equilibrium | None = None
    ):
        super().__init__(
            loose, tilted, rnodes, rstep, interp, traits_, prefixes,
            density_units, opacity_lines, equilibrium)
        self._clouds = clouds

    def clouds(self) -> Clouds:
        """Return its clouds."""
        return self._clouds

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

        # The flux of each pool: of each trait (one value if it has an
        # analytical integral), or of each of its rings, of its parameters
        # by their names in it (the node-wise ones at the subnodes)
        pool_flux = []
        subrnodes = np.asarray(disk.subrnodes(), dtype=float)
        for trait, pnames in zip(
                disk.traits('rpt'), disk.trait_params('rpt').pnames):
            trait_params = {
                name: params[prefixed] for name, prefixed in pnames.items()}
            pool_flux.extend(np.atleast_1d(
                trait.cloud_flux(trait_params, subrnodes)))
        pool_flux = np.asarray(pool_flux, np.float64)

        # Each pool has as many clouds as its flux needs at cflux each
        # (and at least one), which share its flux exactly: a negative
        # flux gives negative clouds
        nclouds = np.where(
            pool_flux != 0,
            np.maximum(np.rint(np.abs(pool_flux) / disk.clouds().cflux()), 1),
            0)
        if not np.sum(nclouds) <= _MAX_CLOUDS:
            raise ModelEvaluationError(
                f"the disk of clouds needs {np.sum(nclouds):.3g} clouds of "
                f"flux cflux = {disk.clouds().cflux()}, but the kernels draw "
                f"at most {_MAX_CLOUDS}; give it a larger cflux")
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
            seed=disk.clouds().seed(),
            nclouds=int(nclouds.sum()),
            ncloudscsum=self._s_ncloudscsum[1],
            has_analytical_integral=self._s_has_analytical_integral[1],
            **grid_and_outputs)
