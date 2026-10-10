from collections.abc import Sequence
from typing import Any, Self

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.math.interpolation import Interpolator
from gbkfit.params import ParamDesc
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .._detail import dump_geometry, dump_lines, dump_name
from ..base import (
    BrightnessComponent3D, ComponentPlan, OpacityComponent3D,
    SpectralComponent3D)
from ..geometries import Geometry
from ..lines import Line, line_parser
from . import _detail, _disk, traits
from ._detail import (
    BHT, BPT, DHT, DPT, OHT, OPT, SPT, VHT, VPT, WPT, ZPT)


__all__ = [
    'BrightnessMCDisk3D',
    'SpectralMCDisk3D',
    'OpacityMCDisk3D'
]


# The most clouds the kernels draw: they count them in int32
_MAX_CLOUDS = np.iinfo(np.int32).max


class MCDisk(_disk.Disk):
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
            prefixes: dict[str, str],
            rdata_key: str
    ):
        super().__init__(
            loose, tilted, rnodes, rstep, interp, traits_,
            prefixes, rdata_key)
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
            self, driver: Driver, nlines: int, dtype: np.dtype
    ) -> _disk.DiskPlan:
        return MCDiskPlan(self, driver, nlines, dtype)


class MCDiskPlan(_disk.DiskPlan):
    """The evaluation of an MCDisk."""

    def __init__(
            self, disk: MCDisk, driver: Driver, nlines: int, dtype: np.dtype):
        super().__init__(disk, driver, nlines, dtype)
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

    def _impl_evaluate(
            self,
            params: dict[str, float | np.ndarray],
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


class BrightnessMCDisk3D(BrightnessComponent3D):
    """
    A thick disk of brightness, made of clouds drawn at random from
    its traits (a component of intensity_3d).

    Parameters
    ----------
    bptraits : Trait or Sequence of Trait
        Its surface brightness polar traits: the brightness as a function
        of radius and azimuth (see traits).
    bhtraits : Trait or Sequence of Trait
        Its surface brightness height traits: the vertical distribution of
        the brightness, one for each polar trait.
    zptraits : Trait or Sequence of Trait, optional
        Its vertical distortion polar traits (e.g. a warp out of its
        plane).
    sptraits : Trait or Sequence of Trait, optional
        Its selection polar traits: the parts of the disk that are seen
        (e.g. a range of azimuths).
    wptraits : Trait or Sequence of Trait, optional
        Its weight polar traits: the spatial weights of the data.
    loose : bool, optional
        Whether its centre and systemic velocity are node-wise (a value
        for each ring); by default, not, or those of its geometry.
    tilted : bool, optional
        Whether its position angle and inclination are node-wise; by
        default, not, or those of its geometry.
    rnmin, rnmax, rnsep, rnlen, rnodes : optional
        The radii of its rings (arcsec): rnodes; or rnmin, rnmax and rnsep
        (every rnsep); or rnmin, rnmax and rnlen (rnlen rings). Those of
        its geometry if it is warped.
    rstep : float, optional
        The width of the rings it is evaluated on, at most half the
        smallest separation of its radii; by default that, or 1 if less.
    interp : str, optional
        How its node-wise parameters are interpolated between its radii:
        'linear', 'akima' or 'pchip'.
    cflux : float
        The flux of each of its clouds.
    seed : int, optional
        The seed of the random numbers of its clouds.
    geometry : Geometry, optional
        A geometry that it shares with other components (see Geometry).
    name : str, optional
        Its name (see Component).

    Raises
    ------
    ConfigError
        If the options are not valid (e.g. the rings, or the number of
        traits), or conflict with its geometry.
    """

    _slots = (BPT, BHT, ZPT, SPT, WPT)

    @staticmethod
    def type() -> str:
        return 'mcdisk'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        return cls(**_detail.load_options(cls, info, cls._slots))

    def dump(self) -> dict[str, Any]:
        return (
            dict(type=self.type())
            | dump_name(self)
            | dump_geometry(self)
            | _detail.dump_disk(self._disk, self._slots, self.geometry()))

    def __init__(
            self,
            cflux: float,
            bptraits: traits.BPTrait | Sequence[traits.BPTrait],
            bhtraits: traits.BHTrait | Sequence[traits.BHTrait],
            zptraits: traits.ZPTrait | Sequence[traits.ZPTrait] | None = None,
            sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None = None,
            wptraits: traits.WPTrait | Sequence[traits.WPTrait] | None = None,
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
        super().__init__(name, geometry)
        self._disk = _detail.make_disk(
            type(self), MCDisk, self._slots,
            rdata_key='bdata',
            geometry=geometry, loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            traits_=dict(
                bptraits=bptraits, bhtraits=bhtraits, zptraits=zptraits,
                sptraits=sptraits, wptraits=wptraits),
            cflux=cflux, seed=seed)

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._disk.pdescs()

    def has_weights(self) -> bool:
        return bool(self._disk.traits('wpt'))

    def constants(self) -> dict[str, Any]:
        return dict(
            rnodes=self._disk.rnodes(), subrnodes=self._disk.subrnodes())

    def plan(
            self,
            driver: Driver,
            spectral: gridutils.Grid,
            dtype: np.dtype,
            lines: tuple[str, ...] | None
    ) -> ComponentPlan:
        return _detail.DiskComponentPlan(
            self, self._disk.plan(driver, 0, dtype))

    def disk_outputs(
            self, outputs: dict[str, DeviceArray | None]
    ) -> dict[str, DeviceArray | None]:
        """Return the outputs of the disk, from those of the component."""
        # The density of the disk is its brightness
        return dict(
            image=outputs['image'],
            wdata=outputs['wdata'],
            rdata=outputs['bdata'],
            opacity=outputs['odata'],
            ordata=outputs['obdata'])


class SpectralMCDisk3D(SpectralComponent3D):
    """
    A thick disk of emission lines, made of clouds drawn at random
    from its traits (a component of kinematics_3d).

    Parameters
    ----------
    bptraits : Trait or Sequence of Trait
        Its surface brightness polar traits: the brightness as a function
        of radius and azimuth (see traits).
    bhtraits : Trait or Sequence of Trait
        Its surface brightness height traits: the vertical distribution of
        the brightness, one for each polar trait.
    vptraits : Trait or Sequence of Trait
        Its velocity polar traits (e.g. the rotation curve).
    vhtraits : Trait or Sequence of Trait, optional
        Its velocity height traits, one for each velocity polar trait; by
        default, the velocity does not change with height.
    dptraits : Trait or Sequence of Trait
        Its velocity dispersion polar traits.
    dhtraits : Trait or Sequence of Trait, optional
        Its velocity dispersion height traits, one for each dispersion
        polar trait; by default, the dispersion does not change with
        height.
    zptraits : Trait or Sequence of Trait, optional
        Its vertical distortion polar traits (e.g. a warp out of its
        plane).
    sptraits : Trait or Sequence of Trait, optional
        Its selection polar traits: the parts of the disk that are seen
        (e.g. a range of azimuths).
    wptraits : Trait or Sequence of Trait, optional
        Its weight polar traits: the spatial weights of the data.
    loose : bool, optional
        Whether its centre and systemic velocity are node-wise (a value
        for each ring); by default, not, or those of its geometry.
    tilted : bool, optional
        Whether its position angle and inclination are node-wise; by
        default, not, or those of its geometry.
    rnmin, rnmax, rnsep, rnlen, rnodes : optional
        The radii of its rings (arcsec): rnodes; or rnmin, rnmax and rnsep
        (every rnsep); or rnmin, rnmax and rnlen (rnlen rings). Those of
        its geometry if it is warped.
    rstep : float, optional
        The width of the rings it is evaluated on, at most half the
        smallest separation of its radii; by default that, or 1 if less.
    interp : str, optional
        How its node-wise parameters are interpolated between its radii:
        'linear', 'akima' or 'pchip'.
    cflux : float
        The flux of each of its clouds.
    seed : int, optional
        The seed of the random numbers of its clouds.
    lines : Sequence of Line, optional
        Its emission lines (see Lines); by default, one line at the
        velocity of the spectral axis.
    geometry : Geometry, optional
        A geometry that it shares with other components (see Geometry).
    name : str, optional
        Its name (see Component).

    Raises
    ------
    ConfigError
        If the options are not valid (e.g. the rings, or the number of
        traits), or conflict with its geometry.
    """

    _slots = (BPT, BHT, VPT, VHT, DPT, DHT, ZPT, SPT, WPT)

    @staticmethod
    def type() -> str:
        return 'mcdisk'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        parseutils.load_option_and_update_info(
            line_parser, info, 'lines')
        return cls(**_detail.load_options(cls, info, cls._slots))

    def dump(self) -> dict[str, Any]:
        return (
            dict(type=self.type())
            | dump_name(self)
            | dump_geometry(self)
            | _detail.dump_disk(self._disk, self._slots, self.geometry())
            | dump_lines(self._lines))

    def __init__(
            self,
            cflux: float,
            bptraits: traits.BPTrait | Sequence[traits.BPTrait],
            vptraits: traits.VPTrait | Sequence[traits.VPTrait],
            dptraits: traits.DPTrait | Sequence[traits.DPTrait],
            bhtraits: traits.BHTrait | Sequence[traits.BHTrait],
            vhtraits: traits.VHTrait | Sequence[traits.VHTrait] | None = None,
            dhtraits: traits.DHTrait | Sequence[traits.DHTrait] | None = None,
            zptraits: traits.ZPTrait | Sequence[traits.ZPTrait] | None = None,
            sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None = None,
            wptraits: traits.WPTrait | Sequence[traits.WPTrait] | None = None,
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
            lines: Sequence[Line] | None = None,
            geometry: Geometry | None = None,
            name: str | None = None
    ):
        super().__init__(name, geometry)
        self._disk = _detail.make_disk(
            type(self), MCDisk, self._slots,
            rdata_key='bdata',
            geometry=geometry, loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            traits_=dict(
                bptraits=bptraits, bhtraits=bhtraits, vptraits=vptraits,
                vhtraits=vhtraits, dptraits=dptraits, dhtraits=dhtraits,
                zptraits=zptraits, sptraits=sptraits, wptraits=wptraits),
            cflux=cflux, seed=seed)
        self._lines = _detail.make_lines(self._disk, lines)

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._disk.pdescs() | self._lines.pdescs()

    def circular_velocity_params(self) -> dict[str, tuple[float, ...]]:
        return self._disk.circular_velocity_params()

    def has_weights(self) -> bool:
        return bool(self._disk.traits('wpt'))

    def constants(self) -> dict[str, Any]:
        return dict(
            rnodes=self._disk.rnodes(), subrnodes=self._disk.subrnodes())

    def line_names(self) -> tuple[str, ...]:
        return self._lines.names()

    def plan(
            self,
            driver: Driver,
            spectral: gridutils.Grid,
            dtype: np.dtype,
            lines: tuple[str, ...] | None
    ) -> ComponentPlan:
        selected = self._lines.select(lines)
        return _detail.SpectralDiskComponentPlan(
            self, self._disk.plan(driver, len(selected), dtype),
            self._lines, selected, spectral)

    def disk_outputs(
            self, outputs: dict[str, DeviceArray | None]
    ) -> dict[str, DeviceArray | None]:
        """Return the outputs of the disk, from those of the component."""
        # The density of the disk is its brightness
        return dict(
            scube=outputs['scube'],
            wdata=outputs['wdata'],
            rdata=outputs['bdata'],
            opacity=outputs['odata'],
            ordata=outputs['obdata'])


class OpacityMCDisk3D(OpacityComponent3D):
    """
    A thick disk of opacity, made of clouds drawn at random from its
    traits (an opacity component of intensity_3d and kinematics_3d).

    Parameters
    ----------
    optraits : Trait or Sequence of Trait
        Its opacity polar traits: the optical depth of the disk seen face-
        on.
    ohtraits : Trait or Sequence of Trait
        Its opacity height traits: the vertical distribution of the
        optical depth, one for each polar trait.
    zptraits : Trait or Sequence of Trait, optional
        Its vertical distortion polar traits (e.g. a warp out of its
        plane).
    sptraits : Trait or Sequence of Trait, optional
        Its selection polar traits: the parts of the disk that are seen
        (e.g. a range of azimuths).
    wptraits : Trait or Sequence of Trait, optional
        Its weight polar traits: the spatial weights of the data.
    loose : bool, optional
        Whether its centre and systemic velocity are node-wise (a value
        for each ring); by default, not, or those of its geometry.
    tilted : bool, optional
        Whether its position angle and inclination are node-wise; by
        default, not, or those of its geometry.
    rnmin, rnmax, rnsep, rnlen, rnodes : optional
        The radii of its rings (arcsec): rnodes; or rnmin, rnmax and rnsep
        (every rnsep); or rnmin, rnmax and rnlen (rnlen rings). Those of
        its geometry if it is warped.
    rstep : float, optional
        The width of the rings it is evaluated on, at most half the
        smallest separation of its radii; by default that, or 1 if less.
    interp : str, optional
        How its node-wise parameters are interpolated between its radii:
        'linear', 'akima' or 'pchip'.
    cflux : float
        The flux of each of its clouds.
    seed : int, optional
        The seed of the random numbers of its clouds.
    geometry : Geometry, optional
        A geometry that it shares with other components (see Geometry).
    name : str, optional
        Its name (see Component).

    Raises
    ------
    ConfigError
        If the options are not valid (e.g. the rings, or the number of
        traits), or conflict with its geometry.
    """

    _slots = (OPT, OHT, ZPT, SPT, WPT)

    @staticmethod
    def type() -> str:
        return 'mcdisk'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        return cls(**_detail.load_options(cls, info, cls._slots))

    def dump(self) -> dict[str, Any]:
        return (
            dict(type=self.type())
            | dump_name(self)
            | dump_geometry(self)
            | _detail.dump_disk(self._disk, self._slots, self.geometry()))

    def __init__(
            self,
            cflux: float,
            optraits: traits.OPTrait | Sequence[traits.OPTrait],
            ohtraits: traits.OHTrait | Sequence[traits.OHTrait],
            zptraits: traits.ZPTrait | Sequence[traits.ZPTrait] | None = None,
            sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None = None,
            wptraits: traits.WPTrait | Sequence[traits.WPTrait] | None = None,
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
        super().__init__(name, geometry)
        self._disk = _detail.make_disk(
            type(self), MCDisk, self._slots,
            rdata_key='odata',
            geometry=geometry, loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            traits_=dict(
                optraits=optraits, ohtraits=ohtraits, zptraits=zptraits,
                sptraits=sptraits, wptraits=wptraits),
            cflux=cflux, seed=seed)

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._disk.pdescs()

    def has_weights(self) -> bool:
        return bool(self._disk.traits('wpt'))

    def constants(self) -> dict[str, Any]:
        return dict(
            rnodes=self._disk.rnodes(), subrnodes=self._disk.subrnodes())

    def plan(
            self,
            driver: Driver,
            spectral: gridutils.Grid,
            dtype: np.dtype,
            lines: tuple[str, ...] | None
    ) -> ComponentPlan:
        return _detail.DiskComponentPlan(
            self, self._disk.plan(driver, 0, dtype))

    def disk_outputs(
            self, outputs: dict[str, DeviceArray | None]
    ) -> dict[str, DeviceArray | None]:
        """Return the outputs of the disk, from those of the component."""
        # The density of the disk is the opacity
        return dict(
            rdata=outputs['odata'])
