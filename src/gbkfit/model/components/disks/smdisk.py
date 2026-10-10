from collections.abc import Sequence
from typing import Any, Self

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamDesc
from gbkfit.utils import gridutils, parseutils
from .._detail import dump_geometry, dump_lines, dump_name
from ..base import (
    BrightnessComponent2D, BrightnessComponent3D, ComponentPlan,
    OpacityComponent3D, SpectralComponent2D, SpectralComponent3D)
from ..geometries import Geometry
from ..lines import Line, line_parser
from . import _detail, _disk, traits
from ._detail import (
    BHT, BPT, DHT, DPT, OHT, OPT, SPT, VHT, VPT, WPT, ZPT)


__all__ = [
    'BrightnessSMDisk2D',
    'BrightnessSMDisk3D',
    'SpectralSMDisk2D',
    'SpectralSMDisk3D',
    'OpacitySMDisk3D'
]


class SMDisk(_disk.Disk):
    """A disk evaluated smoothly: on the pixels or voxels of its grid."""

    def plan(
            self, driver: Driver, nlines: int, dtype: np.dtype
    ) -> _disk.DiskPlan:
        return SMDiskPlan(self, driver, nlines, dtype)


class SMDiskPlan(_disk.DiskPlan):
    """The evaluation of an SMDisk."""

    def _impl_evaluate(
            self,
            params: dict[str, float | np.ndarray],
            grid_and_outputs: dict[str, Any],
            out_extra: dict[str, Any] | None
    ) -> None:
        self._backend.smdisk_evaluate(self._native_disk, **grid_and_outputs)


class BrightnessSMDisk2D(BrightnessComponent2D):
    """
    A thin disk of brightness, evaluated smoothly on the pixels (a
    component of intensity_2d).

    Parameters
    ----------
    bptraits : Trait or Sequence of Trait
        Its surface brightness polar traits: the brightness as a function
        of radius and azimuth (see traits).
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

    _slots = (BPT, SPT, WPT)

    @staticmethod
    def type() -> str:
        return 'smdisk'

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
            bptraits: traits.BPTrait | Sequence[traits.BPTrait],
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
            geometry: Geometry | None = None,
            name: str | None = None
    ):
        super().__init__(name, geometry)
        self._disk = _detail.make_disk(
            type(self), SMDisk, self._slots,
            rdata_key='bdata',
            geometry=geometry, loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            traits_=dict(
                bptraits=bptraits, sptraits=sptraits, wptraits=wptraits))

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
            rdata=outputs['bdata'])


class BrightnessSMDisk3D(BrightnessComponent3D):
    """
    A thick disk of brightness, evaluated smoothly on the voxels (a
    component of intensity_3d).

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
        return 'smdisk'

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
            geometry: Geometry | None = None,
            name: str | None = None
    ):
        super().__init__(name, geometry)
        self._disk = _detail.make_disk(
            type(self), SMDisk, self._slots,
            rdata_key='bdata',
            geometry=geometry, loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            traits_=dict(
                bptraits=bptraits, bhtraits=bhtraits, zptraits=zptraits,
                sptraits=sptraits, wptraits=wptraits))

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


class SpectralSMDisk2D(SpectralComponent2D):
    """
    A thin disk of emission lines, evaluated smoothly on the pixels (a
    component of kinematics_2d).

    Parameters
    ----------
    bptraits : Trait or Sequence of Trait
        Its surface brightness polar traits: the brightness as a function
        of radius and azimuth (see traits).
    vptraits : Trait or Sequence of Trait
        Its velocity polar traits (e.g. the rotation curve).
    dptraits : Trait or Sequence of Trait
        Its velocity dispersion polar traits.
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

    _slots = (BPT, VPT, DPT, SPT, WPT)

    @staticmethod
    def type() -> str:
        return 'smdisk'

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
            bptraits: traits.BPTrait | Sequence[traits.BPTrait],
            vptraits: traits.VPTrait | Sequence[traits.VPTrait],
            dptraits: traits.DPTrait | Sequence[traits.DPTrait],
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
            lines: Sequence[Line] | None = None,
            geometry: Geometry | None = None,
            name: str | None = None
    ):
        super().__init__(name, geometry)
        self._disk = _detail.make_disk(
            type(self), SMDisk, self._slots,
            rdata_key='bdata',
            geometry=geometry, loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            traits_=dict(
                bptraits=bptraits, vptraits=vptraits, dptraits=dptraits,
                sptraits=sptraits, wptraits=wptraits))
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
            rdata=outputs['bdata'])


class SpectralSMDisk3D(SpectralComponent3D):
    """
    A thick disk of emission lines, evaluated smoothly on the voxels
    (a component of kinematics_3d).

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
        return 'smdisk'

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
            lines: Sequence[Line] | None = None,
            geometry: Geometry | None = None,
            name: str | None = None
    ):
        super().__init__(name, geometry)
        self._disk = _detail.make_disk(
            type(self), SMDisk, self._slots,
            rdata_key='bdata',
            geometry=geometry, loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            traits_=dict(
                bptraits=bptraits, bhtraits=bhtraits, vptraits=vptraits,
                vhtraits=vhtraits, dptraits=dptraits, dhtraits=dhtraits,
                zptraits=zptraits, sptraits=sptraits, wptraits=wptraits))
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


class OpacitySMDisk3D(OpacityComponent3D):
    """
    A thick disk of opacity, evaluated smoothly on the voxels (an
    opacity component of intensity_3d and kinematics_3d).

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
        return 'smdisk'

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
            geometry: Geometry | None = None,
            name: str | None = None
    ):
        super().__init__(name, geometry)
        self._disk = _detail.make_disk(
            type(self), SMDisk, self._slots,
            rdata_key='odata',
            geometry=geometry, loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            traits_=dict(
                optraits=optraits, ohtraits=ohtraits, zptraits=zptraits,
                sptraits=sptraits, wptraits=wptraits))

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
