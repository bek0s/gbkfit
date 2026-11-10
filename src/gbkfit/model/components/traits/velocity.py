"""
The velocity polar traits (vpt): speeds along their directions (see
VPTrait), and how the rotation in the gravity of a galaxy is corrected
for pressure (PressureSupport).
"""

import abc
from collections.abc import Sequence
from typing import Annotated, Any

import numpy as np
from annotated_types import Ge

from gbkfit.config import TypedParser, TypedSerializable, check_options
from gbkfit.params.pdescs import ParamDesc, ParamScalarDesc, ParamVectorDesc
from .base import RotationInput, VPTrait, VPTraitPlanar, WaveInput
from .options import (
    SAMPLING_DEFAULT, SAMPLING_TAIL, Direction,
    Harmonic, Sampling, SamplingChoice, Spiral, SpiralWavenumber,
    SpiralWinding, VPTDirectionChoice)
from .shapes import PT_UID_NW_HARMONIC, PT_UID_NW_UNIFORM, PT_UID_UNIFORM


__all__ = [
    'PressureSupport',
    'PressureSupportJeans',
    'PressureSupportSelfGravitating',
    'VPTraitArctan',
    'VPTraitBar',
    'VPTraitBoissier',
    'VPTraitBrandt',
    'VPTraitCourteau',
    'VPTraitEpinat',
    'VPTraitIso',
    'VPTraitLRamp',
    'VPTraitMass',
    'VPTraitNFW',
    'VPTraitNWBar',
    'VPTraitNWHarmonic',
    'VPTraitNWUniform',
    'VPTraitPolyex',
    'VPTraitRix',
    'VPTraitSpiral',
    'VPTraitTanh',
    'VPTraitUniform',
    'VPTraitWedge',
    'pressure_support_parser'
]


# The uids of the velocity polar traits of their own in the native
# kernels (from 201: see shapes for those of the shapes it shares)
VPT_UID_ARCTAN = 201
VPT_UID_BOISSIER = 202
VPT_UID_EPINAT = 203
VPT_UID_LRAMP = 204
VPT_UID_TANH = 205
VPT_UID_POLYEX = 206
VPT_UID_RIX = 207
VPT_UID_COURTEAU = 208
VPT_UID_BRANDT = 209
VPT_UID_ISO = 210
VPT_UID_NFW = 211
VPT_UID_WEDGE = 212
VPT_UID_BAR = 213
VPT_UID_NW_BAR = 214
VPT_UID_SPIRAL = 215
VPT_UID_TABLE = 216


def _splits_bar(r: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """
    Return the azimuths (degrees) at which the azimuthal means split the
    circles of the radii r across a flow along a bar (see VPTraitBar):
    where they cross the edges of its strip, at p - a, p + a, p + 180 - a
    and p + 180 + a (sin(a) = s / (2 r)), or within it, its axis (a = 90
    degrees), where the flow turns.
    """
    r = np.asarray(r, dtype=float)[:, None]
    p = params['p']
    ratio = np.divide(
        np.abs(params['s']) / 2, r, out=np.full_like(r, np.inf), where=r > 0)
    a = np.rad2deg(np.arcsin(np.minimum(ratio, 1)))
    return np.concatenate((p - a, p + a, p + 180 - a, p + 180 + a), axis=1)


class PressureSupport(TypedSerializable, abc.ABC):
    """
    How the rotation of a disk in the gravity of its galaxy is corrected
    for its pressure (the option pressure_support of a disk; see
    VPTraitMass): with the density of the disk, or that of its fluid, the
    mass disks of its option fluid, whose densities add up (e.g. its
    atomic and its molecular gas), with the dispersion of the disk. A
    disk whose dispersion is not isotropic (see the package docstring)
    takes the Jeans equation of its velocity ellipsoid: v^2 = v_c^2 + (R /
    rho) d(rho sigma_R^2)/dR + sigma_R^2 - sigma_phi^2. The dispersion is
    that of its dispersion traits, with that of its vertical equilibrium
    in quadrature, if any (see DPTraitEquilibrium); the azimuthal means
    are those of mass.SurfaceComponent, and the derivative is by the
    second-order three-point formula on the rings; NaN without density
    (see _profiles.pressure_support).
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())


class PressureSupportJeans(PressureSupport):
    """
    The pressure support of an isotropic gas at the midplane of its disk,
    the radial Jeans equation: v^2 = v_c^2 + (R / rho) d(rho sigma^2)/dR,
    with the azimuthal means of the density rho at the midplane (of the
    disk: its density traits times its density height traits at z = 0,
    or its density traits if thin, the equation of a thin disk; or of its
    fluid, at their midplane, as DysmalPy's pressure support of type 3)
    and of rho sigma^2, sigma its velocity dispersion at the midplane.
    See PressureSupport.
    """

    @staticmethod
    def type() -> str:
        return 'jeans'

    @staticmethod
    def type_title() -> str:
        return 'Jeans equation'


class PressureSupportSelfGravitating(PressureSupport):
    """
    The pressure support of a disk whose thickness its own gravity sets,
    h ~ sigma^2 / S, of its surface density S (its density traits, or the
    surface density of its fluid): rho ~ S^2 / sigma^2 and rho sigma^2 ~
    S^2, so that v^2 = v_c^2 + 2 sigma^2 dln S / dln R for an
    axisymmetric disk (Burkert et al. 2010), whatever its density height
    traits (DysmalPy's pressure support of types 1 and 2, for an
    exponential and a Sersic S). A disk whose dispersion is not isotropic
    takes sigma_z in rho. See PressureSupport.
    """

    @staticmethod
    def type() -> str:
        return 'self_gravitating'

    @staticmethod
    def type_title() -> str:
        return 'Self-gravitating disk'


pressure_support_parser = TypedParser(
    PressureSupport, [PressureSupportJeans, PressureSupportSelfGravitating],
    family='pressure_support')


class VPTraitUniform(VPTrait):
    """
    A uniform speed: v (e.g. a flat rotation curve). Its parameter: v,
    the speed.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'uniform'

    @staticmethod
    def type_title() -> str:
        return 'Uniform'

    @staticmethod
    def uid() -> int:
        return PT_UID_UNIFORM

    def __init__(
            self,
            direction: VPTDirectionChoice,
            name: str | None = None
    ):
        check_options(VPTraitUniform.__init__, locals())
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {self.speed(): ParamScalarDesc(
            title=self.speed_title(),
            desc='the speed, the same at every radius', unit='km/s',
            default=200)}


class VPTraitArctan(VPTrait):
    """
    An arctan profile of the speed: v (2 / pi) atan(r / rt). Its
    parameters: rt, the turnover radius, and v, the asymptotic speed.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'arctan'

    @staticmethod
    def type_title() -> str:
        return 'Arctan'

    @staticmethod
    def uid() -> int:
        return VPT_UID_ARCTAN

    def __init__(
            self,
            direction: VPTDirectionChoice,
            name: str | None = None
    ):
        check_options(VPTraitArctan.__init__, locals())
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(
                title='Turnover radius', desc='the radius of the turnover',
                min=0, unit='arcsec', default=1),
            self.speed(): ParamScalarDesc(
                title=self.speed_title(), desc='the asymptotic speed',
                unit='km/s', default=200)}


class VPTraitBoissier(VPTrait):
    """
    An exponential profile of the speed: v (1 - exp(-r / rt)). Its
    parameters: rt, the scale radius, and v, the asymptotic speed.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'boissier'

    @staticmethod
    def type_title() -> str:
        return 'Exponential (Boissier)'

    @staticmethod
    def uid() -> int:
        return VPT_UID_BOISSIER

    def __init__(
            self,
            direction: VPTDirectionChoice,
            name: str | None = None
    ):
        check_options(VPTraitBoissier.__init__, locals())
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(
                title='Scale radius', desc='the scale radius of the rise',
                min=0, unit='arcsec', default=1),
            self.speed(): ParamScalarDesc(
                title=self.speed_title(), desc='the asymptotic speed',
                unit='km/s', default=200)}


class VPTraitEpinat(VPTrait):
    """
    The profile of the rotation curves of Epinat et al. 2008: v (r /
    rt)^g / (1 + (r / rt)^a). Its parameters: rt, the turnover radius, v,
    the speed scale, a, the outer slope, and g, the inner slope.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'epinat'

    @staticmethod
    def type_title() -> str:
        return 'Epinat et al. (2008)'

    @staticmethod
    def uid() -> int:
        return VPT_UID_EPINAT

    def __init__(
            self,
            direction: VPTDirectionChoice,
            name: str | None = None
    ):
        check_options(VPTraitEpinat.__init__, locals())
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(
                title='Turnover radius', desc='the radius of the turnover',
                min=0, unit='arcsec', default=1),
            self.speed(): ParamScalarDesc(
                title=self.speed_title(), desc='the speed scale', unit='km/s',
                default=200),
            'a': ParamScalarDesc(
                title='Outer slope',
                desc='the outer slope: the speed goes as r^(g - a) far out',
                unit='', default=1),
            'g': ParamScalarDesc(
                title='Inner slope',
                desc='the inner slope: the speed goes as r^g near the centre',
                unit='', default=1)}


class VPTraitLRamp(VPTrait):
    """
    A speed that rises linearly to v at rt and is flat beyond: v min(r /
    rt, 1). Its parameters: rt and v.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'lramp'

    @staticmethod
    def type_title() -> str:
        return 'Linear ramp'

    @staticmethod
    def uid() -> int:
        return VPT_UID_LRAMP

    def __init__(
            self,
            direction: VPTDirectionChoice,
            name: str | None = None
    ):
        check_options(VPTraitLRamp.__init__, locals())
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(
                title='Turnover radius',
                desc='the radius where the linear rise ends', min=0,
                unit='arcsec', default=1),
            self.speed(): ParamScalarDesc(
                title=self.speed_title(), desc='the speed at rt and beyond',
                unit='km/s', default=200)}


class VPTraitTanh(VPTrait):
    """
    A tanh profile of the speed: v tanh(r / rt). Its parameters: rt, the
    turnover radius, and v, the asymptotic speed.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'tanh'

    @staticmethod
    def type_title() -> str:
        return 'Tanh'

    @staticmethod
    def uid() -> int:
        return VPT_UID_TANH

    def __init__(
            self,
            direction: VPTDirectionChoice,
            name: str | None = None
    ):
        check_options(VPTraitTanh.__init__, locals())
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(
                title='Turnover radius', desc='the radius of the turnover',
                min=0, unit='arcsec', default=1),
            self.speed(): ParamScalarDesc(
                title=self.speed_title(), desc='the asymptotic speed',
                unit='km/s', default=200)}


class VPTraitPolyex(VPTrait):
    """
    The polyex profile of the rotation curves of Giovanelli and Haynes
    (2002): v (1 - exp(-r / rt)) (1 + a r / rt). Its parameters: rt, the
    scale radius, v, the speed scale, and a, the slope of its outer part.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'polyex'

    @staticmethod
    def type_title() -> str:
        return 'Polyex'

    @staticmethod
    def uid() -> int:
        return VPT_UID_POLYEX

    def __init__(
            self,
            direction: VPTDirectionChoice,
            name: str | None = None
    ):
        check_options(VPTraitPolyex.__init__, locals())
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(
                title='Scale radius',
                desc='the scale radius of the inner rise', min=0,
                unit='arcsec', default=1),
            self.speed(): ParamScalarDesc(
                title=self.speed_title(), desc='the speed scale', unit='km/s',
                default=200),
            'a': ParamScalarDesc(
                title='Outer slope',
                desc='the slope of the outer part, in speed scales per rt',
                unit='', default=0)}


class VPTraitRix(VPTrait):
    """
    The profile of the rotation curves of Rix et al. 1997: v (1 + r /
    rt)^b (1 + (r / rt)^-g)^(-1 / g). Its parameters: rt, the turnover
    radius, v, the speed scale, g, the sharpness of the turnover, and b,
    the power law of the profile at large radii.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'rix'

    @staticmethod
    def type_title() -> str:
        return 'Rix et al. (1997)'

    @staticmethod
    def uid() -> int:
        return VPT_UID_RIX

    def __init__(
            self,
            direction: VPTDirectionChoice,
            name: str | None = None
    ):
        check_options(VPTraitRix.__init__, locals())
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(
                title='Turnover radius', desc='the radius of the turnover',
                min=0, unit='arcsec', default=1),
            self.speed(): ParamScalarDesc(
                title=self.speed_title(), desc='the speed scale', unit='km/s',
                default=200),
            'b': ParamScalarDesc(
                title='Outer slope',
                desc='the power law of the speed at large radii', unit='',
                default=0),
            'g': ParamScalarDesc(
                title='Turnover sharpness',
                desc='the sharpness of the turnover', min=0, unit='',
                default=2)}


class VPTraitCourteau(VPTrait):
    """
    The profile of the rotation curves of Courteau (1997): v (1 + x)^b /
    (1 + x^g)^(1 / g), with x = rt / r. Its parameters: rt, the turnover
    radius, v, the speed scale, and b and g, its shape.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'courteau'

    @staticmethod
    def type_title() -> str:
        return 'Courteau (1997)'

    @staticmethod
    def uid() -> int:
        return VPT_UID_COURTEAU

    def __init__(
            self,
            direction: VPTDirectionChoice,
            name: str | None = None
    ):
        check_options(VPTraitCourteau.__init__, locals())
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(
                title='Turnover radius', desc='the radius of the turnover',
                min=0, unit='arcsec', default=1),
            self.speed(): ParamScalarDesc(
                title=self.speed_title(), desc='the speed scale', unit='km/s',
                default=200),
            'b': ParamScalarDesc(
                title='Outer slope',
                desc='the fall (b > 0) or the rise (b < 0) of the speed to '
                     'its scale beyond the turnover', unit='', default=0),
            'g': ParamScalarDesc(
                title='Turnover sharpness',
                desc='the sharpness of the turnover', min=0, unit='',
                default=2)}


class VPTraitBrandt(VPTrait):
    """
    The profile of the rotation curves of Brandt (1960), of maximum v at
    the radius rt: v (r / rt) / (1/3 + 2/3 (r / rt)^n)^(3 / (2 n)). Its
    parameters: rt, v and n, the sharpness of the turnover.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'brandt'

    @staticmethod
    def type_title() -> str:
        return 'Brandt (1960)'

    @staticmethod
    def uid() -> int:
        return VPT_UID_BRANDT

    def __init__(
            self,
            direction: VPTDirectionChoice,
            name: str | None = None
    ):
        check_options(VPTraitBrandt.__init__, locals())
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(
                title='Peak radius', desc='the radius of the maximum speed',
                min=0, unit='arcsec', default=1),
            self.speed(): ParamScalarDesc(
                title=self.speed_title(), desc='the maximum speed, at rt',
                unit='km/s', default=200),
            'n': ParamScalarDesc(
                title='Turnover sharpness',
                desc='the sharpness of the turnover', min=0, unit='',
                default=1)}


class VPTraitIso(VPTrait):
    """
    The rotation curve of a pseudo-isothermal sphere of core radius rt
    and asymptotic speed v: v sqrt(1 - (rt / r) atan(r / rt)). Its
    parameters: rt and v.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'iso'

    @staticmethod
    def type_title() -> str:
        return 'Pseudo-isothermal sphere'

    @staticmethod
    def uid() -> int:
        return VPT_UID_ISO

    def __init__(
            self,
            direction: VPTDirectionChoice,
            name: str | None = None
    ):
        check_options(VPTraitIso.__init__, locals())
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(
                title='Core radius', desc='the core radius of the sphere',
                min=0, unit='arcsec', default=1),
            self.speed(): ParamScalarDesc(
                title=self.speed_title(), desc='the asymptotic speed',
                unit='km/s', default=200)}


class VPTraitNFW(VPTrait):
    """
    The rotation curve of an NFW halo of scale radius rt, of maximum v
    (at 2.163 rt). It is v sqrt(f(r / rt) / f_max), with f(u) = (ln(1 +
    u) - u / (1 + u)) / u. Its parameters: rt and v.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nfw'

    @staticmethod
    def type_title() -> str:
        return 'NFW halo'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NFW

    def __init__(
            self,
            direction: VPTDirectionChoice,
            name: str | None = None
    ):
        check_options(VPTraitNFW.__init__, locals())
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(
                title='Scale radius', desc='the scale radius of the halo',
                min=0, unit='arcsec', default=1),
            self.speed(): ParamScalarDesc(
                title=self.speed_title(),
                desc='the maximum speed, at 2.163 rt', unit='km/s',
                default=200)}


class VPTraitNWUniform(VPTrait):
    """
    A speed given at each ring: v(r). Its parameter: v, node-wise.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rings',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_uniform'

    @staticmethod
    def type_title() -> str:
        return 'Uniform (per ring)'

    @staticmethod
    def uid() -> int:
        return PT_UID_NW_UNIFORM

    def __init__(
            self,
            direction: VPTDirectionChoice,
            sampling: SamplingChoice = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        check_options(VPTraitNWUniform.__init__, locals())
        super().__init__(Direction(direction), Sampling(sampling), name=name)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return {self.speed(): ParamVectorDesc(
            nnodes, title=self.speed_title(), desc='the speed at each ring',
            unit='km/s', default=200)}


class VPTraitNWHarmonic(VPTrait):
    """
    A harmonic speed: a(r) cos(m (theta - p(r))), of the order m. Its
    parameters, node-wise: a, the amplitude, and p (but for order 0), the
    azimuth of a maximum.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    order : int
        The order m of the harmonic: the pattern repeats every 360 / m
        degrees, and order 0 is an axisymmetric a(r).
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rings',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_harmonic'

    @staticmethod
    def type_title() -> str:
        return 'Harmonic (per ring)'

    @staticmethod
    def uid() -> int:
        return PT_UID_NW_HARMONIC

    def __init__(
            self,
            direction: VPTDirectionChoice,
            order: Annotated[int, Ge(0)],
            sampling: SamplingChoice = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        check_options(VPTraitNWHarmonic.__init__, locals())
        super().__init__(
            Direction(direction), Harmonic(order), Sampling(sampling),
            name=name)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return self._option(Harmonic).params(nnodes, unit='km/s')

    def axisymmetric(self) -> bool:
        # (a harmonic of order 0 is uniform)
        return self._option(Harmonic).order == 0


class VPTraitWedge(VPTrait):
    """
    A uniform speed within two opposite wedges of a disk, and 0 outside
    them (e.g. a radial flow). As DysmalPy's UniformWedgeFlow, it is v at
    the azimuths within s / 2 of p or of p + 180 degrees, |cos(theta - p)|
    >= cos(s / 2), and 0 at the others. Its parameters: v, p (degrees)
    and s (degrees, the full width of each wedge).

    Its wedges have sharp edges, which p and s place: the model, which
    samples the wedges at points, does not change while an edge moves
    between two of them, so that fits by the slopes of the chi-square
    (least squares) do not move p and s, and give them errors of 0. Fit
    them with a fitter that takes no slopes (e.g. CMA-ES, or a sampler).

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'wedge'

    @staticmethod
    def type_title() -> str:
        return 'Wedge flow'

    @staticmethod
    def uid() -> int:
        return VPT_UID_WEDGE

    def __init__(
            self,
            direction: VPTDirectionChoice,
            name: str | None = None
    ):
        check_options(VPTraitWedge.__init__, locals())
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            self.speed(): ParamScalarDesc(
                title=self.speed_title(), desc='the speed within the wedges',
                unit='km/s', default=50),
            'p': ParamScalarDesc(
                title='Wedge azimuth',
                desc='the azimuth of the middle of a wedge', unit='deg',
                default=0, period=180),
            's': ParamScalarDesc(
                title='Wedge width', desc='the full width of each wedge',
                min=0, max=360, unit='deg', default=30)}

    def axisymmetric(self) -> bool:
        return False

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        p, s = params['p'], params['s']
        return np.tile(
            [p - s / 2, p + s / 2, p + 180 - s / 2, p + 180 + s / 2],
            (len(r), 1))


class VPTraitBar(VPTraitPlanar):
    """
    A uniform flow along a bar, in the plane of a disk, away from its
    centre on both sides (inwards if v < 0). As DysmalPy's
    UniformBarFlow, it is the speed v along the axis of the bar within the
    strip of the full width s along the axis, and 0 outside. The axis is
    at the azimuth p (and p + 180 degrees); at the distance x along it
    (positive towards p) and y across it, the velocity is v sign(x) along
    the axis where |y| < s / 2, with v_y = v sign(x) sin(p). The strip
    reaches the edge of the disk. Its parameters: v, p (degrees) and s
    (arcsec).

    Its strip has sharp edges, which s places: the model, which samples
    the strip at points, does not change while an edge moves between two
    of them, so that fits by the slopes of the chi-square (least squares)
    do not move s, and give it an error of 0. Fit it with a fitter that
    takes no slopes (e.g. CMA-ES, or a sampler).

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters (see Trait).
    """

    @staticmethod
    def type() -> str:
        return 'bar'

    @staticmethod
    def type_title() -> str:
        return 'Bar flow'

    @staticmethod
    def uid() -> int:
        return VPT_UID_BAR

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            v=ParamScalarDesc(
                title='Flow speed',
                desc='the speed along the axis, away from the centre',
                unit='km/s', default=50),
            p=ParamScalarDesc(
                title='Bar azimuth', desc='the azimuth of the axis of the bar',
                unit='deg', default=0, period=180),
            s=ParamScalarDesc(
                title='Bar width',
                desc='the full width of the strip of the flow', min=0,
                unit='arcsec', default=1))

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        return _splits_bar(r, params)


class VPTraitNWBar(VPTraitPlanar):
    """
    A flow along a bar whose speed is given at distances along it from
    its centre. As DysmalPy's VariableXBarFlow, it is as VPTraitBar, with
    the speed v(|x|) at the distance |x| from the centre along the axis,
    of a value at each node (at the distances of the radii of the
    nodes); 0 within the first or beyond the last (e.g. a bar of a
    length: speeds of 0 beyond it). Its parameters: p (degrees) and s
    (arcsec); v, node-wise. Fit s with a fitter that takes no slopes (see
    VPTraitBar).

    Parameters
    ----------
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rings',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_bar'

    @staticmethod
    def type_title() -> str:
        return 'Bar flow (per ring)'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_BAR

    def __init__(
            self,
            sampling: SamplingChoice = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        check_options(VPTraitNWBar.__init__, locals())
        super().__init__(Sampling(sampling), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            p=ParamScalarDesc(
                title='Bar azimuth', desc='the azimuth of the axis of the bar',
                unit='deg', default=0, period=180),
            s=ParamScalarDesc(
                title='Bar width',
                desc='the full width of the strip of the flow', min=0,
                unit='arcsec', default=1))

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(v=ParamVectorDesc(
            nnodes, title='Flow speed',
            desc='the speed along the axis, away from the centre, at '
                 'the distance of each ring', unit='km/s', default=50))

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        return _splits_bar(r, params)


class VPTraitSpiral(VPTraitPlanar):
    """
    The velocities of a spiral density wave in the gas of a disk. As
    DysmalPy's SpiralDensityWave (see Spiral), they are the radial part
    v_R = -e m (Omega - o) / k cos(phi) and the tangential part v_t = e
    kappa^2 / (2 k Omega) sin(phi), 0 where there is no wave, where k is
    0 or where the disk does not rotate. Towards the edges of the radii of
    a wave of the dispersion relation, k tends to 0 and the velocity grows
    without bound (v_R = -e c / eta, eta = k c / (m |Omega - o|)): the
    linear theory does not hold there. Its density is the density trait
    'spiral' (see RPTraitSpiral).

    Parameters
    ----------
    order : int
        The number m of arms, at least 1.
    winding : str, optional
        How the arms wind with the wavenumber of the dispersion relation:
        'trailing' (the default) or 'leading'.
    wavenumber : str, optional
        Where the wavenumber comes from: 'dispersion' (the dispersion
        relation, the default) or 'given' (the parameter k).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'spiral'

    @staticmethod
    def type_title() -> str:
        return 'Spiral density wave'

    @staticmethod
    def uid() -> int:
        return VPT_UID_SPIRAL

    def __init__(
            self,
            order: Annotated[int, Ge(1)],
            winding: SpiralWinding | None = None,
            wavenumber: SpiralWavenumber = 'dispersion',
            name: str | None = None
    ):
        check_options(VPTraitSpiral.__init__, locals())
        super().__init__(Spiral(order, winding, wavenumber), name=name)

    def sampling(self) -> str:
        return 'subrings'

    def consts(self) -> tuple[float, ...]:
        return self._option(Spiral).consts(len(self.params_sm()))

    def takes_rotation(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return self._option(Spiral).params_sm(pattern_speed=True)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return self._option(Spiral).params_rnw(
            self.from_galaxy().params, nnodes)

    def from_galaxy(self) -> WaveInput:
        return WaveInput(('ar', 'at', 'f'))

    def wave(
            self,
            r: np.ndarray,
            v: np.ndarray,
            dv: np.ndarray,
            density: np.ndarray | None,
            params: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """
        Return the tables of the wave (see Spiral.wavenumber_at;
        it does not take the density): the amplitudes of its radial and
        tangential velocities ('ar' and 'at', km/s; v_R = ar cos(phi) and
        v_t = at sin(phi)) and f ('f', radians), by name.
        """
        spiral = self._option(Spiral)
        m, e, o = spiral.order, float(params['e']), float(params['o'])
        omega, kappa2, k, waves = spiral.wavenumber_at(r, v, dv, params)
        moving = waves & (k != 0) & (omega != 0)
        return dict(
            ar=np.divide(
                -e * m * (omega - o), k, out=np.zeros_like(k), where=moving),
            at=np.divide(
                e * kappa2, 2 * k * omega, out=np.zeros_like(k),
                where=moving),
            f=Spiral.phase(r, k))


class VPTraitMass(VPTrait):
    """
    The rotation curve of the mass of its galaxy, which the galaxy
    computes at each evaluation, with or without pressure support. It is
    the circular velocity (see gbkfit.model.mass) at the rings it is
    evaluated on, and beyond its last ring, at the radii of its tail (see
    SAMPLING_TAIL; linear between them, and beyond the last, its value
    there), from the parameters of the distance and the mass of its
    galaxy, and with pressure support, the rotation of the gas of its
    disk in that gravity; 0 where its square is negative (the mass pulls
    outwards, e.g. inside a ring, or the pressure exceeds the gravity: no
    rotation, as DysmalPy takes it), NaN where it is infinite (there is
    no circular orbit, e.g. at a jump of the surface density of a thin
    disk). A tangential speed (it has no option direction). It has no
    parameters of its own, and its galaxy must have mass. With the
    pressure support of its disk (see PressureSupport), the rotation of
    the gas of the disk in that gravity.

    The rotation is the same at every height, or with the velocity height
    trait 'mass', that of the gravity at each height (see VHTraitMass).

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters (see Trait).
    """

    @staticmethod
    def type() -> str:
        return 'mass'

    @staticmethod
    def type_title() -> str:
        return 'Rotation from the mass'

    @staticmethod
    def uid() -> int:
        return VPT_UID_TABLE

    def __init__(self, name: str | None = None):
        check_options(VPTraitMass.__init__, locals())
        super().__init__(name=name)

    def direction(self) -> str:
        return 'tan'

    def sampling(self) -> str:
        return SAMPLING_TAIL

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(vt=ParamVectorDesc(
            nnodes, title='Rotation speed',
            desc='the rotation in the gravity of the galaxy', unit='km/s',
            default=200))

    def from_galaxy(self) -> RotationInput:
        return RotationInput(('vt',))
