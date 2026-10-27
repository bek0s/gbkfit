"""
The velocity polar traits (vpt): speeds along their directions (see
VPTrait), and how the rotation in the gravity of a galaxy is corrected
for pressure (PressureSupport).
"""

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.params.pdescs import ParamDesc, ParamScalarDesc, ParamVectorDesc
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from .base import RotationInput, VPTrait, VPTraitPlanar, WaveInput
from .options import Direction, Harmonic, SAMPLING_DEFAULT, Sampling, Spiral


__all__ = [
    'PressureSupport',
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
    'VPTraitWedge'
]


# Their uids in the native kernels
VPT_UID_UNIFORM = 1
VPT_UID_ARCTAN = 2
VPT_UID_BOISSIER = 3
VPT_UID_EPINAT = 4
VPT_UID_LRAMP = 5
VPT_UID_TANH = 6
VPT_UID_POLYEX = 7
VPT_UID_RIX = 8
VPT_UID_COURTEAU = 9
VPT_UID_BRANDT = 10
VPT_UID_ISO = 11
VPT_UID_NFW = 12
VPT_UID_WEDGE = 13
VPT_UID_BAR = 14
VPT_UID_NW_UNIFORM = 101
VPT_UID_NW_HARMONIC = 102
VPT_UID_NW_BAR = 103
VPT_UID_SPIRAL = 104


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


@dataclasses.dataclass(frozen=True)
class PressureSupport:
    """
    How the rotation of a disk in the gravity of its galaxy is corrected
    for its pressure (see VPTraitMass): mode, 'jeans' or
    'self_gravitating', and density, the names of the mass components of
    the galaxy whose densities it takes (their sum), or None for that of
    the disk.
    """
    mode: str
    density: tuple[str, ...] | None = None


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
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'uniform'

    @staticmethod
    def uid() -> int:
        return VPT_UID_UNIFORM

    def __init__(self, direction: str, name: str | None = None):
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {self.speed(): ParamScalarDesc()}


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
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'arctan'

    @staticmethod
    def uid() -> int:
        return VPT_UID_ARCTAN

    def __init__(self, direction: str, name: str | None = None):
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {'rt': ParamScalarDesc(), self.speed(): ParamScalarDesc()}


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
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'boissier'

    @staticmethod
    def uid() -> int:
        return VPT_UID_BOISSIER

    def __init__(self, direction: str, name: str | None = None):
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {'rt': ParamScalarDesc(), self.speed(): ParamScalarDesc()}


class VPTraitEpinat(VPTrait):
    """
    The profile of the rotation curves of Epinat et al. (2008): v (r /
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
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'epinat'

    @staticmethod
    def uid() -> int:
        return VPT_UID_EPINAT

    def __init__(self, direction: str, name: str | None = None):
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(),
            self.speed(): ParamScalarDesc(),
            'a': ParamScalarDesc(),
            'g': ParamScalarDesc()}


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
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'lramp'

    @staticmethod
    def uid() -> int:
        return VPT_UID_LRAMP

    def __init__(self, direction: str, name: str | None = None):
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {'rt': ParamScalarDesc(), self.speed(): ParamScalarDesc()}


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
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'tanh'

    @staticmethod
    def uid() -> int:
        return VPT_UID_TANH

    def __init__(self, direction: str, name: str | None = None):
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {'rt': ParamScalarDesc(), self.speed(): ParamScalarDesc()}


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
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'polyex'

    @staticmethod
    def uid() -> int:
        return VPT_UID_POLYEX

    def __init__(self, direction: str, name: str | None = None):
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(),
            self.speed(): ParamScalarDesc(),
            'a': ParamScalarDesc()}


class VPTraitRix(VPTrait):
    """
    The profile of the rotation curves of Rix et al. (1997): v (1 + r /
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
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'rix'

    @staticmethod
    def uid() -> int:
        return VPT_UID_RIX

    def __init__(self, direction: str, name: str | None = None):
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(),
            self.speed(): ParamScalarDesc(),
            'b': ParamScalarDesc(),
            'g': ParamScalarDesc()}


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
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'courteau'

    @staticmethod
    def uid() -> int:
        return VPT_UID_COURTEAU

    def __init__(self, direction: str, name: str | None = None):
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(),
            self.speed(): ParamScalarDesc(),
            'b': ParamScalarDesc(),
            'g': ParamScalarDesc()}


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
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'brandt'

    @staticmethod
    def uid() -> int:
        return VPT_UID_BRANDT

    def __init__(self, direction: str, name: str | None = None):
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(),
            self.speed(): ParamScalarDesc(),
            'n': ParamScalarDesc()}


class VPTraitIso(VPTrait):
    """
    The profile of the rotation curve of a pseudo-isothermal sphere of
    core radius rt and asymptotic speed v: v sqrt(1 - (rt / r) atan(r /
    rt)). Its parameters: rt and v.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'iso'

    @staticmethod
    def uid() -> int:
        return VPT_UID_ISO

    def __init__(self, direction: str, name: str | None = None):
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {'rt': ParamScalarDesc(), self.speed(): ParamScalarDesc()}


class VPTraitNFW(VPTrait):
    """
    The profile of the rotation curve of an NFW halo of scale radius rt,
    of maximum v (at 2.163 rt): v sqrt(f(r / rt) / f_max), with f(u) =
    (ln(1 + u) - u / (1 + u)) / u. Its parameters: rt and v.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nfw'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NFW

    def __init__(self, direction: str, name: str | None = None):
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {'rt': ParamScalarDesc(), self.speed(): ParamScalarDesc()}


class VPTraitNWUniform(VPTrait):
    """
    A speed given at each ring: v(r). Its parameter: v, node-wise.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'away', 'out' or 'los'
        (see the package docstring).
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_uniform'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_UNIFORM

    def __init__(
            self,
            direction: str,
            sampling: str = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        super().__init__(Direction(direction), Sampling(sampling), name=name)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return {self.speed(): ParamVectorDesc(nnodes)}


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
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_harmonic'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_HARMONIC

    def __init__(
            self,
            direction: str,
            order: int,
            sampling: str = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        super().__init__(
            Direction(direction), Harmonic(order), Sampling(sampling),
            name=name)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return self._option(Harmonic).params(nnodes)

    def axisymmetric(self) -> bool:
        # (a harmonic of order 0 is uniform)
        return self._option(Harmonic).order == 0


class VPTraitWedge(VPTrait):
    """
    A uniform speed within two opposite wedges of a disk (e.g. a radial
    flow, as DysmalPy's UniformWedgeFlow): v at the azimuths within s / 2
    of p or of p + 180 degrees, |cos(theta - p)| >= cos(s / 2), and 0 at
    the others. Its parameters: v, p (degrees) and s (degrees, the full
    width of each wedge).

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
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'wedge'

    @staticmethod
    def uid() -> int:
        return VPT_UID_WEDGE

    def __init__(self, direction: str, name: str | None = None):
        super().__init__(Direction(direction), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            self.speed(): ParamScalarDesc(),
            'p': ParamScalarDesc(),
            's': ParamScalarDesc()}

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
    A uniform flow along a bar, in the plane of a disk (as DysmalPy's
    UniformBarFlow): the speed v along the axis of the bar, away from its
    centre on both sides (inwards if v < 0), within the strip of the full
    width s along the axis, and 0 outside. The axis is at the azimuth p
    (and p + 180 degrees); at the distance x along it (positive towards
    p) and y across it, the velocity is v sign(x) along the axis where
    |y| < s / 2, with v_y = v sign(x) sin(p). The strip reaches the edge
    of the disk. Its parameters: v, p (degrees) and s (arcsec).

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
    def uid() -> int:
        return VPT_UID_BAR

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            v=ParamScalarDesc(),
            p=ParamScalarDesc(),
            s=ParamScalarDesc())

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        return _splits_bar(r, params)


class VPTraitNWBar(VPTraitPlanar):
    """
    A flow along a bar whose speed is given at distances along it (as
    DysmalPy's VariableXBarFlow): as VPTraitBar, with the speed v(|x|) at
    the distance |x| from the centre along the axis, of a value at each
    node (at the distances of the radii of the nodes); 0 within the first
    or beyond the last (e.g. a bar of a length: speeds of 0 beyond it).
    Its parameters: p (degrees) and s (arcsec); v, node-wise. Fit s with
    a fitter that takes no slopes (see VPTraitBar).

    Parameters
    ----------
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_bar'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_BAR

    def __init__(
            self,
            sampling: str = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        super().__init__(Sampling(sampling), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            p=ParamScalarDesc(),
            s=ParamScalarDesc())

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(v=ParamVectorDesc(nnodes))

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        return _splits_bar(r, params)


class VPTraitSpiral(VPTraitPlanar):
    """
    The velocities of a spiral density wave in the gas of a disk, as
    DysmalPy's SpiralDensityWave (see Spiral): the radial
    part v_R = -e m (Omega - o) / k cos(phi) and the tangential part v_t =
    e kappa^2 / (2 k Omega) sin(phi), 0 where there is no wave, where k is
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
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'spiral'

    @staticmethod
    def uid() -> int:
        return VPT_UID_SPIRAL

    def __init__(
            self,
            order: int,
            winding: str | None = None,
            wavenumber: str = 'dispersion',
            name: str | None = None
    ):
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
    The rotation curve of the mass of its galaxy (see gbkfit.model.mass):
    the circular velocity at the rings it is evaluated on, which its
    galaxy computes from the parameters of its distance and its mass at
    each evaluation, and with pressure support, the
    rotation of the gas of its disk in that gravity; 0 where its square is
    negative (the mass pulls outwards, e.g. inside a ring, or the pressure
    exceeds the gravity: no rotation, as DysmalPy takes it), NaN where it
    is infinite (there is no circular orbit, e.g. at a jump of the surface
    density of a thin disk). A tangential speed (it has no option
    direction). It has no parameters of its own, and its galaxy must have
    mass.

    With pressure_support 'jeans', the rotation of an isotropic gas at the
    midplane of its disk (the radial Jeans equation): v^2 = v_c^2 + (R /
    rho) d(rho sigma^2)/dR, with the azimuthal means of the density rho of
    the disk at its midplane (its density traits times its density
    height traits at z = 0, or its density traits if thin: the equation
    of a thin disk) and of rho sigma^2, sigma its velocity dispersion at
    the midplane (with the dispersion of the vertical equilibrium of its
    disk in quadrature, if any, see parts.Equilibrium; azimuthal means as
    those of mass.SurfaceComponent); the derivative by the second-order
    three-point formula on the rings; NaN without density. With
    'self_gravitating', the same for a disk whose thickness its own
    gravity sets, h ~ sigma^2 / S (S its surface density, its density
    traits): rho ~ S^2 / sigma^2 and rho sigma^2 ~ S^2, so that v^2 =
    v_c^2 + 2 sigma^2 dln S / dln R for an axisymmetric disk (Burkert et
    al. 2010), whatever its density height traits (DysmalPy's pressure
    support of types 1 and 2, for an exponential and a Sersic S). A disk
    whose dispersion is not isotropic (see the package docstring) takes
    the Jeans equation of its velocity ellipsoid: v^2 = v_c^2 + (R / rho)
    d(rho sigma_R^2)/dR + sigma_R^2 - sigma_phi^2, with sigma_z in rho of
    'self_gravitating' (see _profiles.pressure_support).

    With pressure_density, the names of mass components of its galaxy
    (one, or several: e.g. its atomic and its molecular gas), the
    pressure takes the sum of their densities instead of that of its
    disk: at their midplane ('jeans'; e.g. the gas of thick mass disks,
    as DysmalPy's pressure support of type 3, which adds its gas
    components), or their surface densities ('self_gravitating'; e.g.
    that of the mass disk, as DysmalPy's types 1 and 2 take its
    effective radius), with the dispersion of its disk.

    The rotation is the same at every height, or with the velocity height
    trait 'mass', that of the gravity at each height (see VHTraitMass).

    Parameters
    ----------
    pressure_support : str, optional
        The pressure support: none (the default), 'jeans' or
        'self_gravitating'.
    pressure_density : str or Sequence of str, optional
        The name of the mass component whose density the pressure takes,
        or the names of several, whose densities add up; by default, that
        of its disk.
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If pressure_support or pressure_density is not valid (names that
        repeat, or none), or pressure_density is given without
        pressure_support.
    """

    @staticmethod
    def type() -> str:
        return 'mass'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_UNIFORM

    def dump(self) -> dict[str, Any]:
        info = {}
        if (support := self._pressure_support) is not None:
            info['pressure_support'] = support.mode
            if (names := support.density) is not None:
                info['pressure_density'] = \
                    names[0] if len(names) == 1 else list(names)
        return super().dump() | info

    def __init__(
            self,
            pressure_support: str | None = None,
            pressure_density: str | Sequence[str] | None = None,
            name: str | None = None
    ):
        super().__init__(name=name)
        if pressure_support not in (None, 'jeans', 'self_gravitating'):
            raise ConfigError(
                f"pressure_support must be 'jeans', 'self_gravitating' or "
                f"none; it is {pressure_support!r}")
        if pressure_density is not None:
            if pressure_support is None:
                raise ConfigError(
                    "pressure_density is the density of the pressure "
                    "support: it needs pressure_support")
            pressure_density = (pressure_density,) \
                if isinstance(pressure_density, str) \
                else tuple(pressure_density)
            if not pressure_density \
                    or len(set(pressure_density)) != len(pressure_density):
                raise ConfigError(
                    f"pressure_density names mass components, at least one "
                    f"and each once; it is {list(pressure_density)}")
            for name in pressure_density:
                parseutils.check_name(name)
        self._pressure_support = None if pressure_support is None \
            else PressureSupport(pressure_support, pressure_density)

    def direction(self) -> str:
        return 'tan'

    def sampling(self) -> str:
        return 'subrings'

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(vt=ParamVectorDesc(nnodes))

    def from_galaxy(self) -> RotationInput:
        return RotationInput(('vt',), self._pressure_support)
