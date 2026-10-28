"""
The height traits: of the density (rht), of the velocity (vht) and of the
velocity dispersion (dht).
"""

import abc
from typing import Annotated, Literal

import numpy as np
import scipy.ndimage
import scipy.special
from annotated_types import Gt
from pydantic import FiniteFloat

from gbkfit.params.pdescs import ParamDesc, ParamScalarDesc, ParamVectorDesc
from gbkfit.utils.parseutils import check_options
from gbkfit.utils.typeutils import Unit
from .base import DHTrait, RHTrait, RotationFactorsInput, VHTrait
from .options import Nodes, SAMPLINGS, SAMPLING_DEFAULT, Sampling, Zmax


__all__ = [
    'DHTraitExponential',
    'DHTraitGauss',
    'DHTraitLinear',
    'DHTraitOne',
    'DHTraitP1',
    'RHTraitExponential',
    'RHTraitGGauss',
    'RHTraitGauss',
    'RHTraitLorentz',
    'RHTraitMoffat',
    'RHTraitP1',
    'RHTraitP2',
    'RHTraitSech2',
    'RHTraitUniform',
    'VHTraitExponential',
    'VHTraitGauss',
    'VHTraitLinear',
    'VHTraitMass',
    'VHTraitOne',
    'VHTraitP1'
]


# Their uids in the native kernels
RHT_UID_UNIFORM = 1
RHT_UID_EXP = 2
RHT_UID_GAUSS = 3
RHT_UID_GGAUSS = 4
RHT_UID_LORENTZ = 5
RHT_UID_MOFFAT = 6
RHT_UID_SECH2 = 7

VHT_UID_ONE = 1
VHT_UID_LINEAR = 2
VHT_UID_EXP = 3
VHT_UID_GAUSS = 4
VHT_UID_MASS = 5

DHT_UID_ONE = 1
DHT_UID_LINEAR = 2
DHT_UID_EXP = 3
DHT_UID_GAUSS = 4

# The heights of the table of the velocity height trait 'mass' (see
# VHTraitMass): this many, from 0 to its zmax
VHT_MASS_NHEIGHTS = 33


class RHTraitP1(RHTrait, abc.ABC):
    """
    A density height trait of one parameter, s.
    """

    def __init__(
            self,
            zmax: Annotated[FiniteFloat, Gt(0), Unit('arcsec')],
            rnodes: bool = False,
            sampling: Literal[SAMPLINGS] = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        check_options(RHTraitP1.__init__, locals())
        super().__init__(
            Zmax(zmax), Nodes(rnodes), Sampling(sampling), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {} if self.rnodes() else dict(
            s=ParamScalarDesc(unit='arcsec'))

    def params_rnw(self, nrnodes: int) -> dict[str, ParamDesc]:
        return {} if not self.rnodes() else dict(
            s=ParamVectorDesc(nrnodes, unit='arcsec'))


class RHTraitP2(RHTrait, abc.ABC):
    """
    A density height trait of two parameters, s and b.
    """

    def __init__(
            self,
            zmax: Annotated[FiniteFloat, Gt(0), Unit('arcsec')],
            rnodes: bool = False,
            sampling: Literal[SAMPLINGS] = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        check_options(RHTraitP2.__init__, locals())
        super().__init__(
            Zmax(zmax), Nodes(rnodes), Sampling(sampling), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {} if self.rnodes() else dict(
            s=ParamScalarDesc(unit='arcsec'),
            b=ParamScalarDesc(unit=''))

    def params_rnw(self, nrnodes: int) -> dict[str, ParamDesc]:
        return {} if not self.rnodes() else dict(
            s=ParamVectorDesc(nrnodes, unit='arcsec'),
            b=ParamVectorDesc(nrnodes, unit=''))


class RHTraitUniform(RHTraitP1):
    """
    A uniform vertical distribution of the density: 1 / (2 s) for |z|
    <= s, 0 beyond. Its parameter: s, the half thickness. It integrates to
    1. Its use is discouraged: its sharp edges alias on the voxels (a
    warning says so).

    Parameters
    ----------
    zmax : float
        The height above the midplane (arcsec) within which the density
        lies (see RHTrait).
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
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
        return RHT_UID_UNIFORM

    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        s, = params
        return np.clip((z + s) / (2 * s), 0, 1)


class RHTraitExponential(RHTraitP1):
    """
    An exponential vertical distribution of the density: exp(-|z| / s)
    / (2 s). Its parameter: s, the scale height. It integrates to 1.

    Parameters
    ----------
    zmax : float
        The height above the midplane (arcsec) within which the density
        lies (see RHTrait).
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'exponential'

    @staticmethod
    def uid() -> int:
        return RHT_UID_EXP

    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        s, = params
        return 0.5 + 0.5 * np.sign(z) * -np.expm1(-np.abs(z) / s)


class RHTraitGauss(RHTraitP1):
    """
    A Gaussian vertical distribution of the density: exp(-z^2 / (2
    s^2)) / (s sqrt(2 pi)). Its parameter: s, the dispersion of the
    Gaussian. It integrates to 1.

    Parameters
    ----------
    zmax : float
        The height above the midplane (arcsec) within which the density
        lies (see RHTrait).
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def uid() -> int:
        return RHT_UID_GAUSS

    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        s, = params
        return 0.5 * (1 + scipy.special.erf(z / (s * np.sqrt(2))))


class RHTraitGGauss(RHTraitP2):
    """
    A generalised Gaussian vertical distribution of the density: b / (2
    s Gamma(1 / b)) exp(-(|z| / s)^b). Its parameters: s, the scale
    height, and b, the shape. It integrates to 1.

    Parameters
    ----------
    zmax : float
        The height above the midplane (arcsec) within which the density
        lies (see RHTrait).
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'ggauss'

    @staticmethod
    def uid() -> int:
        return RHT_UID_GGAUSS

    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        s, b = params
        p = scipy.special.gammainc(1 / b, (np.abs(z) / s) ** b)
        return 0.5 + 0.5 * np.sign(z) * p


class RHTraitLorentz(RHTraitP1):
    """
    A Lorentzian vertical distribution of the density: s / (pi (z^2 +
    s^2)). Its parameter: s, the half width at half maximum; its tails are
    heavy (zmax cuts them). It integrates to 1.

    Parameters
    ----------
    zmax : float
        The height above the midplane (arcsec) within which the density
        lies (see RHTrait).
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'lorentz'

    @staticmethod
    def uid() -> int:
        return RHT_UID_LORENTZ

    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        s, = params
        return 0.5 + np.arctan(z / s) / np.pi


class RHTraitMoffat(RHTraitP2):
    """
    A Moffat vertical distribution of the density: Gamma(b) / (Gamma(b
    - 1/2) s sqrt(pi)) (1 + (z / s)^2)^-b, for b > 1/2. Its parameters: s,
    the core height, and b, the power of the fall. It integrates to 1.

    Parameters
    ----------
    zmax : float
        The height above the midplane (arcsec) within which the density
        lies (see RHTrait).
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'moffat'

    @staticmethod
    def uid() -> int:
        return RHT_UID_MOFFAT

    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        # (the share beyond |z|, of the incomplete beta function)
        s, b = params
        u2 = (z / s) ** 2
        beyond = scipy.special.betainc(b - 0.5, 0.5, 1 / (1 + u2))
        return np.where(z < 0, beyond / 2, 1 - beyond / 2)


class RHTraitSech2(RHTraitP1):
    """
    A sech^2 vertical distribution of the density: sech^2(z / s) / (2
    s). Its parameter: s, the scale height. It integrates to 1.

    Parameters
    ----------
    zmax : float
        The height above the midplane (arcsec) within which the density
        lies (see RHTrait).
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'sech2'

    @staticmethod
    def uid() -> int:
        return RHT_UID_SECH2

    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        s, = params
        return 0.5 * (1 + np.tanh(z / s))


class VHTraitOne(VHTrait):
    """
    No change of the velocity with the height: the factor 1 (the default).

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters (see Trait).
    """

    @staticmethod
    def type() -> str:
        return 'one'

    @staticmethod
    def uid() -> int:
        return VHT_UID_ONE

    def __init__(self, name: str | None = None):
        check_options(VHTraitOne.__init__, locals())
        super().__init__(name=name)


class VHTraitMass(VHTrait):
    """
    The rotation in the gravity of the galaxy at each height: the factor
    v(R, z) / v(R, 0) of its velocity polar trait 'mass' (see
    VPTraitMass), with which it pairs, of the rotation of a circular orbit
    at the height z in the radial force of the mass of the galaxy there,
    v(R, z)^2 = -R F_R(R, z) (see gravity.Gravity), plus the pressure
    support of the trait 'mass' at z, if any: 'jeans', with the density of
    its disk at z (its density traits times its density height traits) or
    that of the mass components of pressure_density (their sum), and its
    dispersion at z (with its dispersion height traits);
    'self_gravitating', in the isothermal sheet of the surface density S
    of the mass components of pressure_density (required: the disk has
    no mass) and the dispersion
    sigma at the midplane, of the density ~ sech^2(z / z0), z0 = sigma^2 /
    (pi G S). v(R, z) is 0 where its square is negative, and NaN where it
    is infinite, as at the midplane; the factor is 1 where v(R, 0) is 0
    (no rotation at any height). Its galaxy computes it at each
    evaluation, at the subnodes of the disk and at VHT_MASS_NHEIGHTS
    heights from 0 to zmax, and its disk interpolates it linearly between
    them; above zmax, it is the factor at zmax. The rings of the disk are
    taken flat (coplanar and concentric), as for the rotation at its
    midplane. It has no parameters of its own.

    Parameters
    ----------
    zmax : float
        The highest height (arcsec) of its table: that of the density
        height traits of its disk.
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If zmax is not a positive length.
    """

    @staticmethod
    def type() -> str:
        return 'mass'

    @staticmethod
    def uid() -> int:
        return VHT_UID_MASS

    def __init__(
            self,
            zmax: Annotated[FiniteFloat, Gt(0), Unit('arcsec')],
            name: str | None = None
    ):
        check_options(VHTraitMass.__init__, locals())
        super().__init__(Zmax(zmax), name=name)

    def zmax(self) -> float:
        """Return its zmax (arcsec): the top of its table."""
        return self._option(Zmax).value

    def rnodes(self) -> bool:
        # (node-wise always: its table at each subnode)
        return True

    def sampling(self) -> str:
        return 'subrings'

    def consts(self) -> tuple[float, ...]:
        return (self.zmax(), VHT_MASS_NHEIGHTS)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return {
            name: ParamVectorDesc(nnodes, unit='')
            for name in self.from_galaxy().params}

    def from_galaxy(self) -> RotationFactorsInput:
        heights = np.linspace(0, self.zmax(), VHT_MASS_NHEIGHTS)
        return RotationFactorsInput(
            tuple(f'f{k}' for k in range(len(heights))),
            tuple(heights.tolist()))


class VHTraitP1(VHTrait, abc.ABC):
    """
    A velocity height trait: a factor of the height, of one parameter
    (param_name), node-wise if rnodes.
    """

    def __init__(
            self,
            rnodes: bool = False,
            sampling: Literal[SAMPLINGS] = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        check_options(VHTraitP1.__init__, locals())
        super().__init__(Nodes(rnodes), Sampling(sampling), name=name)

    @staticmethod
    @abc.abstractmethod
    def param_name() -> str:
        """Return the name of its parameter."""
        pass

    def params_sm(self) -> dict[str, ParamDesc]:
        return {} if self.rnodes() else {
            self.param_name(): ParamScalarDesc(unit='arcsec')}

    def params_rnw(self, nrnodes: int) -> dict[str, ParamDesc]:
        return {} if not self.rnodes() else {
            self.param_name(): ParamVectorDesc(nrnodes, unit='arcsec')}


class VHTraitLinear(VHTraitP1):
    """
    The factor max(0, |z| - z0), which grows linearly above the height z0:
    with a polar trait of the change per unit height (e.g. a lag), and
    another of the velocity in the plane. Its parameter: z0.

    Parameters
    ----------
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'linear'

    @staticmethod
    def uid() -> int:
        return VHT_UID_LINEAR

    @staticmethod
    def param_name() -> str:
        return 'z0'


class VHTraitExponential(VHTraitP1):
    """
    The factor exp(-|z| / s), which falls with the scale height s. Its
    parameter: s.

    Parameters
    ----------
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'exponential'

    @staticmethod
    def uid() -> int:
        return VHT_UID_EXP

    @staticmethod
    def param_name() -> str:
        return 's'


class VHTraitGauss(VHTraitP1):
    """
    The factor exp(-z^2 / (2 s^2)), which falls with the scale height s.
    Its parameter: s.

    Parameters
    ----------
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def uid() -> int:
        return VHT_UID_GAUSS

    @staticmethod
    def param_name() -> str:
        return 's'


class DHTraitOne(DHTrait):
    """
    No change of the dispersion with the height: the factor 1 (the
    default).

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters (see Trait).
    """

    @staticmethod
    def type() -> str:
        return 'one'

    @staticmethod
    def uid() -> int:
        return DHT_UID_ONE

    def __init__(self, name: str | None = None):
        check_options(DHTraitOne.__init__, locals())
        super().__init__(name=name)


class DHTraitP1(DHTrait, abc.ABC):
    """
    A dispersion height trait: a factor of the height, of one parameter
    (param_name), node-wise if rnodes.
    """

    def __init__(
            self,
            rnodes: bool = False,
            sampling: Literal[SAMPLINGS] = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        check_options(DHTraitP1.__init__, locals())
        super().__init__(Nodes(rnodes), Sampling(sampling), name=name)

    @staticmethod
    @abc.abstractmethod
    def param_name() -> str:
        """Return the name of its parameter."""
        pass

    def params_sm(self) -> dict[str, ParamDesc]:
        return {} if self.rnodes() else {
            self.param_name(): ParamScalarDesc(unit='arcsec')}

    def params_rnw(self, nrnodes: int) -> dict[str, ParamDesc]:
        return {} if not self.rnodes() else {
            self.param_name(): ParamVectorDesc(nrnodes, unit='arcsec')}


class DHTraitLinear(DHTraitP1):
    """
    The factor max(0, |z| - z0), which grows linearly above the height z0:
    with a polar trait of the change per unit height (e.g. a lag), and
    another of the dispersion in the plane. Its parameter: z0.

    Parameters
    ----------
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'linear'

    @staticmethod
    def uid() -> int:
        return DHT_UID_LINEAR

    @staticmethod
    def param_name() -> str:
        return 'z0'


class DHTraitExponential(DHTraitP1):
    """
    The factor exp(-|z| / s), which falls with the scale height s. Its
    parameter: s.

    Parameters
    ----------
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'exponential'

    @staticmethod
    def uid() -> int:
        return DHT_UID_EXP

    @staticmethod
    def param_name() -> str:
        return 's'


class DHTraitGauss(DHTraitP1):
    """
    The factor exp(-z^2 / (2 s^2)), which falls with the scale height s.
    Its parameter: s.

    Parameters
    ----------
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def uid() -> int:
        return DHT_UID_GAUSS

    @staticmethod
    def param_name() -> str:
        return 's'
