"""
The height traits: of the density (rht), of the velocity (vht) and of the
velocity dispersion (dht).
"""

import abc
from typing import Annotated

import numpy as np
import scipy.ndimage
import scipy.special
from annotated_types import Gt
from pydantic import FiniteFloat

from gbkfit.config import Unit, check_options
from gbkfit.params.pdescs import ParamDesc, ParamScalarDesc, ParamVectorDesc
from .base import DHTrait, RHTrait, RotationFactorsInput, VHTrait
from .shapes import (
    ExponentialFactor, GaussFactor, LinearFactor, OneFactor)
from .options import (
    SAMPLING_DEFAULT, Nodewise, Sampling, SamplingChoice, Zmax)


__all__ = [
    'DHTraitExponential',
    'DHTraitGauss',
    'DHTraitLinear',
    'DHTraitOne',
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
    'VHTraitOne'
]


# Their uids in the native kernels
RHT_UID_UNIFORM = 1
RHT_UID_EXP = 2
RHT_UID_GAUSS = 3
RHT_UID_GGAUSS = 4
RHT_UID_LORENTZ = 5
RHT_UID_MOFFAT = 6
RHT_UID_SECH2 = 7

# The uid of the velocity height trait 'mass' (of the velocity alone; the
# factors that the velocity and the dispersion share are in shapes)
VHT_UID_MASS = 5


# The heights of the table of the velocity height trait 'mass' (see
# VHTraitMass): this many, from 0 to its zmax
VHT_MASS_NHEIGHTS = 33

# What the parameters of the density height traits are: s, the height
# that scales each profile (e.g. its scale height, or the half thickness
# of 'uniform'), and b, the power of 'ggauss' and 'moffat'
_SCALE_HEIGHT = "the height that sets the thickness of the profile"
_SHAPE = "the power of the profile, which sets its shape"

_P1_PARAMS = dict(
    z0=('Onset height', "the height above which the factor grows"),
    s=('Scale height', "the height over which the factor falls"))


class RHTraitP1(RHTrait, abc.ABC):
    """
    A density height trait of one parameter, s.
    """

    def __init__(
            self,
            zmax: Annotated[FiniteFloat, Gt(0), Unit('arcsec')],
            nodewise: bool = False,
            sampling: SamplingChoice = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        check_options(RHTraitP1.__init__, locals())
        super().__init__(
            Zmax(zmax), Nodewise(nodewise), Sampling(sampling), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {} if self.nodewise() else dict(
            s=ParamScalarDesc(
                title='Scale height', desc=_SCALE_HEIGHT, min=0,
                unit='arcsec', default=0.2))

    def params_rnw(self, nrnodes: int) -> dict[str, ParamDesc]:
        return {} if not self.nodewise() else dict(
            s=ParamVectorDesc(
                nrnodes, title='Scale height', desc=_SCALE_HEIGHT, min=0,
                unit='arcsec', default=0.2))


class RHTraitP2(RHTrait, abc.ABC):
    """
    A density height trait of two parameters, s and b.
    """

    # The value above which b is valid: the profile of 'ggauss' is defined
    # for b > 0 (see its subclasses)
    _B_MIN = 0.0

    def __init__(
            self,
            zmax: Annotated[FiniteFloat, Gt(0), Unit('arcsec')],
            nodewise: bool = False,
            sampling: SamplingChoice = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        check_options(RHTraitP2.__init__, locals())
        super().__init__(
            Zmax(zmax), Nodewise(nodewise), Sampling(sampling), name=name)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {} if self.nodewise() else dict(
            s=ParamScalarDesc(
                title='Scale height', desc=_SCALE_HEIGHT, min=0,
                unit='arcsec', default=0.2),
            b=ParamScalarDesc(
                title='Shape', desc=_SHAPE, min=self._B_MIN,
                min_exclusive=True, unit='', default=1))

    def params_rnw(self, nrnodes: int) -> dict[str, ParamDesc]:
        return {} if not self.nodewise() else dict(
            s=ParamVectorDesc(
                nrnodes, title='Scale height', desc=_SCALE_HEIGHT, min=0,
                unit='arcsec', default=0.2),
            b=ParamVectorDesc(
                nrnodes, title='Shape', desc=_SHAPE, min=self._B_MIN,
                min_exclusive=True, unit='', default=1))


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
    nodewise : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
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
    nodewise : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'exponential'

    @staticmethod
    def type_title() -> str:
        return 'Exponential'

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
    nodewise : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def type_title() -> str:
        return 'Gaussian'

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
    nodewise : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'ggauss'

    @staticmethod
    def type_title() -> str:
        return 'Generalised Gaussian'

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
    nodewise : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'lorentz'

    @staticmethod
    def type_title() -> str:
        return 'Lorentzian'

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
    nodewise : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    # (its normalisation, Gamma(b - 1/2), needs b > 1/2)
    _B_MIN = 0.5

    @staticmethod
    def type() -> str:
        return 'moffat'

    @staticmethod
    def type_title() -> str:
        return 'Moffat'

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
    nodewise : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'sech2'

    @staticmethod
    def type_title() -> str:
        return 'Sech-squared'

    @staticmethod
    def uid() -> int:
        return RHT_UID_SECH2

    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        s, = params
        return 0.5 * (1 + np.tanh(z / s))


class VHTraitOne(OneFactor, VHTrait):
    """
    No change of the velocity with the height: the factor 1 (the default).

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters (see Trait).
    """

    def __init__(self, name: str | None = None):
        check_options(VHTraitOne.__init__, locals())
        super().__init__(name=name)


class VHTraitMass(VHTrait):
    """
    The rotation in the gravity of the galaxy at each height, for the
    velocity polar trait 'mass', with which it pairs. It is the factor
    v(R, z) / v(R, 0) of that trait (see VPTraitMass), of the rotation of
    a circular orbit at the height z in the radial force of the mass of
    the galaxy there, v(R, z)^2 = -R F_R(R, z) (see gravity.Gravity), plus
    the pressure support of the trait 'mass' at z, if any: 'jeans', with
    the density of its disk at z (its density traits times its density
    height traits) or that of the mass components of the mass of the
    pressure support (their sum), and its dispersion at z (with its
    dispersion height traits); 'self_gravitating', in the isothermal
    sheet of the surface density S of the mass components of its mass
    (required: the disk has no mass) and the dispersion sigma at the
    midplane, of the density ~ sech^2(z / z0), z0 = sigma^2 / (pi G S).
    v(R, z) is 0 where its square is negative, and NaN where it
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
    ValueError
        If zmax is not a positive length.
    """

    @staticmethod
    def type() -> str:
        return 'mass'

    @staticmethod
    def type_title() -> str:
        return 'From the gravity'

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

    def nodewise(self) -> bool:
        # (node-wise always: its table at each subnode)
        return True

    def sampling(self) -> str:
        return 'subrings'

    def consts(self) -> tuple[float, ...]:
        return (self.zmax(), VHT_MASS_NHEIGHTS)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        table = self.from_galaxy()
        return {
            name: ParamVectorDesc(
                nnodes, title=f'Rotation factor at {z:.3g} arcsec',
                desc=f"the factor of the rotation at the height {z:.3g} "
                     f"arcsec, which the galaxy computes",
                unit='', default=1)
            for name, z in zip(table.params, table.heights)}

    def from_galaxy(self) -> RotationFactorsInput:
        heights = np.linspace(0, self.zmax(), VHT_MASS_NHEIGHTS)
        return RotationFactorsInput(
            tuple(f'f{k}' for k in range(len(heights))),
            tuple(heights.tolist()))


class VHTraitLinear(LinearFactor, VHTrait):
    """
    A change of the velocity that grows linearly above a height (e.g. a
    lag): the factor max(0, |z| - z0). It goes with a polar trait of the
    change per unit height, and another of the velocity in the plane. Its
    parameter: z0.

    Parameters
    ----------
    nodewise : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """


class VHTraitExponential(ExponentialFactor, VHTrait):
    """
    The factor exp(-|z| / s), which falls with the scale height s. Its
    parameter: s.

    Parameters
    ----------
    nodewise : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """


class VHTraitGauss(GaussFactor, VHTrait):
    """
    The factor exp(-z^2 / (2 s^2)), which falls with the scale height s.
    Its parameter: s.

    Parameters
    ----------
    nodewise : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """


class DHTraitOne(OneFactor, DHTrait):
    """
    No change of the dispersion with the height: the factor 1 (the
    default).

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters (see Trait).
    """

    def __init__(self, name: str | None = None):
        check_options(DHTraitOne.__init__, locals())
        super().__init__(name=name)


class DHTraitLinear(LinearFactor, DHTrait):
    """
    A change of the dispersion that grows linearly above a height: the
    factor max(0, |z| - z0). It goes with a polar trait of the change per
    unit height, and another of the dispersion in the plane. Its
    parameter: z0.

    Parameters
    ----------
    nodewise : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """


class DHTraitExponential(ExponentialFactor, DHTrait):
    """
    The factor exp(-|z| / s), which falls with the scale height s. Its
    parameter: s.

    Parameters
    ----------
    nodewise : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """


class DHTraitGauss(GaussFactor, DHTrait):
    """
    The factor exp(-z^2 / (2 s^2)), which falls with the scale height s.
    Its parameter: s.

    Parameters
    ----------
    nodewise : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """
