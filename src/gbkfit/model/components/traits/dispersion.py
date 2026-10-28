"""
The velocity dispersion polar traits (dpt): isotropic, or along one
direction of a disk (see DPTrait).
"""

from collections.abc import Sequence
from typing import Annotated, Literal

import numpy as np
from annotated_types import Ge

from gbkfit.params.pdescs import ParamDesc, ParamScalarDesc, ParamVectorDesc
from gbkfit.utils.parseutils import check_options
from .base import DPTrait, params_nw_distortion, splits_nw_distortion
from .options import (
    Blobs, DPT_DIRECTIONS, DPT_DIRECTION_DEFAULT, DispersionDirection,
    Harmonic, SAMPLINGS, SAMPLING_DEFAULT, Sampling)


__all__ = [
    'DPTraitExponential',
    'DPTraitGGauss',
    'DPTraitGauss',
    'DPTraitLorentz',
    'DPTraitMixtureExponential',
    'DPTraitMixtureGGauss',
    'DPTraitMixtureGauss',
    'DPTraitMixtureMoffat',
    'DPTraitMoffat',
    'DPTraitNWDistortion',
    'DPTraitNWHarmonic',
    'DPTraitNWUniform',
    'DPTraitSech2',
    'DPTraitUniform'
]


# Their uids in the native kernels
DPT_UID_UNIFORM = 1
DPT_UID_EXP = 2
DPT_UID_GAUSS = 3
DPT_UID_GGAUSS = 4
DPT_UID_LORENTZ = 5
DPT_UID_MOFFAT = 6
DPT_UID_SECH2 = 7
DPT_UID_MIXTURE_EXP = 51
DPT_UID_MIXTURE_GAUSS = 52
DPT_UID_MIXTURE_GGAUSS = 53
DPT_UID_MIXTURE_MOFFAT = 54
DPT_UID_NW_UNIFORM = 101
DPT_UID_NW_HARMONIC = 102
DPT_UID_NW_DISTORTION = 103


class DPTraitUniform(DPTrait):
    """
    A uniform velocity dispersion: a. Its parameter: a, the dispersion
    (km/s).

    Parameters
    ----------
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), 'rad', 'tan'
        or 'ver' (see the package docstring).
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
        return DPT_UID_UNIFORM

    def __init__(
            self,
            direction: Literal[tuple(DPT_DIRECTIONS)] = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitUniform.__init__, locals())
        super().__init__(DispersionDirection(direction), name=name)

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(a=ParamScalarDesc(unit='km/s'))


class DPTraitExponential(DPTrait):
    """
    An exponential velocity dispersion: a exp(-r / s). Its parameters: a,
    the dispersion (km/s) at the centre, and s, the scale length.

    Parameters
    ----------
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), 'rad', 'tan'
        or 'ver' (see the package docstring).
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
        return DPT_UID_EXP

    def __init__(
            self,
            direction: Literal[tuple(DPT_DIRECTIONS)] = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitExponential.__init__, locals())
        super().__init__(DispersionDirection(direction), name=name)

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(unit='km/s'),
            s=ParamScalarDesc(unit='arcsec'))


class DPTraitGauss(DPTrait):
    """
    A Gaussian velocity dispersion: a exp(-r^2 / (2 s^2)). Its parameters:
    a, the dispersion (km/s) at the centre, and s, the dispersion of the
    Gaussian.

    Parameters
    ----------
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), 'rad', 'tan'
        or 'ver' (see the package docstring).
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
        return DPT_UID_GAUSS

    def __init__(
            self,
            direction: Literal[tuple(DPT_DIRECTIONS)] = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitGauss.__init__, locals())
        super().__init__(DispersionDirection(direction), name=name)

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(unit='km/s'),
            s=ParamScalarDesc(unit='arcsec'))


class DPTraitGGauss(DPTrait):
    """
    A generalised Gaussian velocity dispersion: a exp(-(r / s)^b). Its
    parameters: a, the dispersion (km/s) at the centre, s, the scale
    length, and b, the shape (1 is the exponential, and 2 a Gaussian of
    dispersion s / sqrt(2)).

    Parameters
    ----------
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), 'rad', 'tan'
        or 'ver' (see the package docstring).
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
        return DPT_UID_GGAUSS

    def __init__(
            self,
            direction: Literal[tuple(DPT_DIRECTIONS)] = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitGGauss.__init__, locals())
        super().__init__(DispersionDirection(direction), name=name)

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(unit='km/s'),
            s=ParamScalarDesc(unit='arcsec'),
            b=ParamScalarDesc(unit=''))


class DPTraitLorentz(DPTrait):
    """
    A Lorentzian velocity dispersion: a s^2 / (r^2 + s^2). Its parameters:
    a, the dispersion (km/s) at the centre, and s, the half width at half
    maximum.

    Parameters
    ----------
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), 'rad', 'tan'
        or 'ver' (see the package docstring).
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
        return DPT_UID_LORENTZ

    def __init__(
            self,
            direction: Literal[tuple(DPT_DIRECTIONS)] = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitLorentz.__init__, locals())
        super().__init__(DispersionDirection(direction), name=name)

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(unit='km/s'),
            s=ParamScalarDesc(unit='arcsec'))


class DPTraitMoffat(DPTrait):
    """
    A Moffat velocity dispersion: a (1 + (r / s)^2)^-b. Its parameters: a,
    the dispersion (km/s) at the centre, s, the core radius, and b, the
    power of the fall.

    Parameters
    ----------
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), 'rad', 'tan'
        or 'ver' (see the package docstring).
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
        return DPT_UID_MOFFAT

    def __init__(
            self,
            direction: Literal[tuple(DPT_DIRECTIONS)] = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitMoffat.__init__, locals())
        super().__init__(DispersionDirection(direction), name=name)

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(unit='km/s'),
            s=ParamScalarDesc(unit='arcsec'),
            b=ParamScalarDesc(unit=''))


class DPTraitSech2(DPTrait):
    """
    A sech^2 velocity dispersion: a sech^2(r / s). Its parameters: a, the
    dispersion (km/s) at the centre, and s, the scale length.

    Parameters
    ----------
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), 'rad', 'tan'
        or 'ver' (see the package docstring).
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
        return DPT_UID_SECH2

    def __init__(
            self,
            direction: Literal[tuple(DPT_DIRECTIONS)] = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitSech2.__init__, locals())
        super().__init__(DispersionDirection(direction), name=name)

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(unit='km/s'),
            s=ParamScalarDesc(unit='arcsec'))


class DPTraitMixtureExponential(DPTrait):
    """
    A velocity dispersion made of nblobs elliptical blobs (see the module
    docstring), each an exponential of the elliptical distance rho from
    its centre: a exp(-rho / s). Its parameters, vectors of a value for
    each blob: r and t, the radius and the azimuth of its centre, a, its
    amplitude, s, its scale along its major axis, q, its axis ratio (minor
    over major), and p, the angle of its major axis.

    Parameters
    ----------
    nblobs : int
        The number of blobs.
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), 'rad', 'tan'
        or 'ver' (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'mixture_exponential'

    @staticmethod
    def uid() -> int:
        return DPT_UID_MIXTURE_EXP

    def __init__(
            self,
            nblobs: Annotated[int, Ge(1)],
            direction: Literal[tuple(DPT_DIRECTIONS)] = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitMixtureExponential.__init__, locals())
        super().__init__(
            Blobs(nblobs), DispersionDirection(direction), name=name)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return self._option(Blobs).params(shape=False, unit='km/s')


class DPTraitMixtureGauss(DPTrait):
    """
    A velocity dispersion made of nblobs elliptical blobs (see the module
    docstring), each a Gaussian of the elliptical distance rho from its
    centre: a exp(-rho^2 / (2 s^2)). Its parameters, vectors of a value
    for each blob: r and t, the radius and the azimuth of its centre, a,
    its amplitude, s, its scale along its major axis, q, its axis ratio
    (minor over major), and p, the angle of its major axis.

    Parameters
    ----------
    nblobs : int
        The number of blobs.
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), 'rad', 'tan'
        or 'ver' (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'mixture_gauss'

    @staticmethod
    def uid() -> int:
        return DPT_UID_MIXTURE_GAUSS

    def __init__(
            self,
            nblobs: Annotated[int, Ge(1)],
            direction: Literal[tuple(DPT_DIRECTIONS)] = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitMixtureGauss.__init__, locals())
        super().__init__(
            Blobs(nblobs), DispersionDirection(direction), name=name)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return self._option(Blobs).params(shape=False, unit='km/s')


class DPTraitMixtureGGauss(DPTrait):
    """
    A velocity dispersion made of nblobs elliptical blobs (see the module
    docstring), each a generalised Gaussian of the elliptical distance rho
    from its centre: a exp(-(rho / s)^b). Its parameters, vectors of a
    value for each blob: r and t, the radius and the azimuth of its
    centre, a, its amplitude, s, its scale along its major axis, b, its
    shape, q, its axis ratio (minor over major), and p, the angle of its
    major axis.

    Parameters
    ----------
    nblobs : int
        The number of blobs.
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), 'rad', 'tan'
        or 'ver' (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'mixture_ggauss'

    @staticmethod
    def uid() -> int:
        return DPT_UID_MIXTURE_GGAUSS

    def __init__(
            self,
            nblobs: Annotated[int, Ge(1)],
            direction: Literal[tuple(DPT_DIRECTIONS)] = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitMixtureGGauss.__init__, locals())
        super().__init__(
            Blobs(nblobs), DispersionDirection(direction), name=name)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return self._option(Blobs).params(shape=True, unit='km/s')


class DPTraitMixtureMoffat(DPTrait):
    """
    A velocity dispersion made of nblobs elliptical blobs (see the module
    docstring), each a Moffat of the elliptical distance rho from its
    centre: a (1 + (rho / s)^2)^-b. Its parameters, vectors of a value for
    each blob: r and t, the radius and the azimuth of its centre, a, its
    amplitude, s, its scale along its major axis, b, its shape, q, its
    axis ratio (minor over major), and p, the angle of its major axis.

    Parameters
    ----------
    nblobs : int
        The number of blobs.
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), 'rad', 'tan'
        or 'ver' (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'mixture_moffat'

    @staticmethod
    def uid() -> int:
        return DPT_UID_MIXTURE_MOFFAT

    def __init__(
            self,
            nblobs: Annotated[int, Ge(1)],
            direction: Literal[tuple(DPT_DIRECTIONS)] = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitMixtureMoffat.__init__, locals())
        super().__init__(
            Blobs(nblobs), DispersionDirection(direction), name=name)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return self._option(Blobs).params(shape=True, unit='km/s')


class DPTraitNWUniform(DPTrait):
    """
    A velocity dispersion given at each ring: a(r). Its parameter: a,
    node-wise.

    Parameters
    ----------
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the package docstring).
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), 'rad', 'tan'
        or 'ver' (see the package docstring).
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
        return DPT_UID_NW_UNIFORM

    def __init__(
            self,
            sampling: Literal[SAMPLINGS] = SAMPLING_DEFAULT,
            direction: Literal[tuple(DPT_DIRECTIONS)] = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitNWUniform.__init__, locals())
        super().__init__(
            Sampling(sampling), DispersionDirection(direction), name=name)

    def axisymmetric(self) -> bool:
        return True

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(a=ParamVectorDesc(nnodes, unit='km/s'))


class DPTraitNWHarmonic(DPTrait):
    """
    A harmonic dispersion: a(r) cos(m (theta - p(r))), of the order m. Its
    parameters, node-wise: a, the amplitude, and p (but for order 0), the
    azimuth of a maximum; a p that changes with r winds it into a spiral.
    It is negative in parts: it is meant to perturb an axisymmetric trait.

    Parameters
    ----------
    order : int
        The order m of the harmonic: the pattern repeats every 360 / m
        degrees, and order 0 is an axisymmetric a(r).
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the package docstring).
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), 'rad', 'tan'
        or 'ver' (see the package docstring).
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
        return DPT_UID_NW_HARMONIC

    def __init__(
            self,
            order: Annotated[int, Ge(0)],
            sampling: Literal[SAMPLINGS] = SAMPLING_DEFAULT,
            direction: Literal[tuple(DPT_DIRECTIONS)] = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitNWHarmonic.__init__, locals())
        super().__init__(
            Harmonic(order), Sampling(sampling),
            DispersionDirection(direction), name=name)

    def axisymmetric(self) -> bool:
        # (a harmonic of order 0 is uniform)
        return self._option(Harmonic).order == 0

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return self._option(Harmonic).params(nnodes, unit='km/s')


class DPTraitNWDistortion(DPTrait):
    """
    A velocity dispersion around one azimuth at each ring: a(r)
    exp(-(dtheta r)^2 / (2 s(r)^2)), with dtheta the azimuth from p(r)
    (radians, within half a turn). Its parameters, node-wise: a, the peak,
    p, its azimuth, and s, its width along the ring. A p that changes with
    r winds it into a spiral arm.

    Parameters
    ----------
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the package docstring).
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), 'rad', 'tan'
        or 'ver' (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_distortion'

    @staticmethod
    def uid() -> int:
        return DPT_UID_NW_DISTORTION

    def __init__(
            self,
            sampling: Literal[SAMPLINGS] = SAMPLING_DEFAULT,
            direction: Literal[tuple(DPT_DIRECTIONS)] = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitNWDistortion.__init__, locals())
        super().__init__(
            Sampling(sampling), DispersionDirection(direction), name=name)

    def axisymmetric(self) -> bool:
        return False

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return params_nw_distortion(nnodes, unit='km/s')

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        return splits_nw_distortion(r, params, subrnodes)
