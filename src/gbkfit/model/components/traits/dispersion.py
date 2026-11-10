"""
The velocity dispersion polar traits (dpt): isotropic, or along one
direction of a disk (see DPTrait).
"""

from typing import Annotated

from annotated_types import Ge

from gbkfit.config import check_options
from .base import DPTrait
from .shapes import (
    ExponentialShape, GaussShape, GGaussShape, LorentzShape,
    MixtureExponentialShape, MixtureGaussShape, MixtureGGaussShape,
    MixtureMoffatShape, MoffatShape, NWDistortionShape, NWHarmonicShape,
    NWUniformShape, Sech2Shape, UniformShape)
from .options import (
    DPT_DIRECTION_DEFAULT, SAMPLING_DEFAULT,
    Blobs, DPTDirectionChoice, DispersionDirection, Harmonic, Sampling,
    SamplingChoice)


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




class DPTraitUniform(UniformShape, DPTrait):
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
    ValueError
        If an option is not valid.
    """

    def __init__(
            self,
            direction: DPTDirectionChoice = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitUniform.__init__, locals())
        super().__init__(DispersionDirection(direction), name=name)


class DPTraitExponential(ExponentialShape, DPTrait):
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
    ValueError
        If an option is not valid.
    """

    def __init__(
            self,
            direction: DPTDirectionChoice = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitExponential.__init__, locals())
        super().__init__(DispersionDirection(direction), name=name)


class DPTraitGauss(GaussShape, DPTrait):
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
    ValueError
        If an option is not valid.
    """

    def __init__(
            self,
            direction: DPTDirectionChoice = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitGauss.__init__, locals())
        super().__init__(DispersionDirection(direction), name=name)


class DPTraitGGauss(GGaussShape, DPTrait):
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
    ValueError
        If an option is not valid.
    """

    def __init__(
            self,
            direction: DPTDirectionChoice = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitGGauss.__init__, locals())
        super().__init__(DispersionDirection(direction), name=name)


class DPTraitLorentz(LorentzShape, DPTrait):
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
    ValueError
        If an option is not valid.
    """

    def __init__(
            self,
            direction: DPTDirectionChoice = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitLorentz.__init__, locals())
        super().__init__(DispersionDirection(direction), name=name)


class DPTraitMoffat(MoffatShape, DPTrait):
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
    ValueError
        If an option is not valid.
    """

    def __init__(
            self,
            direction: DPTDirectionChoice = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitMoffat.__init__, locals())
        super().__init__(DispersionDirection(direction), name=name)


class DPTraitSech2(Sech2Shape, DPTrait):
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
    ValueError
        If an option is not valid.
    """

    def __init__(
            self,
            direction: DPTDirectionChoice = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitSech2.__init__, locals())
        super().__init__(DispersionDirection(direction), name=name)


class DPTraitMixtureExponential(MixtureExponentialShape, DPTrait):
    """
    A velocity dispersion made of elliptical blobs, each an exponential of
    the elliptical distance rho from its centre: a exp(-rho / s). It has
    nblobs blobs (see the package docstring). Its parameters, vectors of a
    value for each blob: r and t, the radius and the azimuth of its centre,
    a, its amplitude, s, its scale along its major axis, q, its axis ratio
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
    ValueError
        If an option is not valid.
    """

    def __init__(
            self,
            nblobs: Annotated[int, Ge(1)],
            direction: DPTDirectionChoice = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitMixtureExponential.__init__, locals())
        super().__init__(
            Blobs(nblobs), DispersionDirection(direction), name=name)


class DPTraitMixtureGauss(MixtureGaussShape, DPTrait):
    """
    A velocity dispersion made of elliptical blobs, each a Gaussian of the
    elliptical distance rho from its centre: a exp(-rho^2 / (2 s^2)). It
    has nblobs blobs (see the package docstring). Its parameters, vectors
    of a value for each blob: r and t, the radius and the azimuth of its
    centre, a, its amplitude, s, its scale along its major axis, q, its
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
    ValueError
        If an option is not valid.
    """

    def __init__(
            self,
            nblobs: Annotated[int, Ge(1)],
            direction: DPTDirectionChoice = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitMixtureGauss.__init__, locals())
        super().__init__(
            Blobs(nblobs), DispersionDirection(direction), name=name)


class DPTraitMixtureGGauss(MixtureGGaussShape, DPTrait):
    """
    A velocity dispersion made of elliptical blobs, each a generalised
    Gaussian of the elliptical distance rho from its centre:
    a exp(-(rho / s)^b). It has nblobs blobs (see the package docstring).
    Its parameters, vectors of a value for each blob: r and t, the radius
    and the azimuth of its centre, a, its amplitude, s, its scale along its
    major axis, b, its shape, q, its axis ratio (minor over major), and p,
    the angle of its major axis.

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
    ValueError
        If an option is not valid.
    """

    def __init__(
            self,
            nblobs: Annotated[int, Ge(1)],
            direction: DPTDirectionChoice = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitMixtureGGauss.__init__, locals())
        super().__init__(
            Blobs(nblobs), DispersionDirection(direction), name=name)


class DPTraitMixtureMoffat(MixtureMoffatShape, DPTrait):
    """
    A velocity dispersion made of elliptical blobs, each a Moffat of the
    elliptical distance rho from its centre: a (1 + (rho / s)^2)^-b. It has
    nblobs blobs (see the package docstring). Its parameters, vectors of a
    value for each blob: r and t, the radius and the azimuth of its centre,
    a, its amplitude, s, its scale along its major axis, b, its shape, q,
    its axis ratio (minor over major), and p, the angle of its major axis.

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
    ValueError
        If an option is not valid.
    """

    def __init__(
            self,
            nblobs: Annotated[int, Ge(1)],
            direction: DPTDirectionChoice = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitMixtureMoffat.__init__, locals())
        super().__init__(
            Blobs(nblobs), DispersionDirection(direction), name=name)


class DPTraitNWUniform(NWUniformShape, DPTrait):
    """
    A velocity dispersion given at each ring: a(r). Its parameter: a,
    node-wise.

    Parameters
    ----------
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rings',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the package docstring).
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), 'rad', 'tan'
        or 'ver' (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    def __init__(
            self,
            sampling: SamplingChoice = SAMPLING_DEFAULT,
            direction: DPTDirectionChoice = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitNWUniform.__init__, locals())
        super().__init__(
            Sampling(sampling), DispersionDirection(direction), name=name)


class DPTraitNWHarmonic(NWHarmonicShape, DPTrait):
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
        Where the values of its node-wise parameters are given: 'rings',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the package docstring).
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), 'rad', 'tan'
        or 'ver' (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    def __init__(
            self,
            order: Annotated[int, Ge(0)],
            sampling: SamplingChoice = SAMPLING_DEFAULT,
            direction: DPTDirectionChoice = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitNWHarmonic.__init__, locals())
        super().__init__(
            Harmonic(order), Sampling(sampling),
            DispersionDirection(direction), name=name)


class DPTraitNWDistortion(NWDistortionShape, DPTrait):
    """
    A velocity dispersion around one azimuth at each ring: a(r)
    exp(-(dtheta r)^2 / (2 s(r)^2)). Here dtheta is the azimuth from p(r)
    (radians, within half a turn). Its parameters, node-wise: a, the peak,
    p, its azimuth, and s, its width along the ring. A p that changes with
    r winds it into a spiral arm.

    Parameters
    ----------
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rings',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the package docstring).
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), 'rad', 'tan'
        or 'ver' (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    def __init__(
            self,
            sampling: SamplingChoice = SAMPLING_DEFAULT,
            direction: DPTDirectionChoice = DPT_DIRECTION_DEFAULT,
            name: str | None = None
    ):
        check_options(DPTraitNWDistortion.__init__, locals())
        super().__init__(
            Sampling(sampling), DispersionDirection(direction), name=name)
