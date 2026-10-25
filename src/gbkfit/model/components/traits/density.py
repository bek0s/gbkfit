"""
The density polar traits (rpt): the density per unit area of the surface
of a component (see RPTrait), but the maps (see images).
"""

from collections.abc import Callable, Sequence

import numpy as np
import scipy.ndimage
import scipy.special

import gbkfit.math
from gbkfit.params.pdescs import ParamDesc, ParamScalarDesc, ParamVectorDesc
from .base import RPTrait, params_nw_distortion, splits_nw_distortion
from .options import Blobs, Harmonic, SAMPLING_DEFAULT, Sampling, Spiral


__all__ = [
    'RPTraitExponential',
    'RPTraitGGauss',
    'RPTraitGauss',
    'RPTraitLorentz',
    'RPTraitMixtureExponential',
    'RPTraitMixtureGGauss',
    'RPTraitMixtureGauss',
    'RPTraitMixtureMoffat',
    'RPTraitMoffat',
    'RPTraitNWDistortion',
    'RPTraitNWHarmonic',
    'RPTraitNWUniform',
    'RPTraitSech2',
    'RPTraitSpiral',
    'RPTraitUniform'
]


# Their uids in the native kernels
RPT_UID_UNIFORM = 1
RPT_UID_EXP = 2
RPT_UID_GAUSS = 3
RPT_UID_GGAUSS = 4
RPT_UID_LORENTZ = 5
RPT_UID_MOFFAT = 6
RPT_UID_SECH2 = 7
RPT_UID_MIXTURE_EXP = 51
RPT_UID_MIXTURE_GAUSS = 52
RPT_UID_MIXTURE_GGAUSS = 53
RPT_UID_MIXTURE_MOFFAT = 54
RPT_UID_NW_UNIFORM = 101
RPT_UID_NW_HARMONIC = 102
RPT_UID_NW_DISTORTION = 103
RPT_UID_SPIRAL = 104


def _integrate_rings(
        rings: np.ndarray,
        fun: Callable[..., np.ndarray],
        *args: np.ndarray
) -> np.ndarray:
    # ring centers, equally spaced, same width
    rsep = rings[1] - rings[0]
    # Minimum radius of the rings
    rmin = rings - rsep * 0.5
    # Maximum radius of the rings
    rmax = rings + rsep * 0.5
    # Calculate the amplitude of each ring
    ampl = fun(rings, *args)
    return ampl * np.pi * (rmax * rmax - rmin * rmin)


def _ptrait_integrate_uniform(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    a = params['a']
    rsep = rings[1] - rings[0]
    rmin = rings[0] - 0.5 * rsep
    rmax = rings[-1] + 0.5 * rsep
    return np.pi * a * (rmax * rmax - rmin * rmin)


def _ptrait_integrate_exponential(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    a = params['a']
    s = params['s']
    return _integrate_rings(rings, gbkfit.math.expon_1d_fun, a, 0, s)


def _ptrait_integrate_gauss(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    a = params['a']
    s = params['s']
    return _integrate_rings(rings, gbkfit.math.gauss_1d_fun, a, 0, s)


def _ptrait_integrate_ggauss(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    a = params['a']
    s = params['s']
    b = params['b']
    return _integrate_rings(rings, gbkfit.math.ggauss_1d_fun, a, 0, s, b)


def _ptrait_integrate_lorentz(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    a = params['a']
    s = params['s']
    return _integrate_rings(rings, gbkfit.math.lorentz_1d_fun, a, 0, s)


def _ptrait_integrate_moffat(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    a = params['a']
    s = params['s']
    b = params['b']
    return _integrate_rings(rings, gbkfit.math.moffat_1d_fun, a, 0, s, b)


def _ptrait_integrate_sech2(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    a = params['a']
    s = params['s']
    return _integrate_rings(rings, gbkfit.math.sech2_1d_fun, a, 0, s)


def _ptrait_cloud_flux_mixture(
        params: dict[str, np.ndarray], norm: np.ndarray
) -> np.ndarray:
    """
    Return the flux of the clouds of a mixture of blobs (see the native
    rp_trait_mixture_rnd): the sum of the |amplitude| times the integral
    (norm, of a blob of amplitude 1) times the axis ratio of each blob.
    The kernel gives each cloud the sign of the amplitude of its blob.
    """
    a = np.abs(np.asarray(params['a']) * np.asarray(params['q']))
    return np.sum(a * norm)


def _ptrait_integrate_mixture_exponential(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:  # noqa
    s = np.asarray(params['s'])
    return _ptrait_cloud_flux_mixture(params, 2 * np.pi * s * s)


def _ptrait_integrate_mixture_gauss(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:  # noqa
    s = np.asarray(params['s'])
    return _ptrait_cloud_flux_mixture(params, 2 * np.pi * s * s)


def _ptrait_integrate_mixture_ggauss(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:  # noqa
    s = np.asarray(params['s'])
    b = np.asarray(params['b'])
    return _ptrait_cloud_flux_mixture(
        params, 2 * np.pi * s * s * scipy.special.gamma(2 / b) / b)


def _ptrait_integrate_mixture_moffat(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:  # noqa
    s = np.asarray(params['s'])
    b = np.asarray(params['b'])
    if np.any(b <= 1):
        raise RuntimeError(
            "the blobs of mixture_moffat of the Monte Carlo disk need b > 1 "
            "(their flux is infinite otherwise)")
    return _ptrait_cloud_flux_mixture(params, np.pi * s * s / (b - 1))


def _ptrait_integrate_nw_uniform(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    a = params['a']
    c = np.inf
    return _integrate_rings(rings, gbkfit.math.uniform_1d_fun, a, 0, c)


def _ptrait_cloud_flux_nw_harmonic(
        params: dict[str, np.ndarray], rings: np.ndarray, order: int
) -> np.ndarray:
    # The integral of |a cos(k (t - p))| around a ring is 2 / pi of that of
    # |a| for k > 0
    a = params['a'] * (2 / np.pi if order else 1)
    c = np.inf
    return _integrate_rings(rings, gbkfit.math.uniform_1d_fun, a, 0, c)


def _ptrait_integrate_nw_distortion(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    """
    Return the flux of each ring of a distortion: its area times the mean
    around it of a exp(-(t r)^2 / (2 s^2)), for the azimuths t within half
    a turn of the centre of the distortion.
    """
    a = params['a']
    s = np.abs(params['s'])

    def mean(r):
        width = s / r
        erf = scipy.special.erf(
            np.pi / (np.sqrt(2) * np.maximum(width, 1e-300)))
        return np.where(
            width > 0, np.sqrt(2 * np.pi) * width * erf / (2 * np.pi), 0) * a
    return _integrate_rings(rings, mean)


class RPTraitUniform(RPTrait):
    """
    A uniform surface density: a. Its parameter: a, the density.
    Mainly for tests: its use is discouraged (a warning says so).

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters (see Trait).
    """

    @staticmethod
    def type() -> str:
        return 'uniform'

    @staticmethod
    def uid() -> int:
        return RPT_UID_UNIFORM

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(a=ParamScalarDesc())

    def has_analytical_integral(self) -> bool:
        return True

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_uniform(params, rings)


class RPTraitExponential(RPTrait):
    """
    An exponential surface density: a exp(-r / s). Its parameters: a,
    the density at the centre, and s, the scale length.

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters (see Trait).
    """

    @staticmethod
    def type() -> str:
        return 'exponential'

    @staticmethod
    def uid() -> int:
        return RPT_UID_EXP

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc())

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_exponential(params, rings)


class RPTraitGauss(RPTrait):
    """
    A Gaussian surface density: a exp(-r^2 / (2 s^2)). Its parameters:
    a, the density at the centre, and s, the dispersion of the
    Gaussian.

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters (see Trait).
    """

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def uid() -> int:
        return RPT_UID_GAUSS

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc())

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_gauss(params, rings)


class RPTraitGGauss(RPTrait):
    """
    A generalised Gaussian surface density: a exp(-(r / s)^b). Its
    parameters: a, the density at the centre, s, the scale length, and
    b, the shape (1 is the exponential, and 2 a Gaussian of dispersion s /
    sqrt(2)).

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters (see Trait).
    """

    @staticmethod
    def type() -> str:
        return 'ggauss'

    @staticmethod
    def uid() -> int:
        return RPT_UID_GGAUSS

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc(),
            b=ParamScalarDesc())

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_ggauss(params, rings)


class RPTraitLorentz(RPTrait):
    """
    A Lorentzian surface density: a s^2 / (r^2 + s^2). Its parameters:
    a, the density at the centre, and s, the half width at half
    maximum.

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters (see Trait).
    """

    @staticmethod
    def type() -> str:
        return 'lorentz'

    @staticmethod
    def uid() -> int:
        return RPT_UID_LORENTZ

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc())

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_lorentz(params, rings)


class RPTraitMoffat(RPTrait):
    """
    A Moffat surface density: a (1 + (r / s)^2)^-b. Its parameters: a,
    the density at the centre, s, the core radius, and b, the power of
    the fall.

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters (see Trait).
    """

    @staticmethod
    def type() -> str:
        return 'moffat'

    @staticmethod
    def uid() -> int:
        return RPT_UID_MOFFAT

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc(),
            b=ParamScalarDesc())

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_moffat(params, rings)


class RPTraitSech2(RPTrait):
    """
    A sech^2 surface density: a sech^2(r / s). Its parameters: a, the
    density at the centre, and s, the scale length.

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters (see Trait).
    """

    @staticmethod
    def type() -> str:
        return 'sech2'

    @staticmethod
    def uid() -> int:
        return RPT_UID_SECH2

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc())

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_sech2(params, rings)


class RPTraitMixtureExponential(RPTrait):
    """
    A surface density made of nblobs elliptical blobs (see the module
    docstring), each an exponential of the elliptical distance rho from
    its centre: a exp(-rho / s). Its parameters, vectors of a value for
    each blob: r and t, the radius and the azimuth of its centre, a, its
    amplitude, s, its scale along its major axis, q, its axis ratio (minor
    over major), and p, the angle of its major axis.

    Parameters
    ----------
    nblobs : int
        The number of blobs.
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
        return RPT_UID_MIXTURE_EXP

    def __init__(self, nblobs: int, name: str | None = None):
        super().__init__(Blobs(nblobs), name=name)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return self._option(Blobs).params(shape=False)

    def has_analytical_integral(self) -> bool:
        return True

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_mixture_exponential(params, rings)


class RPTraitMixtureGauss(RPTrait):
    """
    A surface density made of nblobs elliptical blobs (see the module
    docstring), each a Gaussian of the elliptical distance rho from its
    centre: a exp(-rho^2 / (2 s^2)). Its parameters, vectors of a value
    for each blob: r and t, the radius and the azimuth of its centre, a,
    its amplitude, s, its scale along its major axis, q, its axis ratio
    (minor over major), and p, the angle of its major axis.

    Parameters
    ----------
    nblobs : int
        The number of blobs.
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
        return RPT_UID_MIXTURE_GAUSS

    def __init__(self, nblobs: int, name: str | None = None):
        super().__init__(Blobs(nblobs), name=name)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return self._option(Blobs).params(shape=False)

    def has_analytical_integral(self) -> bool:
        return True

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_mixture_gauss(params, rings)


class RPTraitMixtureGGauss(RPTrait):
    """
    A surface density made of nblobs elliptical blobs (see the module
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
        return RPT_UID_MIXTURE_GGAUSS

    def __init__(self, nblobs: int, name: str | None = None):
        super().__init__(Blobs(nblobs), name=name)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return self._option(Blobs).params(shape=True)

    def has_analytical_integral(self) -> bool:
        return True

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_mixture_ggauss(params, rings)


class RPTraitMixtureMoffat(RPTrait):
    """
    A surface density made of nblobs elliptical blobs (see the module
    docstring), each a Moffat of the elliptical distance rho from its
    centre: a (1 + (rho / s)^2)^-b. Its parameters, vectors of a value for
    each blob: r and t, the radius and the azimuth of its centre, a, its
    amplitude, s, its scale along its major axis, b, its shape, q, its
    axis ratio (minor over major), and p, the angle of its major axis. The
    Monte Carlo disks need b > 1 (a finite flux).

    Parameters
    ----------
    nblobs : int
        The number of blobs.
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
        return RPT_UID_MIXTURE_MOFFAT

    def __init__(self, nblobs: int, name: str | None = None):
        super().__init__(Blobs(nblobs), name=name)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return self._option(Blobs).params(shape=True)

    def has_analytical_integral(self) -> bool:
        return True

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_mixture_moffat(params, rings)


class RPTraitNWUniform(RPTrait):
    """
    A surface density given at each ring: a(r). Its parameter: a,
    node-wise.

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
        return 'nw_uniform'

    @staticmethod
    def uid() -> int:
        return RPT_UID_NW_UNIFORM

    def __init__(
            self,
            sampling: str = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        super().__init__(Sampling(sampling), name=name)

    def axisymmetric(self) -> bool:
        return True

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(a=ParamVectorDesc(nnodes))

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_nw_uniform(params, rings)


class RPTraitNWHarmonic(RPTrait):
    """
    A harmonic density: a(r) cos(m (theta - p(r))), of the order m. Its
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
        return RPT_UID_NW_HARMONIC

    def __init__(
            self,
            order: int,
            sampling: str = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        super().__init__(Harmonic(order), Sampling(sampling), name=name)

    def axisymmetric(self) -> bool:
        # (a harmonic of order 0 is uniform)
        return self._option(Harmonic).order == 0

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return self._option(Harmonic).params(nnodes)

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_cloud_flux_nw_harmonic(
            params, rings, self._option(Harmonic).order)


class RPTraitNWDistortion(RPTrait):
    """
    A surface density around one azimuth at each ring: a(r)
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
        return RPT_UID_NW_DISTORTION

    def __init__(
            self,
            sampling: str = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        super().__init__(Sampling(sampling), name=name)

    def axisymmetric(self) -> bool:
        return False

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return params_nw_distortion(nnodes)

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        return splits_nw_distortion(r, params, subrnodes)

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_nw_distortion(params, rings)


class RPTraitSpiral(RPTrait):
    """
    The density of a spiral density wave in a disk, as DysmalPy's
    SpiralDensityWave with light (see Spiral): e Sigma_0(R)
    cos(phi), Sigma_0 the density of the disk without its waves (the
    azimuthal mean of the sum of its other density polar traits), 0 where
    there is no wave. It is negative between the arms (with e <= 1, the
    density of the disk stays positive), and its azimuthal mean is 0, so
    that the gravity of its galaxy does not take it (see DiskDensity).
    Its velocities are the velocity trait 'spiral' (see VPTraitSpiral).
    The Monte Carlo disks draw its clouds as those of a harmonic in each
    ring (see RPTraitNWHarmonic), of the phase f of the ring. With a given
    wavenumber, it does not take the rotation of its disk (e.g. the arms
    of an image).

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
        return RPT_UID_SPIRAL

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

    def axisymmetric(self) -> bool:
        return False

    def takes_rotation(self) -> bool:
        return self._option(Spiral).wavenumber == 'dispersion'

    def params_sm(self) -> dict[str, ParamDesc]:
        return self._option(Spiral).params_sm(pattern_speed=False)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return self._option(Spiral).params_rnw(self.wave_params(), nnodes)

    def wave_params(self) -> tuple[str, ...]:
        return 'a', 'f'

    def wave(
            self,
            r: np.ndarray,
            v: np.ndarray,
            dv: np.ndarray,
            density: np.ndarray | None,
            params: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """
        Return the tables of the wave (see Spiral.wavenumber_at),
        in the density of its disk without its waves at the radii r
        (density): its amplitude ('a', e Sigma_0, 0 where there is no
        wave) and f ('f', radians), by name.
        """
        _, _, k, waves = self._option(Spiral).wavenumber_at(r, v, dv, params)
        return dict(
            a=np.where(waves, float(params['e']) * density, 0),
            f=Spiral.phase(r, k))

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_cloud_flux_nw_harmonic(
            params, rings, self._option(Spiral).order)
