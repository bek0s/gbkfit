"""
The vertical structure of mass disks (see disks.MassDisk): a law, the
same probability density p(z) of the height z at every radius (of a
scale height h), whose transform T(k) = int p(z) exp(-k |z|) dz sets the
gravity of the disk in its plane; or that of a component of the galaxy,
which may change with radius.
"""

import abc
from collections.abc import Collection
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import scipy.integrate
import scipy.special
from annotated_types import Gt

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import check_options
from .base import DiskDensity, MassContext


__all__ = [
    'DiskVertical',
    'VerticalComponent',
    'VerticalExponential',
    'VerticalGauss',
    'VerticalLaw',
    'VerticalSech',
    'VerticalSech2',
    'VerticalThin',
    'disk_vertical_parser'
]


class DiskVertical(parseutils.TypedSerializable, abc.ABC):
    """
    The vertical structure of a mass disk: how its mass is spread in
    height at each radius.
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    def followed_component(self) -> str | None:
        """
        Return the name of the component of the galaxy whose density it
        follows; None here.
        """
        return None

    def knots(self, context: MassContext) -> np.ndarray:
        """
        Return the radii (kpc) where its shape bends (e.g. those it takes
        the shape at, linear between them), in the context of its galaxy
        (see MassContext); none here.
        """
        return np.empty(0)

    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name; none here."""
        return {}

    @abc.abstractmethod
    def thin(self, context: MassContext) -> bool:
        """
        Check whether the disk is thin (of no thickness), in the context
        of its galaxy (see MassContext).
        """
        pass

    @abc.abstractmethod
    def thin_given(self, thick: Collection[str]) -> bool:
        """
        Check whether the disk is thin, in a galaxy whose thick components
        (see DiskDensity.thick) have the given names: as thin is, but
        known when the galaxy is made.
        """
        pass

    @abc.abstractmethod
    def height_pdf(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return the pdf of the height (1/kpc) at the points (R, z) (kpc),
        broadcast together.

        Raises
        ------
        NotImplementedError
            If the disk is thin, or the pdf is not given at any point.
        """
        pass

    @abc.abstractmethod
    def share_within(
            self,
            R: npt.ArrayLike,
            h: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return the share of the mass at the radii R within the heights
        |z| < h (kpc), broadcast together: all of it (1) if the disk is
        thin.
        """
        pass


class VerticalLaw(DiskVertical, abc.ABC):
    """
    A vertical law: the same pdf of the height at every radius, of the
    scale height hz (kpc), a parameter (none if thin).
    """

    def pdescs(self) -> dict[str, ParamDesc]:
        return dict(hz=ParamScalarDesc(
            desc="the scale height", minimum=0, unit='kpc'))

    def thin(self, context: MassContext) -> bool:
        return False

    def thin_given(self, thick: Collection[str]) -> bool:
        return False

    def height_pdf(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self.pdf(z, params['hz'])

    def share_within(
            self,
            R: npt.ArrayLike,
            h: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        R, h = np.broadcast_arrays(
            np.asarray(R, dtype=float), np.asarray(h, dtype=float))
        return self.within(h, params['hz'])

    @staticmethod
    @abc.abstractmethod
    def pdf(z: npt.ArrayLike, h: float) -> np.ndarray:
        """Return the pdf p(z) at the heights z, for the scale height h."""
        pass

    @staticmethod
    @abc.abstractmethod
    def within(heights: np.ndarray, h: float) -> np.ndarray:
        """
        Return the share of the pdf within the heights |z| < heights, for
        the scale height h.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def transform(k: npt.ArrayLike, h: float) -> np.ndarray:
        """
        Return T(k) = int p(z) exp(-k |z|) dz at the wavenumbers k >= 0,
        for the scale height h.
        """
        pass


class VerticalThin(VerticalLaw):
    """
    No thickness: all the mass in the plane (a thin disk), T(k) = 1. It
    has no pdf of the height, and no scale height.
    """

    @staticmethod
    def type() -> str:
        return 'thin'

    def pdescs(self) -> dict[str, ParamDesc]:
        return {}

    def thin(self, context: MassContext) -> bool:
        return True

    def thin_given(self, thick: Collection[str]) -> bool:
        return True

    def height_pdf(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        raise NotImplementedError(
            "a thin disk has a surface density, not a density")

    def share_within(
            self,
            R: npt.ArrayLike,
            h: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return np.ones(np.broadcast(np.asarray(R), np.asarray(h)).shape)

    @staticmethod
    def pdf(z: npt.ArrayLike, h: float) -> np.ndarray:
        raise NotImplementedError("a thin disk has no pdf of the height")

    @staticmethod
    def within(heights: np.ndarray, h: float) -> np.ndarray:
        return np.ones_like(heights)

    @staticmethod
    def transform(k: npt.ArrayLike, h: float) -> np.ndarray:
        return np.ones_like(np.asarray(k, dtype=float))


class VerticalExponential(VerticalLaw):
    """
    The exponential law: p(z) = exp(-|z| / h) / (2 h), T(k) = 1 / (1 + k
    h).
    """

    @staticmethod
    def type() -> str:
        return 'exponential'

    @staticmethod
    def pdf(z: npt.ArrayLike, h: float) -> np.ndarray:
        z = np.abs(np.asarray(z, dtype=float))
        return np.exp(-z / h) / (2 * h)

    @staticmethod
    def within(heights: np.ndarray, h: float) -> np.ndarray:
        return -np.expm1(-heights / h)

    @staticmethod
    def transform(k: npt.ArrayLike, h: float) -> np.ndarray:
        return 1 / (1 + np.asarray(k, dtype=float) * h)


class VerticalSech2(VerticalLaw):
    """
    The sech2 law: p(z) = sech^2(z / h) / (2 h), T(k) = (a/2) (psi(a/4 +
    1/2) - psi(a/4)) - 1 with a = k h (psi: the digamma function).
    """

    @staticmethod
    def type() -> str:
        return 'sech2'

    @staticmethod
    def pdf(z: npt.ArrayLike, h: float) -> np.ndarray:
        z = np.abs(np.asarray(z, dtype=float))
        return 1 / (2 * h * np.cosh(np.minimum(z / h, 350)) ** 2)

    @staticmethod
    def within(heights: np.ndarray, h: float) -> np.ndarray:
        return np.tanh(heights / h)

    @staticmethod
    def transform(k: npt.ArrayLike, h: float) -> np.ndarray:
        a = np.asarray(k, dtype=float) * h
        a_safe = np.where(a > 0, a, 1)
        transform = a_safe / 2 * (
            scipy.special.digamma(a_safe / 4 + 0.5)
            - scipy.special.digamma(a_safe / 4)) - 1
        return np.where(a > 0, transform, 1)


class VerticalSech(VerticalLaw):
    """
    The sech law: p(z) = sech(z / h) / (pi h), with the share within |z|
    < H of (2 / pi) arctan(sinh(H / h)), and T(k) = (psi(a/4 + 3/4) -
    psi(a/4 + 1/4)) / pi with a = k h (psi: the digamma function).
    """

    @staticmethod
    def type() -> str:
        return 'sech'

    @staticmethod
    def pdf(z: npt.ArrayLike, h: float) -> np.ndarray:
        # (sech(x) = 2 e^-x / (1 + e^-2x), for x >= 0)
        x = np.abs(np.asarray(z, dtype=float)) / h
        return 2 * np.exp(-x) / (1 + np.exp(-2 * x)) / (np.pi * h)

    @staticmethod
    def within(heights: np.ndarray, h: float) -> np.ndarray:
        return 2 / np.pi * np.arctan(np.sinh(np.minimum(heights / h, 700)))

    @staticmethod
    def transform(k: npt.ArrayLike, h: float) -> np.ndarray:
        a = np.asarray(k, dtype=float) * h
        return (scipy.special.digamma(a / 4 + 0.75)
                - scipy.special.digamma(a / 4 + 0.25)) / np.pi


class VerticalGauss(VerticalLaw):
    """
    The Gaussian law, of dispersion h: T(k) = erfcx(k h / sqrt(2)).
    """

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def pdf(z: npt.ArrayLike, h: float) -> np.ndarray:
        z = np.abs(np.asarray(z, dtype=float))
        return np.exp(-z ** 2 / (2 * h ** 2)) / (np.sqrt(2 * np.pi) * h)

    @staticmethod
    def within(heights: np.ndarray, h: float) -> np.ndarray:
        return scipy.special.erf(heights / (np.sqrt(2) * h))

    @staticmethod
    def transform(k: npt.ArrayLike, h: float) -> np.ndarray:
        return scipy.special.erfcx(np.asarray(k, dtype=float) * h / np.sqrt(2))


class VerticalComponent(DiskVertical):
    """
    The vertical structure of a component of the galaxy (its density
    height traits; thin without them): at each radius, the shape of its
    density, sum_i S_i(R) p_i(z | R) / sum_i S_i(R) (S_i and p_i: the
    surface density and the pdf of the height of each of its terms, e.g.
    of each density polar trait), taken at its subnodes (or at those of
    rings of width rstep) and linear between them. Where the component
    has no density, the shape is that of the nearest subnodes with
    density: interpolated linearly between them on either side, and held
    beyond the first and the last. This is an assumption (e.g. a disk
    may flare on beyond the last ring of the component), and it only
    matters where the shape changes with radius (a flaring component, or
    several terms).

    Parameters
    ----------
    component : str
        The name of the component it follows.
    rstep : float, optional
        The width (in the units of the rings of the component, e.g.
        arcsec) of the rings at whose subnodes it takes the shape; by
        default, the subnodes of the component.

    Raises
    ------
    ConfigError
        If the name of the component or rstep is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'component'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(
            component=self._component, rstep=self._rstep)

    def __init__(
            self, component: str, rstep: Annotated[float, Gt(0)] | None = None
    ):
        check_options(VerticalComponent.__init__, locals())
        parseutils.check_name(component)
        self._component = component
        self._rstep = rstep

    def component(self) -> str:
        """Return the name of the component it follows."""
        return self._component

    def rstep(self) -> float | None:
        """Return the width of the rings of its subnodes (see above)."""
        return self._rstep

    def followed_component(self) -> str | None:
        return self._component

    def thin(self, context: MassContext) -> bool:
        return not context.density(self._component).thick()

    def thin_given(self, thick: Collection[str]) -> bool:
        return self._component not in thick

    def height_pdf(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        # The shape at each point, from those at the two subnodes around it
        if self.thin(context):
            raise NotImplementedError(
                "a thin disk has a surface density, not a density")
        R, z = np.broadcast_arrays(
            np.asarray(R, dtype=float), np.asarray(z, dtype=float))
        density, knots, sigmas, total = self._shape_knots(context)
        left, right, t = _between(R, knots)

        def at(nodes: np.ndarray) -> np.ndarray:
            # (the shape at the subnodes of the given indices, at z)
            out = np.empty(R.shape)
            for k in np.unique(nodes):
                points = nodes == k
                pdfs = density.height_pdfs(knots[k:k + 1], z[points])
                out[points] = sigmas[:, k] @ pdfs[:, 0] / total[k]
            return out
        return (1 - t) * at(left) + t * at(right)

    def share_within(
            self,
            R: npt.ArrayLike,
            h: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        # The shares of the shapes at the two subnodes around each radius
        # (by adaptive quadrature, to 1e-10; the shapes are even in z)
        R, h = np.broadcast_arrays(
            np.asarray(R, dtype=float), np.asarray(h, dtype=float))
        if self.thin(context):
            return np.ones(R.shape)
        density, knots, sigmas, total = self._shape_knots(context)
        left, right, t = _between(R, knots)

        def share(k: int, height: float) -> float:
            def shape(z: float) -> float:
                pdfs = density.height_pdfs(knots[k:k + 1], np.array([z]))
                return float(sigmas[:, k] @ pdfs[:, 0, 0]) / total[k]
            value, _ = scipy.integrate.quad(
                shape, 0, height, epsabs=0, epsrel=1e-10, limit=200)
            return 2 * value
        shares = [
            (1 - tk) * share(lk, hk) + tk * share(rk, hk)
            for lk, rk, tk, hk in zip(
                left.ravel(), right.ravel(), t.ravel(), h.ravel())]
        return np.reshape(shares, R.shape)

    def knots(self, context: MassContext) -> np.ndarray:
        """
        Return the radii (kpc) at which it takes the shape (see above),
        in the context of its galaxy (see MassContext).
        """
        return context.density(self._component).knots(self._rstep)

    def shapes(
            self, R: np.ndarray, z: np.ndarray, context: MassContext
    ) -> np.ndarray:
        """
        Return the shape (1/kpc) at the radii R and the heights z (kpc),
        in the context of its galaxy (see MassContext): an array of shape
        (len(R), len(z)).

        Raises
        ------
        ValueError
            If the component has no density at any of its subnodes.
        """
        density, knots, sigmas, total = self._shape_knots(context)
        pdfs = density.height_pdfs(knots, z)
        shapes = np.einsum('tk,tkz->kz', sigmas, pdfs) / total[:, None]
        # Linear between the subnodes with density, held beyond them
        left, right, t = _between(np.asarray(R, dtype=float), knots)
        t = t[:, None]
        return (1 - t) * shapes[left] + t * shapes[right]

    def _shape_knots(
            self, context: MassContext
    ) -> tuple[DiskDensity, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the density of the component, and the subnodes at which it
        takes the shape, those with density (kpc), with the surface
        density of each term there (an array of shape (terms, subnodes))
        and their sum.

        Raises
        ------
        ValueError
            If the component has no density at any of its subnodes.
        """
        density = context.density(self._component)
        knots = density.knots(self._rstep)
        sigmas = density.surface_densities(knots)
        total = np.sum(sigmas, axis=0)
        full = total > 0
        if not full.any():
            raise ValueError(
                f"the component {self._component!r} has no density, so no "
                f"vertical structure")
        return density, knots[full], sigmas[:, full], total[full]


def _between(
        R: np.ndarray, knots: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the indices of the subnodes on either side of each radius R,
    and how far between them it is (0 to 1), clipped to the first and the
    last: a value linear between the subnodes, held beyond them, is (1 -
    t) times that at left plus t times that at right.
    """
    if len(knots) == 1:
        zeros = np.zeros(R.shape, dtype=int)
        return zeros, zeros, np.zeros(R.shape)
    radii = np.clip(R, knots[0], knots[-1])
    right = np.clip(np.searchsorted(knots, radii), 1, len(knots) - 1)
    left = right - 1
    return left, right, (radii - knots[left]) / (knots[right] - knots[left])


disk_vertical_parser = parseutils.TypedParser(DiskVertical, [
    VerticalThin,
    VerticalExponential,
    VerticalSech2,
    VerticalSech,
    VerticalGauss,
    VerticalComponent])
