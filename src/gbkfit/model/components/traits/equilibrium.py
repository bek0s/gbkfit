"""
The traits of the vertical balance of a disk that its galaxy derives: the
dispersion that holds up its thickness (DPTraitEquilibrium), or the
thickness that its dispersion holds up (RHTraitEquilibrium).
"""

import dataclasses
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
from annotated_types import Gt
from pydantic import FiniteFloat

from gbkfit.config import Choices, Unit, check_options
from gbkfit.params.pdescs import ParamDesc, ParamVectorDesc
from .base import DPTrait, HeightInput, RHTrait
from .heights import (
    RHT_TABLE_NHEIGHTS, RHT_UID_TABLE, RHTraitGauss, RHTraitSech2)
from .options import (
    DispersionDirection, Nodewise, Sampling, TraitOption, Zmax)


__all__ = [
    'BALANCE_GRAVITIES',
    'BalanceGravity',
    'DPTraitEquilibrium',
    'RHTraitEquilibrium'
]


# The gravities of the vertical balance of a disk (see BalanceGravity)
BALANCE_GRAVITIES = ('galaxy', 'sheet', 'rotation')
BalanceGravityChoice: TypeAlias = Annotated[
    Literal[BALANCE_GRAVITIES],
    Choices(
        galaxy='The mass of the galaxy', sheet='Its own sheet',
        rotation='Its rotation')]

# The directions of the dispersion of a vertical equilibrium
EQUILIBRIUM_DIRECTIONS = ('iso', 'ver')
EquilibriumDirectionChoice: TypeAlias = Annotated[
    Literal[EQUILIBRIUM_DIRECTIONS],
    Choices(iso='Isotropic', ver='Vertical (sigma_z)')]


@dataclasses.dataclass(frozen=True)
class BalanceGravity(TraitOption):
    """
    The option gravity of a trait of the vertical balance of a disk: the
    gravity that its disk balances (one of BALANCE_GRAVITIES).
    - 'galaxy': the vertical force of all the mass of its galaxy (see
      mass.MassComponent.vertical_force). It needs the forces of the
      gravity off the plane.
    - 'sheet': the disk as an isothermal sheet of the surface density
      Sigma of its fluid (see Disk), z0 = sigma^2 / (pi G Sigma), in
      Newtonian gravity (Burkert et al. 2010): it leaves out the vertical
      pull of the other masses.
    - 'rotation': the pull of a spherical mass, of the angular speed
      Omega = v / R of the rotation of its fitted tangential velocity
      traits (GalPaK3D); in a galaxy without mass alone, as where a disk
      dominates the vertical frequency is 2.5-3 times Omega.
    """
    name: str

    def dump(self) -> dict[str, Any]:
        return dict(gravity=self.name)


class DPTraitEquilibrium(DPTrait):
    """
    The dispersion that holds a thick disk up against a gravity (its
    option gravity, see BalanceGravity), at its thickness (that of its
    density height traits), which its galaxy derives at each evaluation:
    sigma_eq(R) at the subnodes of the disk, linear between them. It adds
    to the dispersion of the other dispersion traits in quadrature, sigma
    = hypot(|their sum|, sigma_eq), at every height (isothermal in z: its
    dispersion height trait is 'one'), as an independent source of random
    motion; the fitted traits add up linearly. By gravity:
    - 'galaxy': sigma_eq^2 = int_0^inf z rho(z) (-F_z(R, z)) dz /
      int_0^inf rho(z) dz, the dispersion of the column weighted by its
      density (the vertical virial theorem, Binney & Tremaine 2008, eq.
      4.222), exact for any mass and profile of the height; the dispersion
      of the midplane is that of a profile isothermal in its own potential
      only (audit 2026-10-31 #13).
    - 'sheet': sigma_eq^2 = 2 sqrt(3) G Sigma h, of the surface density
      Sigma of its fluid and the rms height h: exact for the isothermal
      sheet, sech^2(z / z0) (<z^2> = pi^2 z0^2 / 12).
    - 'rotation': sigma_eq = h v / R, of the rms height h and the rotation
      v of the fitted tangential traits (the vertical virial theorem in a
      spherical gravity, for h << R; the thick disk of GalPaK3D).
    At the centre, where v / R and F_z / z are 0 / 0, the value at the
    next subnode; NaN where there is no solution (e.g. a negative Sigma).
    It has no parameters and no height trait (the dispersion height
    traits pair with the other dispersion traits), and comes last in the
    dispersion traits of its disk (at most one). It is a trait of thick
    disks only.

    Parameters
    ----------
    gravity : str
        The gravity: 'galaxy', 'sheet' or 'rotation'.
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), every
        direction, or 'ver', z only (the dispersion that holds the disk
        up).
    name : str, optional
        Its name (see Trait); it has no parameters to prefix.

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'equilibrium'

    @staticmethod
    def type_title() -> str:
        return 'Vertical equilibrium'

    @staticmethod
    def uid() -> int:
        raise TypeError(
            "the vertical equilibrium is not a trait of the kernels: its "
            "galaxy derives its dispersion, which they take apart")

    def __init__(
            self,
            gravity: BalanceGravityChoice,
            direction: EquilibriumDirectionChoice = 'iso',
            name: str | None = None
    ):
        check_options(DPTraitEquilibrium.__init__, locals())
        super().__init__(
            BalanceGravity(gravity), DispersionDirection(direction),
            name=name)

    def gravity(self) -> str:
        """Return the gravity that it balances (see BalanceGravity)."""
        return self._option(BalanceGravity).name

    def axisymmetric(self) -> bool:
        return True


class RHTraitEquilibrium(RHTrait):
    """
    The thickness that the dispersion of a disk holds up against a gravity
    (its option gravity, see BalanceGravity), which its galaxy derives at
    each evaluation at the subnodes of the disk, linear between them,
    from its vertical dispersion sigma_z at the midplane (that of its
    dispersion traits, isothermal in z: its dispersion height traits are
    'one'). By gravity:
    - 'galaxy': the profile of the vertical force of all the mass of its
      galaxy, rho(z) ~ exp(-(Phi(R, z) - Phi(R, 0)) / sigma_z^2), Phi(R,
      z) - Phi(R, 0) = -int_0^z F_z dz' (Olling 1995; Bacchini et al.
      2019), tabulated at RHT_TABLE_NHEIGHTS heights from 0 to zmax,
      linear between them. It needs the forces of the gravity off the
      plane; no mass disk follows its layer (not yet: the mass would take
      the thickness that its own force sets).
    - 'sheet': the isothermal sheet of the surface density Sigma of its
      fluid, sech^2(z / z0) / (2 z0), z0 = sigma_z^2 / (pi G Sigma)
      (Burkert et al. 2010); in Newtonian gravity.
    - 'rotation': in the pull of a spherical mass, a Gaussian of the
      scale h = sigma_z / Omega, Omega = v / R of the rotation v of its
      fitted tangential traits; in a galaxy without mass.
    At the centre, where v / R is 0 / 0, Omega at the next subnode. It
    has no parameters. It is the one density height trait of its disk,
    which every density polar trait shares, and its disk derives no
    dispersion besides (see DPTraitEquilibrium). A thick disk draws it,
    within zmax of its midplane; a thin disk does not ('sheet' and
    'rotation' alone), and its pressure support takes the density at its
    midplane, Sigma / (2 z0) or Sigma / (sqrt(2 pi) h) (see
    traits.PressureSupport).

    Parameters
    ----------
    gravity : str
        The gravity: 'galaxy', 'sheet' or 'rotation'.
    zmax : float, optional
        The height above the midplane (arcsec) within which the density
        lies (see RHTrait): required on a thick disk, which draws it; none
        on a thin one.
    name : str, optional
        Its name, which prefixes its parameter (see Trait).

    Raises
    ------
    ValueError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'equilibrium'

    @staticmethod
    def type_title() -> str:
        return 'Vertical equilibrium'

    def uid(self) -> int:
        # (the kernels draw its profile, of the scale its galaxy gives, or
        # its table)
        return RHT_UID_TABLE if self._profile is None \
            else self._profile.uid()

    def dump(self) -> dict[str, Any]:
        info: dict[str, Any] = dict(type=self.type(), gravity=self.gravity())
        if self._zmax is not None:
            info['zmax'] = self._zmax
        if self.name() is not None:
            info['name'] = self.name()
        return info

    def __init__(
            self,
            gravity: BalanceGravityChoice,
            zmax: Annotated[FiniteFloat, Gt(0), Unit('arcsec')] | None = None,
            name: str | None = None
    ):
        check_options(RHTraitEquilibrium.__init__, locals())
        # (node-wise, of values at each subnode, which its galaxy gives;
        # its zmax, if not drawn, any)
        super().__init__(
            BalanceGravity(gravity), Zmax(1.0 if zmax is None else zmax),
            Nodewise(True), Sampling('subrings'), name=name)
        self._zmax = zmax
        profiles = dict(sheet=RHTraitSech2, rotation=RHTraitGauss)
        self._profile = None if gravity == 'galaxy' \
            else profiles[gravity](
                1.0 if zmax is None else zmax, nodewise=True,
                sampling='subrings')

    def gravity(self) -> str:
        """Return the gravity that it balances (see BalanceGravity)."""
        return self._option(BalanceGravity).name

    def drawn(self) -> bool:
        """Check whether a thick disk can draw it: whether it has a zmax."""
        return self._zmax is not None

    def heights(self) -> np.ndarray:
        """Return the heights of its table (arcsec), of 'galaxy'."""
        return np.linspace(0, self.zmax(), RHT_TABLE_NHEIGHTS)

    def midplane_density(self, s: np.ndarray) -> np.ndarray:
        """
        Return its density at the midplane, of the scale s: 1 / (2 s) of
        the sheet, 1 / (sqrt(2 pi) s) of the Gaussian.
        """
        factor = 2 if self.gravity() == 'sheet' else np.sqrt(2 * np.pi)
        return 1 / (factor * s)

    def consts(self) -> tuple[float, ...]:
        if self._profile is None:
            return (self.zmax(), RHT_TABLE_NHEIGHTS)
        return super().consts()

    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        if self._profile is None:
            # (its table lies within zmax)
            return np.where(np.asarray(z) < 0, 0.0, 1.0)
        return self._profile.cdf(z, *params)

    def params_rnw(self, nrnodes: int) -> dict[str, ParamDesc]:
        if self._profile is not None:
            return dict(s=ParamVectorDesc(
                nrnodes, title='Scale height',
                desc="the scale height of the vertical equilibrium",
                min=0, unit='arcsec', default=0.2))
        pdf, half = self._table_names()
        heights = self.heights()
        return {
            name: ParamVectorDesc(
                nrnodes, title=f'{title} at {z:.3g} arcsec',
                desc=f"{desc} at the height {z:.3g} arcsec, which the "
                     f"galaxy computes", unit=unit, default=0)
            for names, title, desc, unit in (
                (pdf, 'Density', "the pdf of the height", '1/arcsec'),
                (half, 'Cumulative density',
                 "the share of the column between the midplane and", ''))
            for name, z in zip(names, heights)}

    def _table_names(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """
        Return the names of the parameters of its table: its pdf and its
        cumulative half at each height, of 'galaxy'.
        """
        return (tuple(f'p{k}' for k in range(RHT_TABLE_NHEIGHTS)),
                tuple(f'h{k}' for k in range(RHT_TABLE_NHEIGHTS)))

    def from_galaxy(self) -> HeightInput:
        if self._profile is not None:
            return HeightInput(('s',))
        pdf, half = self._table_names()
        return HeightInput(pdf + half)
