"""
The trait of the vertical balance of a disk that its galaxy derives: the
dispersion that holds up its thickness (DPTraitEquilibrium).
"""

import dataclasses
from typing import Annotated, Any, Literal, TypeAlias

from gbkfit.config import Choices, check_options
from .base import DPTrait
from .options import DispersionDirection, TraitOption


__all__ = [
    'BALANCE_GRAVITIES',
    'BalanceGravity',
    'DPTraitEquilibrium'
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
