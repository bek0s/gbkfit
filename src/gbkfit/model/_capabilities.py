"""
What a galaxy gives the parts that need it (Capability: its mass, its
distance, and what its gravity gives of its mass), and what a part needs
of it (Need): the components and the mass components state them, and the
galaxy checks them when it is made (see components._needs).
"""

import enum
from typing import NamedTuple


__all__ = [
    'Capability',
    'Need'
]


class Capability(enum.Enum):
    """
    What a galaxy gives the components that need it (see Component.needs),
    and what uses it from outside (e.g. a lens of its mass, see
    Galaxy.check_needs): its mass, its distance, and what its gravity gives
    of its mass (see Gravity.gives).
    """

    # The mass (e.g. the circular velocity in its gravity)
    MASS = 'mass'
    # The distance: the length of an arcsec in the galaxy, where its
    # angles (e.g. the radii of its components) meet the kpc of its mass
    DISTANCE = 'distance'
    # The forces of the mass anywhere (off the plane too), those of a field
    # (they have no curl)
    FORCES = 'forces'
    # Forces that are the gradient of a potential, in which orbits keep
    # their energy
    POTENTIAL = 'potential'
    # The vertical structure of self-gravitating isothermal sheets of
    # Newtonian gravity (h ~ sigma^2 / (G Sigma))
    SHEETS = 'sheets'
    # The shares of the components in the squared circular velocity (they
    # add up to that of the mass)
    SHARES = 'shares'
    # The bending of light by the mass as its gravity bends it: as its
    # Newtonian mass, projected on the sky (see mass.SkyProjected)
    LENSING = 'lensing'


class Need(NamedTuple):
    """
    Something that a component needs from its galaxy (see
    Component.needs): a capability, what needs it, and the mass
    components whose densities it takes, if any.

    Attributes
    ----------
    capability : Capability
        The capability.
    what : str
        Of the mass alone, the message of the error without it (e.g. "the
        'mass' velocity traits of the component 'gas' need the mass of
        their galaxy"); else, what takes the capability (e.g. "the jeans
        equilibrium of the component 'gas' takes the vertical force of the
        gravity of its galaxy").
    densities : tuple of str
        The names of the mass components whose densities it takes (e.g.
        the pressure support of a disk; see density_needs).
    surface : bool
        Whether it takes their surface densities, not their densities.
    fluid : bool
        Whether they are its fluid (the gas of a disk, see Disk): its
        galaxy checks that they are mass disks (see MassModel.check_fluid).
    """
    capability: Capability
    what: str
    densities: tuple[str, ...] = ()
    surface: bool = False
    fluid: bool = False
