"""
The units of mass models: lengths in kpc, masses in Msun, velocities in
km/s, densities in Msun/kpc^3, potentials in (km/s)^2 and forces (per
unit mass) in (km/s)^2/kpc.
"""

import astropy.constants


__all__ = [
    'G'
]


# The gravitational constant (kpc (km/s)^2 / Msun)
G = astropy.constants.G.to_value('kpc km2 / (s2 solMass)')
