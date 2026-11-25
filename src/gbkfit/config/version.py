"""
The version of the format of configurations.
"""

__all__ = [
    'FORMAT_VERSION'
]


# The version of the format of configurations, which configuration files
# (their section 'version') and projects (their marker) carry: 2, the
# configurations of 2026-10-25 on, 3, of 2026-10-28 on, 4, of 2026-11-04
# on, 5, of 2026-11-05 on, 6, of 2026-11-08 on, 7, of 2026-11-09 on,
# 8, of 2026-11-11 on, 9, of 2026-11-12, 10, of 2026-11-12 on (the
# likelihoods of setups), 11, of 2026-11-14 on (no lines of sight: the
# components decide, and the option depth), and 12, of 2026-11-14 on (the
# observable merged into the observation), 13, of 2026-11-16 on (no
# emission 'stars': light in bands, and the option continuum of spectral
# observations), 14, of 2026-11-16 on (the vertical equilibrium of the
# dispersion of a thick disk is its option equilibrium, not a dispersion
# trait), 15, of 2026-11-17 on (gravitational lenses are a section of
# their own, not the foreground of an observation), 16, of 2026-11-18 on
# (the alpha of Palatini f(R) gravity of the sign of f(R) = R + alpha
# R^2), 17, of 2026-11-18 on (the mass of a galaxy that a lens bends
# light by has a place of its own: mass_xpos and mass_ypos, and mass_posa
# and mass_incl if it has a plane), 18, of 2026-11-19 on (the noise of
# mock data is an option of each observation, not of the section mock),
# 19, of 2026-11-19 on (the parameters of a lens of one piece are named
# after the piece too), 20, of 2026-11-21 on (the footprint of a dataset
# is one, not one for each map of moments), and 21, of 2026-11-24 on (the
# dataset leaves the observation: the section datasets of a setup pairs
# observations with data by name), and 22, of 2026-11-25 on (what an
# observation sees of the galaxies leaves it: the section seen of a
# setup), and 23, of 2026-11-25 on (a point is not a type of PSF or LSF),
# and 24, of 2026-11-26 on (an observation says what its data trace, its
# tracer: lines, a continuum or the velocities of stars; the emission of
# a component is its light, of lines or a continuum; GBKFIT knows no
# bands; points move with kinematics, particles always);
# gbkfit.migration rewrites the older ones
FORMAT_VERSION = 24
