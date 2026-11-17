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
# trait), and 15, of 2026-11-17 on (gravitational lenses are a section of
# their own, not the foreground of an observation); gbkfit.migration
# rewrites the older ones
FORMAT_VERSION = 15
