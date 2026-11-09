"""
The version of the format of configurations.
"""

__all__ = [
    'FORMAT_VERSION'
]


# The version of the format of configurations, which configuration files
# (their section 'version') and projects (their marker) carry: 2, the
# configurations of 2026-10-25 on, 3, of 2026-10-28 on, 4, of 2026-11-04
# on, 5, of 2026-11-05 on, 6, of 2026-11-08 on, and 7, of 2026-11-09 on;
# gbkfit.migration rewrites the older ones
FORMAT_VERSION = 7
