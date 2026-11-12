"""
The moments of spectra: what they are called, how they are measured,
and what a Gauss-Hermite series of them implies.

The moments of a spectrum say where its light is along the spectral
axis, in velocities: moment 0 is its flux, moment 1 its mean velocity,
moment 2 its dispersion, and the central moments of the orders 3 to 7
(in (km/s)^k) its shape. The data items of datasets of moments are named
after them, moment0 to moment7 (see MOMENT_ITEMS), but for the
coefficients h3 and h4 of a Gauss-Hermite series (e.g. as pPXF fits it
to the spectra of stars), dimensionless numbers of the shape that are
not central moments.

Three methods measure them from the spectra of a cube, each up to its
own order (see MOMENT_METHODS): the moments themselves (to 7), those of
a Gaussian fitted to each spectrum (its flux, centre and dispersion, to
2), and those of a Gauss-Hermite series fitted to it (to h4).
MomentsPlan measures them on a driver, at each evaluation of a model;
moment_maps measures them for a cube on the host (e.g. of data), and
returns MomentMaps.

A Gauss-Hermite series of velocities is not a Gaussian: its V and sigma
are not its mean and dispersion. gauss_hermite_moments gives those
(e.g. for Jeans models, which predict the mean and the dispersion of a
line of sight, fitted to the series of pPXF), and gauss_hermite_norm
its normalisation.

Datasets name their items by it, the observables measure their models
with it, and the prep task, the Studio and users measure or convert
data with it. It needs no model and no data.
"""

from . import gauss_hermite, measure, names
from .gauss_hermite import *
from .measure import *
from .names import *


__all__ = []
__all__ += gauss_hermite.__all__
__all__ += measure.__all__
__all__ += names.__all__
