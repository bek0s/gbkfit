"""
Helpers shared by the PSFs, the LSFs and the primary beams.
"""

import astropy.units as u
import numpy as np

from gbkfit.utils import fitsutils, parseutils
from gbkfit.utils.parseutils import ConfigError


# The analytic PSFs and LSFs are drawn out to at least MIN_EXTENT scale
# lengths (e.g. sigma), and further for profiles with wide wings: until
# the wings beyond hold at most WING_FLUX of their flux. Beyond that
# they are 0, and the arrays are normalised to sum to 1.
MIN_EXTENT = 8
WING_FLUX = 0.01


def check_scale(name, value):
    """Raise ConfigError unless a scale length (e.g. sigma) is > 0."""
    if not value > 0:
        raise ConfigError(f"{name} must be greater than 0; it is {value}")


def check_ratio(ratio):
    """Raise ConfigError unless an axis ratio is in (0, 1]."""
    if not 0 < ratio <= 1:
        raise ConfigError(
            f"ratio must be greater than 0 and at most 1; it is {ratio}")


def check_finite_size(kernel, *size):
    """
    Raise RuntimeError unless the size of the array of a PSF or LSF is
    finite: profiles whose wings are too heavy (e.g. a Moffat of beta
    close to its limit) put WING_FLUX of their flux at no finite distance.
    """
    if not np.all(np.isfinite(size)):
        raise RuntimeError(
            f"{kernel.__class__.__name__} {kernel.dump()}: its wings are too "
            f"heavy for an array to hold {1 - WING_FLUX:.0%} of its flux")


def embed(kernel, size, offset):
    """
    Put a kernel of odd shape (numpy order) in an array of the given size
    (FITS order), with its centre at size // 2 + offset on each axis, as
    the analytic PSFs and LSFs put theirs. It must fit.
    """
    shape = tuple(size)[::-1]
    offset = tuple(offset)[::-1]
    data = np.zeros(shape)
    slices = []
    for n, k, o in zip(shape, kernel.shape, offset):
        start = n // 2 + o - k // 2
        if start < 0 or start + k > n:
            raise RuntimeError(
                f"a kernel of size {k} does not fit in an array of size {n}")
        slices.append(slice(start, start + k))
    data[tuple(slices)] = kernel
    return data


def read_image(x):
    """
    Read an image and its world coordinates from a file option (see
    parseutils.parse_file and fitsutils.read_data).
    """
    return fitsutils.read_data(*parseutils.parse_file(x))


def spectral_points(wcs, n):
    """
    The points (a Quantity) of the first n pixels of the spectral axis of
    a WCS, or None if it has no spectral axis.
    """
    if wcs.wcs.spec < 0:
        return None
    spectral = wcs.sub([wcs.wcs.spec + 1])
    return u.Quantity(
        spectral.pixel_to_world_values(np.arange(n)), spectral.wcs.cunit[0])


def sum_weights(weights, n, desc):
    """
    Return the weights of the terms of a sum (one for each of n terms),
    each positive, normalised to sum to 1; desc names the sum in messages.
    """
    weights = np.asarray(weights, dtype=float)
    if n < 1:
        raise ConfigError(f"{desc} needs at least one term")
    if weights.shape != (n,) or not np.all(weights > 0):
        raise ConfigError(
            f"{desc} needs a positive weight for each of its {n} terms; "
            f"its weights are {weights.tolist()}")
    return weights / weights.sum()


def terms_at_velocities(terms, velocities, rest):
    """
    Return the terms of a sum or a convolution (PSFs or LSFs) at each of
    the velocities of a spectral axis: a list of terms per velocity.
    """
    per_term = [term.at_velocities(velocities, rest) for term in terms]
    return [list(terms_) for terms_ in zip(*per_term)]


def terms_velocity_range(terms, rest):
    """The range of velocities that the terms of a sum or a convolution
    are known at (see PSF.velocity_range)."""
    ranges = [term.velocity_range(rest) for term in terms]
    return max(r[0] for r in ranges), min(r[1] for r in ranges)


def dump_terms(parser, terms, prefix, dump_path, overwrite):
    """
    Return the options of the terms of a sum or a convolution, with the
    files of each named with its own prefix (term0_, term1_, ...).
    """
    return [
        parser.dump(
            term, prefix=f'{prefix}term{i}_', dump_path=dump_path,
            overwrite=overwrite)
        for i, term in enumerate(terms)]
