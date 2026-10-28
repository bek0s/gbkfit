"""
Helpers shared by the PSFs, the LSFs and the primary beams.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import astropy.units as u
import astropy.wcs
import numpy as np

from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError

if TYPE_CHECKING:
    from .lsfs import LSF
    from .psfs import PSF

# The terms of a sum or a convolution: all PSFs, or all LSFs
_Term = TypeVar('_Term', 'PSF', 'LSF')


# The analytic PSFs and LSFs are drawn out to at least MIN_EXTENT scale
# lengths (e.g. sigma), and further for profiles with wide wings: until
# the wings beyond hold at most WING_FLUX of their flux. Beyond that
# they are 0, and the arrays are normalised to sum to 1.
MIN_EXTENT = 8
WING_FLUX = 0.01


def check_finite_size(kernel: 'PSF | LSF', *size: float) -> None:
    """
    Raise RuntimeError unless the size of the array of a PSF or LSF is
    finite: profiles whose wings are too heavy (e.g. a Moffat of beta
    close to its limit) put WING_FLUX of their flux at no finite distance.
    """
    if not np.all(np.isfinite(size)):
        raise RuntimeError(
            f"{kernel.__class__.__name__} {kernel.dump()}: its wings are too "
            f"heavy for an array to hold {1 - WING_FLUX:.0%} of its flux")


def embed(
        kernel: np.ndarray, size: Sequence[int], offset: Sequence[int]
) -> np.ndarray:
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


def spectral_points(wcs: astropy.wcs.WCS, n: int) -> u.Quantity | None:
    """
    Return the points of the first n pixels of the spectral axis of a WCS,
    or None if it has no spectral axis.
    """
    if wcs.wcs.spec < 0:
        return None
    spectral = wcs.sub([wcs.wcs.spec + 1])
    return u.Quantity(
        spectral.pixel_to_world_values(np.arange(n)), spectral.wcs.cunit[0])


def sum_weights(
        weights: Sequence[float], n: int, desc: str
) -> np.ndarray:
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


def terms_at(
        terms: Sequence[_Term],
        points: u.Quantity,
        convention: gridutils.VelocityConvention | None
) -> list[list[_Term]]:
    """
    Return the terms of a sum or a convolution (PSFs or LSFs) at each of
    the points of the spectral axis (1D): a list of terms per point.
    """
    per_term = [term.at(points, convention) for term in terms]
    return [list(terms_) for terms_ in zip(*per_term)]


def terms_velocity_range(
        terms: Sequence[_Term], convention: gridutils.VelocityConvention | None
) -> tuple[float, float]:
    """
    Return the range of velocities that the terms of a sum or a
    convolution are known at (see PSF.velocity_range).
    """
    ranges = [term.velocity_range(convention) for term in terms]
    return max(r[0] for r in ranges), min(r[1] for r in ranges)


def dump_terms(
        parser: parseutils.TypedParser,
        terms: Sequence[_Term],
        prefix: str,
        dump_path: bool,
        overwrite: bool
) -> list[dict[str, Any]]:
    """
    Return the options of the terms of a sum or a convolution, with the
    files of each named with its own prefix (term0_, term1_, ...).
    """
    return [
        parser.dump(
            term, prefix=f'{prefix}term{i}_', dump_path=dump_path,
            overwrite=overwrite)
        for i, term in enumerate(terms)]
