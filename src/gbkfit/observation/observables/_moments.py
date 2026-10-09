import numpy as np

import gbkfit.math


__all__ = [
    'MomentsPlan',
    'SPEC_RANGE',
    'SPEC_STEP',
    'check_moment_options',
    'default_spec_size',
    'spectral_axis_from_data'
]


# The default channel width (km/s) of the spectral axis of the cube that
# moments are computed from, and its default size (km/s)
SPEC_STEP = 1
SPEC_RANGE = 1000

# The smallest dispersion (km/s) a spectral axis derived from the data
# leaves room for (see spectral_axis_from_data)
_MIN_DISPERSION = 100


def default_spec_size(spec_step):
    """The number of channels of spec_step (km/s) of the default range."""
    return int(gbkfit.math.roundu_odd(SPEC_RANGE / spec_step))


def spectral_axis_from_data(dataset, spec_step):
    """
    The size and the centre of a spectral axis with the given channel
    width that covers the velocities of a moment map dataset: the range
    of mmap1, and three times the largest dispersion on each side. That
    is the largest value of mmap2, but at least _MIN_DISPERSION (also
    without mmap2), so that the lines of a model with a larger
    dispersion than the data's are not cut.
    """
    velocity = dataset['mmap1'].data()
    dispersion = _MIN_DISPERSION
    if 'mmap2' in dataset:
        dispersion = max(dispersion, np.nanmax(dataset['mmap2'].data()))
    margin = 3 * dispersion
    vmin = np.nanmin(velocity) - margin
    vmax = np.nanmax(velocity) + margin
    return dict(
        spec_size=int(gbkfit.math.roundu_odd((vmax - vmin) / spec_step)),
        spec_rval=float((vmin + vmax) / 2))


def check_moment_options(desc, orders, mask_cutoff):
    """
    The moment orders, sorted and unique; raise RuntimeError unless they
    are valid and masking is enabled (mask_cutoff), as the moments need.
    """
    orders = tuple(sorted(set(orders)))
    if not orders:
        raise RuntimeError("at least one moment order is required")
    if any(order < 0 or order > 7 for order in orders):
        raise RuntimeError("moment orders must be between 0 and 7")
    if mask_cutoff is None:
        raise RuntimeError(
            f"masking cannot be disabled for {desc}; "
            f"set the mask_cutoff to a value greater or equal to 0")
    return orders


class MomentsPlan:
    """
    The moments of the given orders of the spectra of cubes (nz, ny, nx)
    on a driver: a map (ny, nx) for each order, one mask shared by them
    (the spectra whose moment 0 is not above mask_cutoff), and the
    weights of each.
    """

    def __init__(self, driver, size, orders, mask_cutoff, dtype):
        """size is that of the maps (nx, ny)."""
        size_all = tuple(size) + (len(orders),)
        self._orders = orders
        self._mask_cutoff = mask_cutoff
        self._mmaps_o = driver.mem_alloc_d(len(orders), np.int32)
        self._mmaps_d = driver.mem_alloc_d(size_all[::-1], dtype)
        self._mmaps_m = driver.mem_alloc_d(tuple(size)[::-1], dtype)
        self._mmaps_w = driver.mem_alloc_d(size_all[::-1], dtype)
        driver.mem_copy_h2d(np.array(orders, dtype=np.int32), self._mmaps_o)
        driver.mem_fill(self._mmaps_d, np.nan)
        driver.mem_fill(self._mmaps_m, 0)
        driver.mem_fill(self._mmaps_w, 1)
        self._backend = driver.native_class('DModel', dtype)()

    def evaluate(self, step, zero, cube, wcube):
        """
        The moments of cube (and its weights, wcube, if any), whose world
        coordinates are step and zero (see fitsutils.Grid): for each
        order, its map (d), the mask (m) and its weights (w).
        """
        self._backend.mmaps_moments(
            step, zero, cube, wcube, self._mask_cutoff, self._mmaps_o,
            self._mmaps_d, self._mmaps_m, self._mmaps_w)
        return {
            f'mmap{order}': dict(
                d=self._mmaps_d[i], m=self._mmaps_m, w=self._mmaps_w[i])
            for i, order in enumerate(self._orders)}
