"""
Helpers shared by the observables of moments.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

import gbkfit.math
from gbkfit.dataset import Dataset
from gbkfit.driver import DeviceArray, Driver
from gbkfit.instrument import Instrument
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _dcube, _detail
from .base import ModelData

if TYPE_CHECKING:
    from .pixel_moments import PixelMoments
    from .region_moments import RegionMoments


# The default channel width (km/s) of the spectral axis of the cube that
# moments are computed from, and its default size (km/s)
SPEC_STEP = 1
SPEC_RANGE = 1000

# How the moment maps are measured from the spectra of a cube: their
# moments, or the moments of a Gaussian fitted to them (its flux, centre
# and dispersion)
METHODS = ('moments', 'gaussian_fit')

# The smallest dispersion (km/s) a spectral axis derived from the data
# leaves room for (see spectral_axis_from_data)
_MIN_DISPERSION = 100


def default_spec_size(spec_step: float) -> int:
    """Return the number of channels of spec_step (km/s) in SPEC_RANGE."""
    return int(gbkfit.math.roundu_odd(SPEC_RANGE / spec_step))


def spectral_axis_from_data(
        dataset: Dataset,
        spec_step: float,
        spec_size: int | None = None,
        spec_rval: float | None = None
) -> dict[str, Any]:
    """
    Return the size and the centre of a spectral axis with the given
    channel width that covers the velocities of a dataset of moments: the
    range of moment1, and three times the largest dispersion on each side.
    That is the largest value of moment2, but at least _MIN_DISPERSION
    (also without moment2), so that the lines of a model with a larger
    dispersion than the data's are not cut. A given size or centre is
    kept: the centre is that of the range, and the size covers the range
    around the centre. Without moment1, they are the defaults (the size
    None, the centre 0).
    """
    if 'moment1' not in dataset:
        spec_rval = 0 if spec_rval is None else spec_rval
        return dict(spec_size=spec_size, spec_rval=spec_rval)
    velocity = dataset['moment1'].data()
    dispersion = _MIN_DISPERSION
    if 'moment2' in dataset:
        dispersion = max(dispersion, np.nanmax(dataset['moment2'].data()))
    margin = 3 * dispersion
    vmin = np.nanmin(velocity) - margin
    vmax = np.nanmax(velocity) + margin
    if spec_rval is None:
        spec_rval = float((vmin + vmax) / 2)
    if spec_size is None:
        half = max(vmax - spec_rval, spec_rval - vmin)
        spec_size = int(gbkfit.math.roundu_odd(2 * half / spec_step))
    return dict(spec_size=spec_size, spec_rval=spec_rval)


def check_moment_options(
        orders: Sequence[int],
        mask_cutoff: float | None,
        method: str,
        spec_size: int | None,
        spec_step: float
) -> tuple[int, ...]:
    """
    Return the moment orders, sorted and unique. Raise ConfigError unless
    they are valid for the method (see METHODS; a Gaussian has moments 0
    to 2), masking is enabled (mask_cutoff, not negative), as the moments
    need, and the spectral axis has channels (if its size is given) of a
    positive width.
    """
    if method not in METHODS:
        raise ConfigError(
            f"method must be one of {list(METHODS)}; it is {method!r}")
    max_order = 7 if method == 'moments' else 2
    orders = tuple(sorted(set(orders)))
    if not orders:
        raise ConfigError("at least one moment order is required")
    if any(order < 0 or order > max_order for order in orders):
        raise ConfigError(
            f"the moment orders of the method {method} must be between 0 "
            f"and {max_order}")
    if mask_cutoff is None:
        raise ConfigError(
            "masking cannot be disabled for moments; set the mask_cutoff to "
            "a value greater or equal to 0")
    _detail.check_mask_options(mask_cutoff, False)
    _detail.check_positive('spec_step', spec_step)
    if spec_size is not None and spec_size < 1:
        raise ConfigError(f"spec_size must be at least 1; it is {spec_size}")
    return orders


def warn_unmasked_noise(mask_cutoff: float, instrument: Instrument) -> None:
    """
    Warn if the mask cutoff of moments is 0 with a PSF or an LSF, whose
    convolution leaves noise in the faint parts of the model.
    """
    if mask_cutoff == 0 and (
            instrument.psf() is not None or instrument.lsf() is not None):
        parseutils.warn(
            "mask_cutoff is 0, but the instrument has a PSF or an LSF: the "
            "FFT-based convolution leaves noise in the faint parts of the "
            "model, whose moments can give artefacts; a mask_cutoff "
            "greater than 0 is highly recommended")


def spectra_dcube(
        observable: 'PixelMoments | RegionMoments',
        scale: Sequence[int],
        instrument: Instrument,
        dtype: np.dtype
) -> _dcube.DCube:
    """
    Return the DCube of the spectra that the moments of an observable are
    measured from: on its spatial grid, with its spectral axis centred on
    spec_rval. The masking of DCube is disabled; the moments are masked
    by moment 0.
    """
    spec_size = observable.spec_size()
    return _dcube.DCube(
        size=observable.size() + (spec_size,),
        step=observable.step() + (observable.spec_step(),),
        rpix=observable.rpix() + (spec_size / 2 - 0.5,),
        rval=observable.rval() + (observable.spec_rval(),),
        rota=observable.rota(),
        rest=observable.spec_rest(),
        scale=tuple(scale) + (1,),
        primary_beam=instrument.primary_beam(),
        psf=instrument.psf(),
        lsf=instrument.lsf(),
        smooth_weights=False,
        mask_cutoff=None,
        mask_apply=False,
        dtype=dtype)


class MomentsPlan:
    """
    The moments of the given orders of the spectra of cubes (nz, ny, nx)
    on a driver, measured with the given method (see METHODS): a map
    (ny, nx) for each order, one mask shared by them (the spectra whose
    moment 0 is not above mask_cutoff, and those whose fit fails), and
    one weight map shared by them (with the method moments).
    """

    def __init__(
            self,
            driver: Driver,
            size: Sequence[int],
            orders: Sequence[int],
            mask_cutoff: float,
            method: str,
            dtype: np.dtype
    ):
        """size is that of the maps (nx, ny)."""
        size_all = tuple(size) + (len(orders),)
        self._orders = orders
        self._mask_cutoff = mask_cutoff
        self._method = method
        self._mmaps_o = driver.mem_alloc_d(len(orders), np.int32)
        self._mmaps_d = driver.mem_alloc_d(size_all[::-1], dtype)
        self._mmaps_m = driver.mem_alloc_d(tuple(size)[::-1], dtype)
        self._mmaps_w = driver.mem_alloc_d(tuple(size)[::-1], dtype)
        driver.mem_copy_h2d(np.array(orders, dtype=np.int32), self._mmaps_o)
        driver.mem_fill(self._mmaps_d, np.nan)
        driver.mem_fill(self._mmaps_m, 0)
        driver.mem_fill(self._mmaps_w, 1)
        self._backend = driver.native_class('DModel', dtype)()

    def evaluate(
            self,
            step: Sequence[float],
            zero: Sequence[float],
            cube: DeviceArray,
            wcube: DeviceArray | None
    ) -> ModelData:
        """
        Return the moments of cube (and its weights, wcube, if any), whose
        world coordinates are step and zero (see gridutils.Grid): for each
        order, its map (d), the mask (m) and its weights (w).
        """
        if self._method == 'gaussian_fit':
            self._backend.mmaps_gaussian(
                step, zero, cube, self._mask_cutoff, self._mmaps_o,
                self._mmaps_d, self._mmaps_m)
        else:
            self._backend.mmaps_moments(
                step, zero, cube, wcube, self._mask_cutoff, self._mmaps_o,
                self._mmaps_d, self._mmaps_m, self._mmaps_w)
        return {
            f'moment{order}': dict(
                d=self._mmaps_d[i], m=self._mmaps_m, w=self._mmaps_w)
            for i, order in enumerate(self._orders)}
