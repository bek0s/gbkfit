"""
Helpers shared by the observables of moments.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
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
# moments, the moments of a Gaussian fitted to them (its flux, centre and
# dispersion), or a Gauss-Hermite series fitted to them (its flux, centre,
# dispersion, h3 and h4)
METHODS = ('moments', 'gaussian_fit', 'gauss_hermite')

# The highest order of the maps of each method
_MAX_ORDERS = dict(moments=7, gaussian_fit=2, gauss_hermite=4)

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
        flux_floor: float,
        method: str,
        spec_size: int | None,
        spec_step: float
) -> tuple[int, ...]:
    """
    Return the moment orders, sorted and unique. Raise ConfigError unless
    they are valid for the method (see METHODS; a Gaussian has moments 0
    to 2, a Gauss-Hermite series 0 to 4), the flux floor is from 0 to
    below 1, and the spectral axis has channels (if its size is given) of
    a positive width.
    """
    if method not in METHODS:
        raise ConfigError(
            f"method must be one of {list(METHODS)}; it is {method!r}")
    max_order = _MAX_ORDERS[method]
    orders = tuple(sorted(set(orders)))
    if not orders:
        raise ConfigError("at least one moment order is required")
    if any(order < 0 or order > max_order for order in orders):
        raise ConfigError(
            f"the moment orders of the method {method} must be between 0 "
            f"and {max_order}")
    if not 0 <= flux_floor < 1:
        raise ConfigError(
            f"flux_floor is a fraction of the flux of the brightest "
            f"spectrum, from 0 to below 1; it is {flux_floor}")
    _detail.check_positive('spec_step', spec_step)
    if spec_size is not None and spec_size < 1:
        raise ConfigError(f"spec_size must be at least 1; it is {spec_size}")
    return orders


def check_velocities(
        spec_rest: str | astropy.units.Quantity | None,
        spec_convention: str | None
) -> None:
    """
    Raise ConfigError unless the convention of the velocities of the maps
    is given (its rest, or 'relativistic' without one): the maps of the
    data were measured in one, and none is right for all (e.g. radio for
    HI, optical for lines of stars and ionised gas; at 5000 km/s, the
    rotation of the others differs by 1.6%).
    """
    if spec_rest is None and spec_convention is None:
        raise ConfigError(
            "the velocities of moment maps need their convention: give "
            "spec_rest (e.g. '1420.405752 MHz' for the radio velocities of "
            "HI, or '6562.8 Angstrom' for optical ones), or spec_convention "
            "'relativistic'")


def check_moment_units(dataset: Dataset, method: str) -> None:
    """
    Raise ConfigError unless the moments of the orders above 2 of a
    dataset are those of the method, by their units, if they have one:
    the central moments of 'moments', in (km/s)^k, or the coefficients h3
    and h4 of the Gauss-Hermite series of 'gauss_hermite', dimensionless
    (see the datasets of moments).
    """
    hermite = method == 'gauss_hermite'
    for key, item in dataset.items():
        order = int(key.removeprefix('moment'))
        unit = item.unit()
        if order <= 2 or unit is None:
            continue
        if unit.is_equivalent(astropy.units.dimensionless_unscaled) \
                != hermite:
            kind = f'h{order}, dimensionless' if hermite \
                else 'a central moment, in velocity^k'
            raise ConfigError(
                f"moment {order} of the method {method} is {kind}; the data "
                f"are in {unit}")


def warn_unmasked_noise(flux_floor: float, instrument: Instrument) -> None:
    """
    Warn if the flux floor of moments is 0 with a PSF or an LSF, whose
    convolution leaves noise in the faint parts of the model.
    """
    if flux_floor == 0 and (
            instrument.psf() is not None or instrument.lsf() is not None):
        parseutils.warn(
            "flux_floor is 0, but the instrument has a PSF or an LSF: the "
            "FFT-based convolution leaves noise in the faint parts of the "
            "model, whose moments can give artefacts; a flux_floor "
            "greater than 0 is highly recommended")


def spectra_dcube(
        observable: 'PixelMoments | RegionMoments',
        oversampling: Sequence[int],
        instrument: Instrument,
        dtype: np.dtype,
        fft_dtype: np.dtype,
        median_size: int | None = None
) -> _dcube.DCube:
    """
    Return the DCube of the spectra that the moments of an observable are
    measured from: on its spatial grid, with its spectral axis centred on
    spec_rval, smoothed by the median filter of median_size (if given; see
    DCube).
    """
    spec_size = observable.spec_size()
    return _dcube.DCube(
        size=observable.size() + (spec_size,),
        step=observable.step() + (observable.spec_step(),),
        rpix=observable.rpix() + (spec_size / 2 - 0.5,),
        rval=observable.rval() + (observable.spec_rval(),),
        rota=observable.rota(),
        rest=observable.spec_rest(),
        medium=observable.spec_medium(),
        convention=observable.spec_convention(),
        oversampling=tuple(oversampling) + (1,),
        primary_beam=instrument.primary_beam(),
        psf=instrument.psf(),
        lsf=instrument.lsf(),
        smooth_weights=False,
        dtype=dtype,
        fft_dtype=fft_dtype,
        median_size=median_size)


class MomentsPlan:
    """
    The moments of the given orders of the spectra of cubes (nz, ny, nx)
    on a driver, measured with the given method (see METHODS): a map
    (ny, nx) for each order, one mask shared by them (the spectra whose
    moment 0 is not above flux_floor times the largest of the map, in
    absolute value, and those whose fit fails), and one weight map shared
    by them (with the method moments).
    """

    def __init__(
            self,
            driver: Driver,
            size: Sequence[int],
            orders: Sequence[int],
            flux_floor: float,
            method: str,
            dtype: np.dtype
    ):
        """size is that of the maps (nx, ny)."""
        size_all = tuple(size) + (len(orders),)
        self._orders = orders
        self._flux_floor = flux_floor
        self._method = method
        self._mmaps_o = driver.mem_alloc_d(len(orders), np.int32)
        self._mmaps_d = driver.mem_alloc_d(size_all[::-1], dtype)
        self._mmaps_m = driver.mem_alloc_d(tuple(size)[::-1], dtype)
        self._mmaps_w = driver.mem_alloc_d(tuple(size)[::-1], dtype)
        driver.mem_copy_h2d(np.array(orders, dtype=np.int32), self._mmaps_o)
        driver.mem_fill(self._mmaps_d, np.nan)
        driver.mem_fill(self._mmaps_m, 0)
        driver.mem_fill(self._mmaps_w, 1)
        # (the map of moment 0 that gives the floor: its own order, mask
        # and weights)
        self._flux_o = driver.mem_alloc_d(1, np.int32)
        self._flux_d = driver.mem_alloc_d((1,) + tuple(size)[::-1], dtype)
        self._flux_m = driver.mem_alloc_d(tuple(size)[::-1], dtype)
        self._flux_w = driver.mem_alloc_d(tuple(size)[::-1], dtype)
        driver.mem_copy_h2d(np.zeros(1, dtype=np.int32), self._flux_o)
        self._driver = driver
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
        # TODO: unexplained, not investigated. Where the model has almost
        # no flux (moment 0 ~1e-4 of its peak), its moments are noise, and
        # a step of 3e-4 in any parameter switches pixels between masked
        # and not (probably moment 0 crossing the floor): the chi-square
        # jumps by 1e3-2e4 and least squares rejects every step (M83 of
        # LVHIS, tilted rings, a brightness falling to 0 at the edge of
        # the data). A guess: the round-off of the float32 FFT convolution
        # (moment 0 had pixels of -3.5e-8, though the brightness is >= 0).
        # Workaround: a disk bright beyond the data.
        cutoff = self._flux_floor * self._largest_flux(step, zero, cube)
        if self._method in ('gaussian_fit', 'gauss_hermite'):
            self._backend.mmaps_gaussian(
                step, zero, cube, cutoff, self._mmaps_o,
                self._mmaps_d, self._mmaps_m,
                hermite=self._method == 'gauss_hermite')
        else:
            self._backend.mmaps_moments(
                step, zero, cube, wcube, cutoff, self._mmaps_o,
                self._mmaps_d, self._mmaps_m, self._mmaps_w)
        return {
            f'moment{order}': dict(
                d=self._mmaps_d[i], m=self._mmaps_m, w=self._mmaps_w)
            for i, order in enumerate(self._orders)}

    def _largest_flux(
            self,
            step: Sequence[float],
            zero: Sequence[float],
            cube: DeviceArray
    ) -> float:
        """
        Return the largest moment 0 of the spectra of cube, in absolute
        value (0 without flux): of its map, measured with nothing masked
        (a cutoff below 0) and copied to the host.
        """
        if self._flux_floor == 0:
            return 0.0
        self._backend.mmaps_moments(
            step, zero, cube, None, -1, self._flux_o, self._flux_d,
            self._flux_m, self._flux_w)
        flux = np.abs(self._driver.mem_copy_d2h(self._flux_d))
        return float(np.nanmax(flux, initial=0))
