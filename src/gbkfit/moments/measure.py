"""
The measurement of moment maps from the spectra of cubes: on a driver,
at each evaluation of a model (MomentsPlan), and of a cube on the host
(e.g. of data, or of a model evaluated alone; moment_maps).
"""

import dataclasses
from collections.abc import Sequence
from typing import Annotated, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt
from annotated_types import Ge, Lt

from gbkfit.driver import DeviceArray, Driver
from gbkfit.driver.drivers.host import DriverHost
from gbkfit.params import ModelEvaluationError
from .names import MOMENT_METHODS, check_moment_orders, moment_item


__all__ = [
    'DeviceMomentMaps',
    'FluxFloor',
    'MomentMaps',
    'MomentsPlan',
    'moment_maps'
]


# The flux floor of the moments: a fraction of the flux of the brightest
# spectrum (see MomentsPlan)
FluxFloor: TypeAlias = Annotated[float, Ge(0), Lt(1)]


class DeviceMomentMaps(NamedTuple):
    """
    The moment maps of the spectra of a cube, measured on a driver (see
    MomentsPlan).

    Attributes
    ----------
    data : dict of str to DeviceArray
        The map (ny, nx) of each order, by the name of its data item (see
        moment_item); NaN where its spectrum is masked.
    mask : DeviceArray
        Where the moments are measured (1) or the spectrum is masked (0),
        shared by the maps.
    weights : DeviceArray
        The weights of the maps (of the weights of the cube, with the
        method moments; 1 otherwise), shared by them.
    """
    data: dict[str, DeviceArray]
    mask: DeviceArray
    weights: DeviceArray


class MomentsPlan:
    """
    The moments of the given orders of the spectra of cubes (nz, ny, nx)
    on a driver, measured with the given method (see MOMENT_METHODS): a
    map (ny, nx) for each order, one mask shared by them (the spectra
    whose moment 0 is not above flux_floor times the largest of the map,
    in absolute value, and those whose fit fails), and one weight map
    shared by them (with the method moments). To evaluate, a cube whose
    moment 0 is not a number (NaN) somewhere is a model that failed
    there, not one without flux there; measure takes cubes of data, with
    channels without a value (see measure; e.g. mock data, whose noise
    blanks the spectra of values without an error).

    Parameters
    ----------
    driver : Driver
        The driver of the cubes.
    size : Sequence of int
        The size of the maps (nx, ny).
    orders : Sequence of int
        The orders of the moments (see check_moment_orders).
    flux_floor : float
        The share of the largest moment 0 of the map (in absolute value)
        that the moment 0 of a spectrum must be above.
    method : str
        The method (see MOMENT_METHODS).
    dtype : np.dtype
        The floating type of the cubes.
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
    ) -> DeviceMomentMaps:
        """
        Return the moments of cube (and its weights, wcube, if any), a
        model whose world coordinates are step and zero (see Grid): for
        each order, its map, and the mask and the weights they share.

        Parameters
        ----------
        step, zero : Sequence of float
            The step of each axis of the cube, and the world position of
            its first pixel (see Grid.zero).
        cube : DeviceArray
            The cube (nz, ny, nx).
        wcube : DeviceArray or None
            Its weights, if any.

        Returns
        -------
        DeviceMomentMaps
            The maps.

        Raises
        ------
        ModelEvaluationError
            If the moment 0 of a spectrum of the cube is not a number.
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
        flux = self._flux(step, zero, cube)
        failed = np.count_nonzero(np.isnan(flux))
        if failed:
            raise ModelEvaluationError(
                f"the model has no flux (NaN) in {failed} of its spectra: it "
                f"cannot be evaluated there (e.g. a rotation or a dispersion "
                f"without a solution)")
        return self.measure(step, zero, cube, wcube, self._cutoff(flux))

    def measure(
            self,
            step: Sequence[float],
            zero: Sequence[float],
            cube: DeviceArray,
            wcube: DeviceArray | None,
            cutoff: float | None = None
    ) -> DeviceMomentMaps:
        """
        Return the moments of cube, as evaluate, but of data: with the
        spectra whose moment 0 is not above cutoff (in absolute value)
        masked, and a spectrum without one (NaN) missing, not a model
        that failed. The fits of the methods gauss_fit and gauss_hermite
        leave the channels without a value (NaN) out; the moments of a
        spectrum with one are not numbers.

        Parameters
        ----------
        step, zero, cube, wcube
            As those of evaluate.
        cutoff : float, optional
            The moment 0 that a spectrum must be above (in absolute
            value): by default, the flux floor times the largest of the
            spectra that have one.

        Returns
        -------
        DeviceMomentMaps
            The maps, as evaluate.
        """
        if cutoff is None:
            cutoff = self._cutoff(self._flux(step, zero, cube))
        if self._method in ('gauss_fit', 'gauss_hermite'):
            self._backend.mmaps_gaussian(
                step, zero, cube, cutoff, self._mmaps_o,
                self._mmaps_d, self._mmaps_m,
                hermite=self._method == 'gauss_hermite')
        else:
            self._backend.mmaps_moments(
                step, zero, cube, wcube, cutoff, self._mmaps_o,
                self._mmaps_d, self._mmaps_m, self._mmaps_w)
        return DeviceMomentMaps(
            {moment_item(order, self._method): self._mmaps_d[i]
             for i, order in enumerate(self._orders)},
            self._mmaps_m, self._mmaps_w)

    def _flux(
            self,
            step: Sequence[float],
            zero: Sequence[float],
            cube: DeviceArray
    ) -> np.ndarray:
        """
        Return the map of the moment 0 of the spectra of cube, measured
        with nothing masked (a cutoff below 0), on the host.
        """
        self._backend.mmaps_moments(
            step, zero, cube, None, -1, self._flux_o, self._flux_d,
            self._flux_m, self._flux_w)
        return self._driver.mem_copy_d2h(self._flux_d)

    def _cutoff(self, flux: np.ndarray) -> float:
        """
        Return the cutoff of the moment 0 of the spectra of a map of it
        (see _flux): the flux floor times the largest (in absolute
        value) of those that are numbers; 0 without any.
        """
        return self._flux_floor * float(np.nanmax(np.abs(flux), initial=0))


@dataclasses.dataclass(frozen=True)
class MomentMaps:
    """
    The moment maps of a spectral cube (see moment_maps).

    Attributes
    ----------
    data : dict of str to np.ndarray
        The map (ny, nx) of each order, by name (moment0, moment1, ...;
        h3 and h4 of a Gauss-Hermite series, see moment_maps).
    mask : np.ndarray
        Where the moments are measured (ny, nx): False where the spectrum
        is masked, whose moments from order 1 are NaN (moment 0 keeps its
        value, but is NaN where the spectrum has no value at all, and
        with the method moments, where a channel has none).
    """
    data: dict[str, np.ndarray]
    mask: np.ndarray

    def skewness(self) -> np.ndarray:
        """
        Return the map of the skewness of the spectra: moment 3 over the
        dispersion (moment 2) cubed; NaN where the dispersion is 0. Of
        maps measured as moments (a Gauss-Hermite series has h3 and h4,
        its own measures of the shape, instead).

        Raises
        ------
        KeyError
            If the maps have no moment 2 or 3.
        """
        return self._standardized(3)

    def kurtosis(self) -> np.ndarray:
        """
        Return the map of the excess kurtosis of the spectra: moment 4 over
        the dispersion (moment 2) to the power 4, less 3 (0 for a
        Gaussian); NaN where the dispersion is 0. Of maps measured as
        moments, as skewness.

        Raises
        ------
        KeyError
            If the maps have no moment 2 or 4.
        """
        return self._standardized(4) - 3

    def _standardized(self, order: int) -> np.ndarray:
        """Return a central moment over the dispersion to its order."""
        if f'moment{order}' not in self.data:
            raise KeyError(
                f"the maps have no moment {order} (those of a Gauss-Hermite "
                f"series have h3 and h4, not central moments)")
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = self.data[f'moment{order}'] \
                / self.data['moment2'] ** order
        return np.where(np.isfinite(ratio), ratio, np.nan)


def moment_maps(
        cube: npt.ArrayLike,
        velocities: npt.ArrayLike,
        orders: Sequence[int] = (0, 1, 2),
        method: str = 'moments',
        flux_floor: float = 0.0,
        driver: Driver | None = None
) -> MomentMaps:
    """
    Return the moment maps of a spectral cube, measured as the observable
    of moments (PixelMoments) measures those of its models: the moments
    of each spectrum (its flux, mean velocity, dispersion, and central
    moments above 2), or those of a Gaussian or a Gauss-Hermite series
    fitted to it. A channel without a value (NaN) is missing, not one
    without flux: the fits leave it out, and the moments of a spectrum
    with one are masked (NaN, moment 0 too).

    Parameters
    ----------
    cube : ArrayLike
        The cube (channels, ny, nx).
    velocities : ArrayLike
        The velocities of its channels, evenly spaced, in either order.
    orders : Sequence of int, optional
        The orders of the maps.
    method : str, optional
        How the maps are measured: 'moments'; 'gauss_fit' (the flux,
        centre and dispersion of a Gaussian fitted to each spectrum, for
        the orders 0 to 2); or 'gauss_hermite' (of a Gauss-Hermite series,
        for the orders 0 to 4: its flux, centre, dispersion, h3 and h4).
    flux_floor : float, optional
        The spectra whose moment 0 is not above this fraction of the
        largest (in absolute value) are masked: from 0 to below 1.
    driver : Driver, optional
        The driver that measures them (a DriverHost by default).

    Returns
    -------
    MomentMaps
        The maps.

    Raises
    ------
    ValueError
        If the cube is not of the velocities, they are not evenly spaced
        (or fewer than two), or the options are not valid.
    """
    cube = np.asarray(cube)
    velocities = np.asarray(velocities, dtype=float)
    if cube.ndim != 3 or velocities.shape != (len(cube),):
        raise ValueError(
            "the cube must be (channels, ny, nx), of one velocity per "
            "channel")
    steps = np.diff(velocities)
    if len(velocities) < 2 or not np.allclose(
            steps, steps[0], rtol=1e-6, atol=0) or steps[0] == 0:
        raise ValueError(
            "the velocities must be at least two, and evenly spaced")
    if method not in MOMENT_METHODS:
        raise ValueError(
            f"the method must be one of {MOMENT_METHODS}; it is {method!r}")
    if not 0 <= flux_floor < 1:
        raise ValueError(
            f"the flux floor must be from 0 to below 1; it is {flux_floor}")
    orders = check_moment_orders(orders, method)
    # (the velocities from the lowest, so that moment 0 is positive)
    if steps[0] < 0:
        cube, velocities = cube[::-1], velocities[::-1]
    dtype = np.float32 if cube.dtype == np.float32 else np.float64
    cube = cube.astype(dtype)
    blank = np.isnan(cube).all(axis=0)
    step = (velocities[-1] - velocities[0]) / (len(velocities) - 1)
    flux = np.nansum(cube, axis=0) * step
    cutoff = flux_floor * float(np.max(np.abs(flux), initial=0))
    if driver is None:
        driver = DriverHost()
    _, ny, nx = cube.shape
    plan = MomentsPlan(
        driver, (nx, ny), orders, flux_floor, method, dtype)
    maps = plan.measure(
        (1, 1, step), (0, 0, velocities[0]), driver.mem_copy_h2d(cube),
        None, cutoff)
    data = {key: driver.mem_copy_d2h(value)
            for key, value in maps.data.items()}
    mask = driver.mem_copy_d2h(maps.mask) == 1
    mask &= ~blank
    for value in data.values():
        value[blank] = np.nan
    return MomentMaps(data=data, mask=mask)
