"""
Helpers shared by the observables of moments.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

import numpy as np
from annotated_types import Ge, Lt

from gbkfit.config import Choices, warn
from gbkfit.grid import SpectralAxis, check_convention, make_cube_grid
from gbkfit.dataset import Data, Dataset
from gbkfit.driver import DeviceArray, Driver
from gbkfit.instrument import Instrument
from gbkfit.model import STARS, LineEmission, StarsEmission
from gbkfit.params import ModelEvaluationError
from gbkfit.utils import numutils
from . import _model_cube
from .base import ModelData, ObservationSetup

if TYPE_CHECKING:
    from gbkfit.model import Scene
    from .pixel_moments import PixelMoments
    from .region_moments import RegionMoments


__all__ = [
    'MOMENT_METHODS',
    'MomentsPlan',
    'check_moment_orders'
]


_log = logging.getLogger(__name__)


# The default channel width (km/s) of the spectral axis of the cube that
# moments are computed from, and its default span (km/s)
SPEC_STEP = 1
SPEC_RANGE = 1000

# How the moment maps are measured from the spectra of a cube: their
# moments, the moments of a Gaussian fitted to them (its flux, centre and
# dispersion), or a Gauss-Hermite series fitted to them (its flux, centre,
# dispersion, h3 and h4)
MOMENT_METHODS = ('moments', 'gaussian_fit', 'gauss_hermite')
Method: TypeAlias = Annotated[Literal[MOMENT_METHODS], Choices(
    moments='Moments', gaussian_fit='Gaussian fit',
    gauss_hermite='Gauss-Hermite fit')]

# The flux floor of the moments: a fraction of the flux of the brightest
# spectrum (see PixelMoments)
FluxFloor: TypeAlias = Annotated[float, Ge(0), Lt(1)]

# The highest order of the maps of each method
_MAX_ORDERS = dict(moments=7, gaussian_fit=2, gauss_hermite=4)

# The smallest dispersion (km/s) a spectral axis derived from the data
# leaves room for (see spectral_axis)
_MIN_DISPERSION = 100

# The percentiles of the moments of the data that a spectral axis derived
# from them takes (see spectral_axis): of moment1, the values within three
# dispersions of the range between them; of moment2, the upper one
_PERCENTILES = (0.5, 99.5)


def spectral_axis(
        spectral: SpectralAxis,
        dataset: Dataset | None = None
) -> SpectralAxis:
    """
    Return the spectral axis of the cube that moments are measured from:
    the given one, with the defaults of the options not given: the step
    SPEC_STEP; the size and the velocity rval, at the channel rpix (the
    centre by default), that cover the velocities of the moments of the
    data, if they have moment 1, or else SPEC_RANGE and 0. The velocities
    of the data are the range of moment1, and three times the largest
    dispersion on each side: that of moment2, but at least
    _MIN_DISPERSION (also without moment2), so that the lines of a model
    with a larger dispersion than the data's are not cut. So that a few
    bad values (e.g. at noisy edges) do not make the axis long, the
    largest dispersion is the upper percentile of _PERCENTILES of
    moment2, and the range that of the values of moment1 within three of
    it of the range between its percentiles; a warning tells how many
    are not (the velocities of the model there are its own, but such
    data are likely bad: mask them, e.g. by the sigma clipping of
    gbkfit.prep). A given size, rpix or rval is kept. Without rval, the
    axis is centred on the range (or on 0), and rval is the velocity of
    its channel rpix: a given rpix only refers to a channel of the same
    axis. With rval, the size covers the range around rval, at the
    centre (or from the first channel, rval - rpix step, given rpix).

    Raises
    ------
    ValueError
        If the convention of its velocities is not known (see
        check_convention).
    """
    check_convention(spectral)
    step = spectral.step() or SPEC_STEP
    size, rpix, rval = spectral.size(), spectral.rpix(), spectral.rval()
    centre = 0.0
    if dataset is not None and 'moment1' in dataset:
        velocity = dataset['moment1'].data()
        dispersion = _MIN_DISPERSION
        if 'moment2' in dataset:
            dispersion = max(dispersion, np.nanpercentile(
                dataset['moment2'].data(), _PERCENTILES[1]))
        margin = 3 * dispersion
        low, high = np.nanpercentile(velocity, _PERCENTILES)
        kept = (velocity >= low - margin) & (velocity <= high + margin)
        if beyond := int(np.count_nonzero(np.isfinite(velocity) & ~kept)):
            _log.warning(
                f"{beyond} values of moment1 are beyond {low - margin:.0f} "
                f"to {high + margin:.0f} km/s (its {_PERCENTILES} "
                f"percentiles, and three times the dispersion beyond), "
                f"which the cube of the model covers: they are likely bad; "
                f"mask them (e.g. by sigma clipping, see gbkfit.prep)")
        vmin = np.min(velocity[kept]) - margin
        vmax = np.max(velocity[kept]) + margin
        centre = float((vmin + vmax) / 2)
        if size is None:
            if rval is None:
                size = int(numutils.roundu_odd((vmax - vmin) / step))
            elif rpix is None:
                half = max(vmax - rval, rval - vmin)
                size = int(numutils.roundu_odd(2 * half / step))
            else:
                size = int(np.ceil((vmax - rval) / step + rpix)) + 1
    if size is None:
        size = int(numutils.roundu_odd(SPEC_RANGE / step))
    middle = size / 2 - 0.5
    if rpix is None:
        rpix = middle
    if rval is None:
        rval = centre + (rpix - middle) * step
    return spectral.replaced(SpectralAxis(
        size=size, step=step, rpix=rpix, rval=rval))


def check_moment_orders(
        orders: Sequence[int],
        method: str
) -> tuple[int, ...]:
    """
    Return the moment orders, sorted and unique. Raise ValueError unless
    they are valid for the method (see MOMENT_METHODS; a Gaussian has moments 0
    to 2, a Gauss-Hermite series 0 to 4).
    """
    max_order = _MAX_ORDERS[method]
    orders = tuple(sorted(set(orders)))
    if not orders:
        raise ValueError("at least one moment order is required")
    if any(order < 0 or order > max_order for order in orders):
        raise ValueError(
            f"the moment orders of the method {method} must be between 0 "
            f"and {max_order}")
    return orders


def moment_item(order: int, method: str) -> str:
    """
    Return the name of the data item of the moment of an order measured by
    a method (see MOMENT_METHODS): moment0 to moment7, the central moments (the
    flux, the mean velocity, the dispersion, and above in (km/s)^k), but
    the coefficients h3 and h4 of a Gauss-Hermite series (gauss_hermite;
    dimensionless, as pPXF names them).
    """
    if method == 'gauss_hermite' and order > 2:
        return f'h{order}'
    return f'moment{order}'


def moment_maps(
        items: Mapping[str, Data], orders: Sequence[int], method: str
) -> tuple[dict[int, Data], dict[int, Data]]:
    """
    Return the data items of the moments of the orders measured by a method
    (see moment_item), as datasets of moments take them: the central
    moments, and the coefficients of a Gauss-Hermite series, by order.
    """
    moments, hermite = {}, {}
    for order in orders:
        name = moment_item(order, method)
        (hermite if name.startswith('h') else moments)[order] = items[name]
    return moments, hermite


def check_moment_items(dataset: Dataset, method: str) -> None:
    """
    Raise ValueError if a dataset has moments that a method does not
    measure: the coefficients h3 and h4 of a Gauss-Hermite series without
    the method gauss_hermite, or central moments of the orders 3 and 4 with
    it (see moment_item: of the same order, they are other numbers).
    """
    hermite = method == 'gauss_hermite'
    for key in dataset:
        if key.startswith('h') and not hermite:
            raise ValueError(
                f"{key}, a coefficient of a Gauss-Hermite series, is "
                f"measured by the method gauss_hermite; the method is "
                f"{method}")
        if hermite and key in ('moment3', 'moment4'):
            raise ValueError(
                f"{key} is a central moment, in (km/s)^{key[-1]}; the method "
                f"gauss_hermite measures h{key[-1]}, a coefficient of a "
                f"Gauss-Hermite series (name the data h{key[-1]}, if they "
                f"are)")


def warn_unmasked_noise(flux_floor: float, instrument: Instrument) -> None:
    """
    Warn if the flux floor of moments is 0 with a PSF or an LSF, whose
    convolution leaves noise in the faint parts of the model.
    """
    if flux_floor == 0 and (
            instrument.psf() is not None or instrument.lsf() is not None):
        warn(
            "flux_floor is 0, but the instrument has a PSF or an LSF: the "
            "FFT-based convolution leaves noise in the faint parts of the "
            "model, whose moments can give artefacts; a flux_floor "
            "greater than 0 is highly recommended")


def spectra_model_cube(
        observable: 'PixelMoments | RegionMoments',
        setup: ObservationSetup,
        median_size: int | None = None
) -> _model_cube.ModelCube:
    """
    Return the ModelCube of the spectra that the moments of an observable are
    measured from, as an observation of the given setup sees them (see
    ObservationSetup): on its spatial grid and its spectral axis (see
    spectral_axis), smoothed by the median filter of median_size (if
    given; see ModelCube).
    """
    grid = observable.grid()
    coords = grid.coords
    return _model_cube.ModelCube(
        make_cube_grid(
            grid.size, coords.step, coords.rpix, coords.rval, coords.rota,
            observable.spectral()),
        oversampling=setup.oversampling + (1,),
        primary_beam=setup.instrument.primary_beam(),
        psf=setup.instrument.psf(),
        lsf=setup.instrument.lsf(),
        smooth_weights=False,
        dtype=setup.dtype,
        fft_dtype=setup.fft_dtype,
        median_size=median_size)


def check_one_line(scene: 'Scene') -> None:
    """
    Raise ValueError if the moments of the spectra of a scene would mix
    several lines: its emitters draw more than one (e.g. Ha and [NII] of
    one component, or of two; or stars and a line of gas). Moments cannot
    tell the lines of a spectrum apart, whatever the spectral axis covers,
    so that an observation of moments names the line of its data (its
    option lines), whose moments its data are.
    """
    lines = set()
    for galaxy, emitters in enumerate(scene.emitters()):
        for index in emitters:
            emission = scene.component(galaxy, index).emission()
            if isinstance(emission, StarsEmission):
                if scene.lines() is not None and STARS in scene.lines():
                    lines.add(STARS)
            elif isinstance(emission, LineEmission):
                names = emission.lines().names()
                selected = emission.lines().select(scene.lines())
                # The one line of a component without lines: whatever line
                # the spectral axis is of, the same for all such components
                if names:
                    lines.update(names[i] for i in selected)
                else:
                    lines.add(None)
    if len(lines) > 1:
        named = sorted(str(line) for line in lines if line is not None)
        raise ValueError(
            f"the moments of a spectrum mix its lines, and the observation "
            f"sees several: {named}; name the line of its data with its "
            f"option lines (e.g. lines: [{named[0]!r}])")


class MomentsPlan:
    """
    The moments of the given orders of the spectra of cubes (nz, ny, nx)
    on a driver, measured with the given method (see MOMENT_METHODS): a map
    (ny, nx) for each order, one mask shared by them (the spectra whose
    moment 0 is not above flux_floor times the largest of the map, in
    absolute value, and those whose fit fails), and one weight map shared
    by them (with the method moments). To evaluate, a cube whose moment
    0 is not a number (NaN) somewhere is a model that failed there, not
    one without flux there; measure takes cubes of data, with channels
    without a value (see measure).
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
        world coordinates are step and zero (see Grid): for each
        order, its map (d), the mask (m) and its weights (w).

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
        cutoff = self._flux_floor * float(np.max(np.abs(flux), initial=0))
        return self.measure(step, zero, cube, wcube, cutoff)

    def measure(
            self,
            step: Sequence[float],
            zero: Sequence[float],
            cube: DeviceArray,
            wcube: DeviceArray | None,
            cutoff: float
    ) -> ModelData:
        """
        Return the moments of cube, as evaluate, but with the spectra
        whose moment 0 is not above cutoff (in absolute value) masked.
        The fits of the methods gaussian_fit and gauss_hermite leave the
        channels without a value (NaN) out; the moments of a spectrum
        with one are not numbers.
        """
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
            moment_item(order, self._method): dict(
                d=self._mmaps_d[i], m=self._mmaps_m, w=self._mmaps_w)
            for i, order in enumerate(self._orders)}

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
