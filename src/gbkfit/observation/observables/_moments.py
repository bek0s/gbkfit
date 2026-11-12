"""
Helpers shared by the observables of moments: what needs their scene,
instrument or model cube (the moments themselves, their names and their
measurement, are those of gbkfit.moments).
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from gbkfit.config import warn
from gbkfit.grid import SpectralAxis, check_convention, make_cube_grid
from gbkfit.dataset import Dataset
from gbkfit.instrument import Instrument
from gbkfit.model import STARS, LineEmission, StarsEmission
from gbkfit.moments import DeviceMomentMaps
from gbkfit.utils import numutils
from . import _model_cube
from .base import ModelData, ModelItem, ObservationSetup

if TYPE_CHECKING:
    from gbkfit.model import Scene
    from .pixel_moments import PixelMoments
    from .region_moments import RegionMoments


_log = logging.getLogger(__name__)


# The default channel width (km/s) of the spectral axis of the cube that
# moments are computed from, and its default span (km/s)
SPEC_STEP = 1
SPEC_RANGE = 1000

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


def model_data(maps: DeviceMomentMaps, regions: bool = False) -> ModelData:
    """
    Return the model data of moment maps measured on a driver (see
    MomentsPlan): the model item of each, with the mask and the weights
    that they share. The moments of regions are measured from a cube of
    one row of their spectra (see RegionMomentsPlan): their items are
    the vectors of that row.
    """
    if regions:
        maps = DeviceMomentMaps(
            {key: data[0] for key, data in maps.data.items()},
            maps.mask[0], maps.weights[0])
    return {
        key: ModelItem(data, maps.mask, maps.weights)
        for key, data in maps.data.items()}
