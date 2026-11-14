"""
The checks of the options of an observation on the sky (see
ObservationSky): its offsets, its flux scale, the unit of its model
without data, and the beams of its data against its PSF.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import astropy.units
import numpy as np

import gbkfit.units
from gbkfit.config import config_path, make_typed_desc, warn
from gbkfit.dataset import Dataset
from gbkfit.grid import VelocityConvention
from gbkfit.instrument import PSF, Instrument, PSFGauss
from gbkfit.io import Beam, beam_area
from gbkfit.model import LightUnit

if TYPE_CHECKING:
    from .base import ObservationSky


def light_unit(
        unit: str,
        dataset: Dataset | None,
        observation: 'ObservationSky',
        instrument: Instrument
) -> tuple[LightUnit, Beam | None]:
    """
    Return the unit of the light of the model of an observation without
    data, as given (see ObservationSky: unit), and its beam: for a unit
    per beam, that of the PSF of its instrument (see _psf_beam), of each
    channel of the spectral axis of the observation for a PSF that
    varies; else None.

    Raises
    ------
    ValueError
        If it has data, which give the unit, the unit is not a unit, or a
        unit per beam has no PSF, or one without a beam.
    """
    if dataset is not None:
        raise ValueError(
            "the data of an observation give the unit of its model; remove "
            "its unit")
    with config_path('unit'):
        unit = gbkfit.units.parse_unit(unit)
    if gbkfit.units.spatial_unit(unit) != astropy.units.beam:
        return LightUnit(unit), None
    psf = instrument.psf()
    if psf is None:
        raise ValueError(
            f"a unit per beam ({gbkfit.units.format_unit(unit)}) takes the "
            f"beam of the PSF of the instrument, which has none")
    if not psf.varies():
        beam = _psf_beam(psf)
        return LightUnit(unit, beam.area()), beam
    spectral = observation.spectral()
    if spectral is None:
        raise ValueError(
            "a PSF that varies along the spectral axis needs an observation "
            "with one")
    velocities = (
        spectral.rval() + spectral.step()
        * (np.arange(spectral.size()) - spectral.rpix())) \
        * astropy.units.km / astropy.units.s
    convention = VelocityConvention(
        spectral.rest(), spectral.medium(), spectral.convention())
    beams = [_psf_beam(channel) for channel in psf.at(velocities, convention)]
    beam = Beam(
        np.array([b.bmaj for b in beams]), np.array([b.bmin for b in beams]),
        np.array([b.bpa for b in beams]), velocities)
    return LightUnit(unit, beam.area(), velocities), beam


def _psf_beam(psf: PSF) -> Beam:
    """
    Return the beam of a PSF (of one channel), as of the unit of light
    per beam: a Gaussian PSF is its own beam (see PSFGauss.beam); another
    PSF is a round Gaussian beam of its solid angle (see PSF.beam_area),
    the beam that data per beam carry (see Data).
    """
    if isinstance(psf, PSFGauss):
        return psf.beam()
    # (the solid angle of a round Gaussian beam is beam_area(1, 1) fwhm^2)
    fwhm = float(np.sqrt(psf.beam_area() / beam_area(1, 1)))
    return Beam(fwhm, fwhm)


def check_offsets(
        offsets: Mapping[str, bool] | None, name: str | None
) -> dict[str, bool]:
    """
    Return the offsets of the frame of an observation (see Observation),
    those it fits, or raise ValueError if they are not position and
    velocity, true or false, or it has some but no name.
    """
    if offsets is None:
        return {}
    if not isinstance(offsets, Mapping) \
            or not set(offsets) <= {'position', 'velocity'} \
            or not all(isinstance(v, bool) for v in offsets.values()):
        raise ValueError(
            f"offsets are 'position' and 'velocity', true or false; they "
            f"are {offsets!r}")
    fitted = {key: True for key, value in offsets.items() if value}
    if fitted and name is None:
        raise ValueError(
            "an observation with offsets needs a name, which names their "
            "parameters (e.g. '{name}_dx')")
    return fitted


def check_flux_scale(
        flux_scale: str | None,
        observation: 'ObservationSky',
        name: str | None
) -> None:
    """
    Raise ValueError unless a flux scale is one of an observation (see
    ObservationSky.flux_scales), and the observation has a name.
    """
    if flux_scale is None:
        return
    kinds = observation.flux_scales()
    if flux_scale not in kinds:
        desc = make_typed_desc(observation.__class__, 'observation')
        raise ValueError(
            f"{desc} has the flux scales {list(kinds)}; not {flux_scale!r}")
    if name is None:
        raise ValueError(
            "an observation with a flux scale needs a name, which names its "
            "parameter ('{name}_flux_scale')")


def has_correlated_noise(dataset: Dataset) -> bool:
    """Return whether the noise of an item of the data is correlated."""
    return any(
        item.spatial_correlation() is not None
        or item.spectral_correlation() is not None
        for item in dataset.values())


def check_beams(
        dataset: Dataset,
        instrument: Instrument,
        observation: 'ObservationSky'
) -> None:
    """
    Warn (see warn) if an item of the data has a beam, and the PSF of the
    instrument is not that beam: the data, of a radio interferometer, are
    smoothed by their beam, a Gaussian, and their light per beam is per
    it, but the PSF smooths the model otherwise. A cube of a beam for each
    channel is checked against the PSF at the point of each (see PSF.at).
    """
    psf = instrument.psf()
    for key in dataset.keys():
        beam = dataset[key].beam()
        if beam is None:
            continue
        if beam.spectral is None:
            if psf is not None and psf.varies():
                warn(
                    f"the data ({key}) have one beam, {_beam_text(beam)}, "
                    f"but the PSF of the instrument varies along the "
                    f"spectral axis: the model is smoothed by the PSF, and "
                    f"its light per beam is per beam of the data")
            elif not _is_beam(psf, beam):
                warn(
                    f"the data ({key}) have the beam {_beam_text(beam)}, "
                    f"but the PSF of the instrument is {_psf_text(psf)}: the "
                    f"model is smoothed by the PSF, and its light per beam is "
                    f"per beam of the data")
            continue
        # A beam for each channel, against the PSF at the point of each
        beams = [
            Beam(*values) for values in zip(beam.bmaj, beam.bmin, beam.bpa)]
        psfs = [None] * len(beams)
        if psf is not None:
            try:
                psfs = psf.at(
                    beam.spectral,
                    VelocityConvention.of(observation.grid().coords))
            except ValueError:
                # (points of the PSF that the spectral axis cannot place;
                # the model says so)
                continue
        differ = [
            i for i, (x, y) in enumerate(zip(psfs, beams))
            if not _is_beam(x, y)]
        if differ:
            i = differ[0]
            warn(
                f"the data ({key}) have a beam for each channel, but the PSF "
                f"of the instrument is another at {len(differ)} of their "
                f"{len(beams)} channels: e.g. at "
                f"{gbkfit.units.format_quantity(beam.spectral[i])}, the beam "
                f"{_beam_text(beams[i])} and the PSF {_psf_text(psfs[i])}; "
                f"the model is smoothed by the PSF, and its light per beam "
                f"is per beam of the data")


def _is_beam(psf: PSF | None, beam: Beam) -> bool:
    """
    Return whether a PSF is a beam (of one channel): a Gaussian of its
    axes, and of its angle unless it is round.
    """
    if not isinstance(psf, PSFGauss):
        return False
    own = psf.beam()
    return bool(np.allclose(
        (beam.bmaj, beam.bmin), (own.bmaj, own.bmin), rtol=1e-3, atol=0)
        and (np.isclose(beam.bmaj, beam.bmin, rtol=1e-3)
             or abs((beam.bpa - own.bpa + 90) % 180 - 90) < 0.1))


def _beam_text(beam: Beam) -> str:
    """A beam (of one channel), in messages."""
    return f"{beam.bmaj:g} x {beam.bmin:g} arcsec at {beam.bpa:g} deg"


def _psf_text(psf: PSF | None) -> str:
    """A PSF, in messages: its beam, if it is a Gaussian."""
    if psf is None:
        return "none (a point)"
    if isinstance(psf, PSFGauss):
        return f"the beam {_beam_text(psf.beam())}"
    return f"of type {psf.type()!r}, not a Gaussian"
