"""
The spectra of a source in a slitlet of the micro-shutter array of
JWST NIRSpec, from the products of its pipeline: those of each exposure
(cal), or rectified (s2d), for the datasets of slitless spectra.
"""

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import astropy.coordinates
import astropy.units
import numpy as np
import numpy.typing as npt

from gbkfit.utils.parseutils import ConfigError

if TYPE_CHECKING:
    from gbkfit.dataset import DatasetSlitlessSpectra
    from gbkfit.observation.observables.apertures import ApertureShutters
    from gbkfit.observation.observables.slitless_spectra import Exposure
    from gbkfit.observation.observables.traces import TraceTable


__all__ = [
    'SlitletSpectra',
    'nirspec_msa_rectified',
    'nirspec_msa_spectra'
]


@dataclasses.dataclass(frozen=True)
class SlitletSpectra:
    """
    The spectra of a source in a slitlet of the micro-shutter array of
    JWST NIRSpec in one exposure, ready for slitless spectra behind the
    shutters (see SlitlessSpectra.from_data, with the trace, the aperture
    and the exposures; and nirspec_msa_spectra).

    Attributes
    ----------
    dataset : DatasetSlitlessSpectra
        The spectra on the detector, with the world coordinates of the
        undispersed light.
    trace : TraceTable
        The trace, of the model of the instrument.
    aperture : ApertureShutters
        The open shutters of the slitlet.
    exposures : tuple of Exposure
        The exposures combined into the spectra: the exposure, and those
        of the other nods subtracted as its background, if any.
    resampled : bool
        Whether the spectra were resampled from the detector (rectified).
    """
    dataset: 'DatasetSlitlessSpectra'
    trace: 'TraceTable'
    aperture: 'ApertureShutters'
    exposures: tuple['Exposure', ...]
    resampled: bool = False


def nirspec_msa_spectra(
        filename: str, undo_corrections: bool = True
) -> list[SlitletSpectra]:
    """
    Read the spectra of a source in a slitlet of the micro-shutter array
    of JWST NIRSpec, of each of its exposures: from the cal file of the
    source of stage 3 of the pipeline, its cutouts of the detector, not
    resampled (needs the package stdatamodels: the extra jwst).

    The cutout of each exposure is the image of slitless spectra behind
    its shutters (see SlitlessSpectra). Its grid is that of the detector,
    with the world coordinates of the undispersed light: its y along the
    slit (against the y of the slit frame of the pipeline) and its x
    perpendicular (within a degree of the dispersion), its reference
    pixel where the centre of the shutter of the source lands at the
    middle wavelength of the cutout, and its pixels of the sizes by which
    the light of the sky moves a pixel of the detector there. The trace
    is the model of the instrument of the pipeline (its WCS from the slit
    to the detector, for the light of each wavelength from each place in
    the shutters, which lands as that of other wavelengths from across a
    shutter), sampled at nodes about every 4 pixels of the detector along
    the wavelength and half a pixel across the places. The wavelengths
    are those of the instrument (in vacuum; the pipeline applies no
    barycentric correction). The pixels flagged DO_NOT_USE, or not finite,
    are masked.

    If the pipeline subtracted the background of each exposure with the
    others (its step bkg_subtract, as for nods), each also has the other
    exposures, of the sky moved to where it was in them (through the
    plane of the shutters, shared by them), of weights -1 / (n - 1).

    The model of slitless spectra behind shutters makes the losses of the
    light through them, so the corrections of the pipeline for them (its
    steps pathloss and barshadow) are undone (from the corrections in the
    file), unless undo_corrections is False.

    Parameters
    ----------
    filename : str
        The cal file of the source (e.g.
        jw01210-o001_s000009382_nirspec_f290lp-g395m_cal.fits).
    undo_corrections : bool, optional
        Whether to undo the corrections for the losses through the
        shutters.

    Returns
    -------
    list of SlitletSpectra
        The spectra of each exposure.

    Raises
    ------
    ConfigError
        If the file is not of the exposures of a source of the
        micro-shutter array, or a slitlet has closed shutters between
        open ones or no shutter of the source.
    """
    from stdatamodels.jwst import datamodels
    with datamodels.open(filename) as model:
        exposures = getattr(model, 'exposures', None)
        if not exposures or model.meta.exposure.type != 'NRS_MSASPEC':
            raise ConfigError(
                f"{filename!r} is not a cal file of the exposures of a "
                f"source of the micro-shutter array of NIRSpec")
        slitlets = [_NirspecSlitlet(exposure) for exposure in exposures]
        return [
            _slitlet_spectra(exposure, slitlets, i, undo_corrections)
            for i, exposure in enumerate(exposures)]


# The nodes of the trace of a slitlet of NIRSpec: about every this many
# pixels of the detector along the wavelength, and every this many pixels
# of the places, to this many beyond the open shutters (the trace is of
# the light through them)
_NIRSPEC_WAVE_NODE_PIXELS = 4
_NIRSPEC_PLACE_NODE_PIXELS = 0.5
_NIRSPEC_PLACE_MARGIN = 1

# The columns at each end of the cutout of a slitlet of NIRSpec beyond the
# wavelengths of its trace
_NIRSPEC_END_COLUMNS = 4


class _NirspecSlitlet:
    """
    The geometry of a slitlet of NIRSpec in an exposure, from its WCS: its
    slit frame (x across the shutters, y along the slitlet, in units of
    the open size of a shutter, from the centre of the shutter of the
    source), on the sky (in arcsec of the frame of the model: x to the
    west, y to the north, from the centre of the shutter of the source)
    and on the detector (pixels of the cutout).
    """

    def __init__(self, exposure: Any):
        wcs = exposure.meta.wcs
        self.to_world = wcs.get_transform('slit_frame', 'world')
        self.from_world = wcs.get_transform('world', 'slit_frame')
        self.to_detector = wcs.get_transform('slit_frame', 'detector')
        self.to_msa = wcs.get_transform('slit_frame', 'msa_frame')
        # (the wavelengths of the cutout, but of its last columns: the light
        # from across the shutters lands beyond them)
        ny, nx = exposure.data.shape
        y, x = np.mgrid[:ny, _NIRSPEC_END_COLUMNS:nx - _NIRSPEC_END_COLUMNS]
        wave = wcs.get_transform('detector', 'slit_frame')(x, y)[2]
        self.wave = np.nanmin(wave), np.nanmax(wave)
        self.wave_ref = float(np.mean(self.wave))
        ra, dec, self.world_wave = self.to_world(0.0, 0.0, self.wave_ref)
        self.origin = astropy.coordinates.SkyCoord(
            ra * astropy.units.deg, dec * astropy.units.deg)
        # (the arcsec of a unit of x and of y of the slit frame)
        self.axes = np.column_stack(
            [self.sky(1.0, 0.0), self.sky(0.0, 1.0)])
        self.state = exposure.shutter_state
        self.yscale = exposure.slit_yscale

    def sky(self, x: npt.ArrayLike, y: npt.ArrayLike) -> np.ndarray:
        """
        Return the places on the sky (arcsec) of places of the slit frame,
        of shape (2, ...).
        """
        ra, dec, _ = self.to_world(x, y, self.wave_ref)
        east, north = self.origin.spherical_offsets_to(
            astropy.coordinates.SkyCoord(
                ra * astropy.units.deg, dec * astropy.units.deg))
        return np.stack([-east.to_value('arcsec'), north.to_value('arcsec')])

    def slit(self, sky: np.ndarray) -> np.ndarray:
        """
        Return the places of the slit frame of places on the sky (arcsec,
        of shape (2, ...)): of the inverse of the WCS, refined by a step of
        Newton (the inverse of the WCS is itself approximate).
        """
        world = self.origin.spherical_offsets_by(
            -sky[0] * astropy.units.arcsec, sky[1] * astropy.units.arcsec)
        x, y, _ = self.from_world(
            world.ra.deg, world.dec.deg, self.world_wave)
        slit = np.stack([x, y])
        residual = sky - self.sky(x, y)
        return slit + np.tensordot(
            np.linalg.inv(self.axes), residual, axes=1)

    def msa_to_slit(self, msa: np.ndarray) -> np.ndarray:
        """
        Return the places of the slit frame of places in the plane of the
        shutters (of the affine map between them), of shape (2, ...).
        """
        corners = [np.array(self.to_msa(x, y, self.wave_ref)[:2])
                   for x, y in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))]
        matrix = np.column_stack(
            [corners[1] - corners[0], corners[2] - corners[0]])
        return np.tensordot(
            np.linalg.inv(matrix), msa - corners[0][:, None], axes=1)


def nirspec_msa_rectified(
        s2d_filename: str, cal_filename: str, undo_corrections: bool = True
) -> SlitletSpectra:
    """
    Read the rectified spectra of a source in a slitlet of the
    micro-shutter array of JWST NIRSpec: the s2d file of the source of
    stage 3 of the pipeline (its exposures resampled onto the sky by
    drizzle and combined), with the model of the instrument of the cal
    file of the source (see nirspec_msa_spectra; needs the package
    stdatamodels: the extra jwst).

    The rectified spectra are the image of slitless spectra behind the
    shutters, resampled. Its grid is that of the s2d (columns along the
    wavelength, rows along the slit), with the world coordinates of the
    undispersed light of the first exposure (see nirspec_msa_spectra;
    with pixels along the slit of those of the s2d). The trace takes the
    light of each wavelength at each place of the shutters to the
    detector (by the model of the instrument), then to the s2d as the
    pipeline did (by the WCS of the exposure, which takes the light to
    come from the centre of the shutter, or from a point source, and that
    of the s2d). The exposures are those of the cal file, each of the sky
    moved to where it was in it and its image moved back (the pipeline
    aligned them on the sky), weighted by their exposure times; and, if
    the pipeline subtracted the background of each with the others, those
    of the others, of weights -1 / (n - 1) of the exposure.

    The correction of the pipeline for the path losses through the
    shutters (pathloss), a function of the wavelength, is undone at the
    wavelengths of the s2d, unless undo_corrections is False; that for
    the bars between them (barshadow) cannot be undone after resampling.

    Parameters
    ----------
    s2d_filename : str
        The s2d file of the source (e.g.
        jw01210-o001_s000009382_nirspec_f290lp-g395m_s2d.fits).
    cal_filename : str
        The cal file of the source (see nirspec_msa_spectra).
    undo_corrections : bool, optional
        Whether to undo the correction for the path losses.

    Returns
    -------
    SlitletSpectra
        The spectra (resampled).

    Raises
    ------
    ConfigError
        If the files are not of a source of the micro-shutter array, the
        bar shadows were corrected, or a slitlet has closed shutters
        between open ones or no shutter of the source.
    """
    from gbkfit.dataset import Data, DatasetSlitlessSpectra
    from gbkfit.observation.observables.slitless_spectra import Exposure
    from stdatamodels.jwst import datamodels
    with datamodels.open(cal_filename) as cal, \
            datamodels.open(s2d_filename) as s2d:
        exposures = getattr(cal, 'exposures', None)
        if not exposures or cal.meta.exposure.type != 'NRS_MSASPEC' \
                or s2d.meta.exposure.type != 'NRS_MSASPEC':
            raise ConfigError(
                f"{cal_filename!r} and {s2d_filename!r} are not the cal and "
                f"s2d files of a source of the micro-shutter array of "
                f"NIRSpec")
        if s2d.barshadow_corrected:
            raise ConfigError(
                "the bar shadows of rectified spectra cannot be undone; "
                "reduce them without the step barshadow")
        slitlets = [_NirspecSlitlet(exposure) for exposure in exposures]
        first = slitlets[0]
        data, error = s2d.data.copy(), s2d.err.copy()
        if undo_corrections and s2d.meta.cal_step.pathloss == 'COMPLETE':
            factor = _nirspec_pathloss_at(exposures[0], s2d.wavelength)
            data *= factor
            error *= factor
        mask = (s2d.wht > 0) & np.isfinite(data) & np.isfinite(error)
        to_s2d = s2d.meta.wcs.get_transform('world', 'detector')
        from_detector = exposures[0].meta.wcs.get_transform(
            'detector', 'world')

        def rectified(x: npt.ArrayLike, y: npt.ArrayLike,
                      wave: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
            # (of places of the slit frame of the first exposure)
            detector = first.to_detector(*np.broadcast_arrays(x, y, wave))
            return to_s2d(*from_detector(*detector))
        geometry = _nirspec_image(first, rectified)
        rpix, step, rota = geometry
        aperture = _nirspec_shutters(first)
        trace = _nirspec_trace(
            first, s2d.data.shape, aperture, *geometry, rectified)
        times = np.array([e.meta.exposure.exposure_time for e in exposures])
        weights = times / times.sum()
        nods = [_nirspec_nod(first, other, rota) for other in slitlets]
        combined = []
        for k, (nod, weight) in enumerate(zip(nods, weights)):
            back = (-nod[0], -nod[1])
            combined.append(Exposure(nod, back, weight))
            if exposures[k].meta.cal_step.bkg_subtract == 'COMPLETE':
                combined += [
                    Exposure(other, back, -weight / (len(nods) - 1))
                    for j, other in enumerate(nods) if j != k]
        dataset = DatasetSlitlessSpectra(
            Data(data, mask, error, s2d.meta.bunit_data),
            step=step, rpix=tuple(rpix),
            rval=(first.origin.ra.deg, first.origin.dec.deg), rota=rota)
        return SlitletSpectra(
            dataset, trace, aperture, tuple(combined), True)


def _nirspec_pathloss_at(
        exposure: Any, wavelength: np.ndarray
) -> np.ndarray:
    """
    Return the correction of the pipeline for the path losses of the light
    of the source of an exposure (of its type, a function of the
    wavelength) at wavelengths (of the WCS of the pipeline).
    """
    correction = exposure.pathloss_point \
        if exposure.pathloss_correction_type == 'POINT' \
        else exposure.pathloss_uniform
    wave = np.nanmedian(exposure.wavelength, axis=0)
    values = np.nanmedian(correction, axis=0)
    known = np.isfinite(wave) & np.isfinite(values)
    order = np.argsort(wave[known])
    return np.interp(
        wavelength, wave[known][order], values[known][order],
        left=np.nan, right=np.nan)


def _slitlet_spectra(
        exposure: Any,
        slitlets: list[_NirspecSlitlet],
        index: int,
        undo_corrections: bool
) -> SlitletSpectra:
    """
    Return the spectra of an exposure of a source in a slitlet of NIRSpec
    (see nirspec_msa_spectra), of the exposure of the given index among
    those of the geometries of slitlets.
    """
    from gbkfit.dataset import Data, DatasetSlitlessSpectra
    from gbkfit.observation.observables.slitless_spectra import Exposure
    slitlet = slitlets[index]
    data, error = exposure.data.copy(), exposure.err.copy()
    if undo_corrections:
        for factor in _nirspec_loss_corrections(exposure):
            data *= factor
            error *= factor
    mask = ((exposure.dq & 1) == 0) & np.isfinite(data) & np.isfinite(error)
    geometry = _nirspec_image(slitlet, slitlet.to_detector)
    rpix, step, rota = geometry
    aperture = _nirspec_shutters(slitlet)
    trace = _nirspec_trace(
        slitlet, exposure.data.shape, aperture, *geometry,
        slitlet.to_detector)
    exposures = (Exposure(),)
    if exposure.meta.cal_step.bkg_subtract == 'COMPLETE':
        exposures += tuple(
            Exposure(_nirspec_nod(slitlet, other, rota),
                     weight=-1 / (len(slitlets) - 1))
            for other in slitlets if other is not slitlet)
    dataset = DatasetSlitlessSpectra(
        Data(data, mask, error, exposure.meta.bunit_data),
        step=step, rpix=tuple(rpix),
        rval=(slitlet.origin.ra.deg, slitlet.origin.dec.deg), rota=rota)
    return SlitletSpectra(dataset, trace, aperture, exposures)


def _nirspec_image(
        slitlet: _NirspecSlitlet,
        to_image: Callable[..., tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, tuple[float, float], float]:
    """
    Return the world coordinates of the undispersed light of an image of
    the spectra of a slitlet of NIRSpec, whose pixels the light of the
    places of the slit frame lands on by to_image (x, y, wavelength): its
    reference pixel, where the centre of the shutter of the source lands
    at the middle wavelength; its pixel sizes, by which the light moves a
    pixel there; and its rotation, of y against that of the slit, x 90
    degrees clockwise of it.
    """
    ax, ay = slitlet.axes.T
    rota = float(np.degrees(np.arctan2(ay[0], -ay[1])))
    rpix = np.array(to_image(0.0, 0.0, slitlet.wave_ref), dtype=float)

    def moved(x: float, y: float) -> float:
        return np.hypot(*(np.array(
            to_image(x, y, slitlet.wave_ref), dtype=float) - rpix))
    step = (float(np.hypot(*ax) * 0.01 / moved(0.01, 0.0)),
            float(np.hypot(*ay) * 0.01 / moved(0.0, 0.01)))
    return rpix, step, rota


def _nirspec_loss_corrections(exposure: Any) -> list[np.ndarray]:
    """
    Return the corrections of the pipeline for the losses through the
    shutters that it divided the data of an exposure by (pathloss, of the
    type of the source, and barshadow), those it applied.
    """
    corrections = []
    if exposure.meta.cal_step.pathloss == 'COMPLETE':
        corrections.append(
            exposure.pathloss_point
            if exposure.pathloss_correction_type == 'POINT'
            else exposure.pathloss_uniform)
    if exposure.barshadow_corrected:
        corrections.append(exposure.barshadow)
    return corrections


def _on_sky(image: np.ndarray, rota: float) -> np.ndarray:
    """
    Return places along the axes of an image of the rotation rota (arcsec,
    of shape (2, ...)) on the sky, in the frame of the model (see
    gridutils.sky_positions).
    """
    angle = np.radians(rota)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.stack([cos * image[0] - sin * image[1],
                     sin * image[0] + cos * image[1]])


def _along_image(sky: np.ndarray, rota: float) -> np.ndarray:
    """Return places on the sky along the axes of an image (see _on_sky)."""
    return _on_sky(sky, -rota)


def _nirspec_trace(
        slitlet: _NirspecSlitlet,
        shape: tuple[int, int],
        aperture: 'ApertureShutters',
        rpix: np.ndarray,
        step: tuple[float, float],
        rota: float,
        to_image: Callable[..., tuple[np.ndarray, np.ndarray]]
) -> 'TraceTable':
    """
    Return the trace of a slitlet of NIRSpec on an image of its spectra
    (of the given shape and world coordinates of the undispersed light),
    through its open shutters (the aperture): the offsets of the light of
    each wavelength at each place of the image, from its place on the sky
    to the slit frame, and from there to the image by to_image (x, y,
    wavelength).
    """
    from gbkfit.observation.observables.traces import TraceTable
    nx = shape[1]
    wave = np.linspace(
        *slitlet.wave, max(4, nx // _NIRSPEC_WAVE_NODE_PIXELS))
    # (the places of the open shutters, and a margin)
    rectangles = aperture.rectangles()
    x, y = (
        np.arange(np.floor(rpix[axis] + edges.min() / step[axis])
                  - _NIRSPEC_PLACE_MARGIN,
                  np.ceil(rpix[axis] + edges.max() / step[axis])
                  + _NIRSPEC_PLACE_MARGIN + _NIRSPEC_PLACE_NODE_PIXELS / 2,
                  _NIRSPEC_PLACE_NODE_PIXELS)
        for axis, edges in ((0, rectangles[:, :2]), (1, rectangles[:, 2:])))
    yy, xx = np.meshgrid(y, x, indexing='ij')
    slit_x, slit_y = slitlet.slit(_on_sky(np.stack(
        [(xx - rpix[0]) * step[0], (yy - rpix[1]) * step[1]]), rota))
    # (the transforms of jwst take inputs of one shape)
    image_x, image_y = to_image(*np.broadcast_arrays(
        slit_x[None], slit_y[None], wave[:, None, None]))
    dx, dy = np.asarray(image_x) - xx, np.asarray(image_y) - yy
    if not (np.all(np.isfinite(dx)) and np.all(np.isfinite(dy))):
        raise ConfigError(
            "the model of the instrument does not reach the detector from "
            "every place and wavelength of the slitlet")
    return TraceTable(wave, x, y, dx, dy)


def _nirspec_shutters(slitlet: _NirspecSlitlet) -> 'ApertureShutters':
    """
    Return the open shutters of a slitlet of NIRSpec, along the axes of
    its image, from the centre of the shutter of the source: of the open
    size and the pitch of its shutters along the slit (y, against that of
    the slit frame).
    """
    from gbkfit.observation.observables.apertures import ApertureShutters
    state = slitlet.state
    if 'x' not in state or set(state) - {'1', 'x'}:
        raise ConfigError(
            f"the slitlet of the shutters {state!r} has closed shutters "
            f"between open ones, or none of the source")
    width, length = np.hypot(*slitlet.axes)
    pitch = slitlet.yscale * length
    middle = (len(state) - 1) / 2 - state.index('x')
    return ApertureShutters(
        size=(width, length), pitch=pitch, count=len(state),
        centre=(0.0, -middle * pitch))


def _nirspec_nod(
        slitlet: _NirspecSlitlet, other: _NirspecSlitlet, rota: float
) -> tuple[float, float]:
    """
    Return the offset of the sky (arcsec along the axes of the image of
    the rotation rota of a slitlet) in another exposure of it: where the
    place of the sky at the centre of the shutter of the source lands in
    the other exposure, in the slit frame of this one, through the plane
    of the shutters.
    """
    east, north = other.origin.spherical_offsets_to(slitlet.origin)
    centre = other.slit(np.array(
        [-east.to_value('arcsec'), north.to_value('arcsec')]))
    msa = np.array(other.to_msa(*centre, other.wave_ref)[:2])
    offset = slitlet.axes @ slitlet.msa_to_slit(msa[:, None])[:, 0]
    return tuple(float(v) for v in _along_image(offset, rota))
