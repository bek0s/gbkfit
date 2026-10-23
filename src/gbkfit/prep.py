"""
Data preparation: data read from FITS files made ready for the datasets
of gbkfit (cropped, masked, clipped, in the axis order and direction of
the model, and with velocity spectral axes), the binning of maps and
cubes for region datasets (and Voronoi bins of a map to a target
signal-to-noise), and the tables of the PSFs and LSFs of instruments.
The prep task of gbkfit-cli is a client of this module.
"""

import dataclasses
import logging
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import astropy.constants
import astropy.coordinates
import astropy.io.fits as fits
import astropy.stats
import astropy.table
import astropy.units
import astropy.wcs
import astropy.wcs.utils
import numpy as np
import numpy.typing as npt
import scipy.ndimage
import scipy.spatial
import skimage.measure

from gbkfit.utils import fitsutils, gridutils, unitutils
from gbkfit.utils.parseutils import ConfigError

if TYPE_CHECKING:
    from gbkfit.dataset import DatasetSlitlessSpectra
    from gbkfit.observation.observables.apertures import ApertureShutters
    from gbkfit.observation.observables.slitless_spectra import Exposure
    from gbkfit.observation.observables.traces import TraceTable


__all__ = [
    'FitsArrays',
    'SlitletSpectra',
    'bin_index',
    'bin_spectra',
    'bin_values',
    'bounding_box',
    'grown_mask',
    'largest_regions_mask',
    'muse_lsf_table',
    'muse_psf_table',
    'nirspec_msa_rectified',
    'nirspec_msa_spectra',
    'noise_errors',
    'pixel_fluxes',
    'range_mask',
    'sigma_clip_mask',
    'subtract_continuum',
    'voronoi_bins'
]


_log = logging.getLogger(__name__)

# The keywords of the world coordinates of each axis, or pair of axes
_WCS_KEYWORDS = re.compile(
    r'(WCSAXES'
    r'|(CRPIX|CRVAL|CDELT|CUNIT|CTYPE|CROTA)\d+'
    r'|(PC|CD|PV|PS)\d+_\d+)$')

# The keywords of the rest of the spectral axis, also in their old forms
_REST_KEYWORDS = ('RESTWAV', 'RESTFRQ', 'RESTWAVE', 'RESTFREQ')

# The wavelengths (Angstrom) of the tables of MUSE (its range, and more)
_MUSE_WAVELENGTHS = np.arange(4600.0, 9401.0, 50.0)

# The bin accretion of Voronoi binning (Cappellari & Copin 2003): a bin
# stops growing above this roundness, and is kept if its S/N reaches this
# fraction of the target (the pixels of the others join the nearest bin)
_VORONOI_MAX_ROUNDNESS = 0.3
_VORONOI_KEPT_FRACTION = 0.8

# The most iterations of the tessellation of Voronoi binning
_VORONOI_MAX_ITERATIONS = 1000


def _has_wcs(header: fits.Header) -> bool:
    """Return whether a header has world coordinates."""
    return any(_WCS_KEYWORDS.match(key) for key in header)


def _with_wcs(header: fits.Header, wcs: astropy.wcs.WCS) -> fits.Header:
    """
    Return a copy of a header with the world coordinates of wcs, and its
    rest (if any).
    """
    header = header.copy()
    for key in [
            key for key in header
            if _WCS_KEYWORDS.match(key) or key in _REST_KEYWORDS]:
        del header[key]
    header.update(wcs.to_header())
    return header


def _drop_axes(header: fits.Header, axes: Sequence[int]) -> fits.Header:
    """
    Return the header of the data without the given axes (numpy axes, of
    length 1), without the world coordinates of these axes.
    """
    if not _has_wcs(header):
        return header
    wcs = astropy.wcs.WCS(header)
    for axis in sorted(wcs.naxis - 1 - axis for axis in axes)[::-1]:
        wcs = wcs.dropaxis(axis)
    return _with_wcs(header, wcs)


def _shift_axes(header: fits.Header, offset: npt.ArrayLike) -> fits.Header:
    """
    Return the header of the data whose first pixel is the pixel at the
    given offset (on each numpy axis) of the original data: positive where
    the data were cropped, negative where they were padded.
    """
    if not _has_wcs(header):
        return header
    wcs = astropy.wcs.WCS(header)
    wcs.wcs.crpix -= np.asarray(offset)[::-1]
    return _with_wcs(header, wcs)


def _check_same_rest(wcs: astropy.wcs.WCS, rest: astropy.units.Quantity
                     ) -> None:
    """
    Raise ConfigError unless the velocity axis of wcs has no rest, or the
    given one (they are compared as frequencies).
    """
    hz, spectral = astropy.units.Hz, astropy.units.spectral()
    current = (
        wcs.wcs.restfrq * hz if wcs.wcs.restfrq > 0
        else wcs.wcs.restwav * astropy.units.m if wcs.wcs.restwav > 0
        else None)
    if current is None:
        return
    current = current.to_value(hz, spectral)
    if not np.isclose(rest.to_value(hz, spectral), current, rtol=1e-9):
        raise ConfigError(
            f"the velocities of the spectral axis refer to the rest "
            f"{current * hz}, not {rest}; give that rest, or convert the "
            f"velocities to the new rest first")


def _spectral_to_velocity(
        header: fits.Header,
        rest: str | astropy.units.Quantity
) -> fits.Header:
    """
    Return the header with its spectral axis converted to the velocities
    of a rest (see gridutils.make_rest), or an axis of logarithmic
    wavelengths (WAVE-LOG) with the rest, which the reading of the data
    takes as log velocities: a linear wavelength axis (WAVE, or
    AWAV with a rest in air too) to the optical velocities c (w / rest -
    1), a linear frequency axis (FREQ) to the radio velocities c (1 - f /
    rest). Both are linear in the pixels, so the conversion is exact; the
    header gets the rest (RESTWAV or RESTFRQ), and for wavelengths their
    medium, in which the velocities and the rest are (AIRORVAC: 'air' for
    AWAV, 'vac' for WAVE; see fitsutils.read_data). A velocity axis (VOPT,
    VRAD) gets the rest only, and if it has one, it must be the same: its
    velocities refer to it.
    """
    rest = gridutils.make_rest(rest)
    wcs = astropy.wcs.WCS(header)
    s = wcs.wcs.spec
    if s < 0:
        raise ConfigError("the data have no spectral axis to convert")
    ctype = wcs.wcs.ctype[s]
    kind = ctype[:4]
    if ctype == 'WAVE-LOG':
        # (logarithmic wavelengths stay: they are read as the log
        # velocities of the rest, linear, see fitsutils.read_data)
        wcs.wcs.restwav = rest.to_value(
            astropy.units.m, astropy.units.spectral())
        wcs.wcs.restfrq = 0
        _log.info(f"the spectral axis ({ctype}) gets the rest {rest}")
        header = _with_wcs(header, wcs)
        header['AIRORVAC'] = 'vac'
        return header
    if ctype[4:].strip('-').strip():
        raise ConfigError(
            f"the spectral axis ({ctype}) is not linear in its coordinate")
    c = astropy.constants.c.to_value('km/s')
    m, hz = astropy.units.m, astropy.units.Hz
    rest_wavelength = rest.to_value(m, astropy.units.spectral())
    rest_frequency = rest.to_value(hz, astropy.units.spectral())
    unit = astropy.units.Unit(str(wcs.wcs.cunit[s]) or '1')
    if kind in ('WAVE', 'AWAV'):
        # The wavelength in units of the rest, and the optical velocity
        ratio = unit.to(m) / rest_wavelength
        crval, factor = c * (wcs.wcs.crval[s] * ratio - 1), c * ratio
        velocity = 'VOPT'
    elif kind == 'FREQ':
        ratio = unit.to(hz) / rest_frequency
        crval, factor = c * (1 - wcs.wcs.crval[s] * ratio), -c * ratio
        velocity = 'VRAD'
    elif kind in ('VOPT', 'VRAD'):
        crval, factor, velocity = None, None, kind
        _check_same_rest(wcs, rest)
    else:
        raise ConfigError(
            f"the spectral axis ({ctype}) is not a wavelength, frequency, "
            f"or optical or radio velocity")
    if crval is not None:
        wcs.wcs.ctype[s] = velocity
        wcs.wcs.cunit[s] = 'km/s'
        wcs.wcs.crval[s] = crval
        if wcs.wcs.has_cd():
            cd = wcs.wcs.cd.copy()
            cd[s, :] *= factor
            wcs.wcs.cd = cd
        else:
            wcs.wcs.cdelt[s] *= factor
    if velocity == 'VOPT':
        wcs.wcs.restwav, wcs.wcs.restfrq = rest_wavelength, 0
    else:
        wcs.wcs.restfrq, wcs.wcs.restwav = rest_frequency, 0
    _log.info(
        f"converting the spectral axis ({ctype}) to {velocity} velocities "
        f"of the rest {rest}")
    header = _with_wcs(header, wcs)
    if kind in ('WAVE', 'AWAV'):
        header['AIRORVAC'] = 'air' if kind == 'AWAV' else 'vac'
    return header


def _first_hdu(filename: str) -> tuple[np.ndarray, fits.Header]:
    """
    Return the data and the header of the first HDU of a file with data
    (e.g. the extension after an empty primary HDU, as in MUSE and JWST
    cubes). Integer data become floats, which can hold the NaN of invalid
    pixels.
    """
    with fits.open(filename) as hdus:
        hdu = next((hdu for hdu in hdus if hdu.data is not None), None)
        if hdu is None:
            raise ConfigError(f"{filename}: the file has no data")
        data = np.array(hdu.data)
        header = hdu.header.copy()
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(float)
    return data, header


def _read_array(filename: str) -> np.ndarray:
    """
    Return the data of a file without the axes of length 1 (see
    _read_fits), without its header: those of errors and masks, which
    follow the axes of the data.
    """
    return np.squeeze(_first_hdu(filename)[0])


def _read_fits(
        filename: str, velocity: str | None = None
) -> tuple[np.ndarray, fits.Header]:
    """
    Return the data and the header of a file, without the axes of length
    1 (see _first_hdu), with the convention of the velocities of its
    spectral axis, if its header does not state it (velocity, see
    fitsutils.with_velocity_convention).
    """
    data, header = _first_hdu(filename)
    header = fitsutils.with_velocity_convention(header, velocity, filename)
    # Astropy writes the FITS default for a missing reference pixel, so
    # write the one of the model (see fitsutils.read_data)
    if _has_wcs(header):
        header = fitsutils.centre_missing_crpix(header, data.shape)
    axes = [axis for axis, length in enumerate(data.shape) if length == 1]
    data = np.squeeze(data)
    # The header of the squeezed data: its NAXISn, and the world
    # coordinates of its axes
    header = fits.PrimaryHDU(data, _drop_axes(header, axes)).header
    return data, header


@dataclasses.dataclass(frozen=True)
class FitsArrays:
    """
    Data read from FITS files, being prepared for a dataset: the data,
    their errors (if any) and mask (1 where valid), with the header of the
    data (their world coordinates, if any). The data and errors are NaN
    where the mask is 0. The axes are in the order of the model (e.g. RA
    before Dec; see fitsutils.model_axis_order).

    The methods return new arrays; those that move the first pixel
    (crop, minified, padded) keep the world coordinates of the pixels.

    Attributes
    ----------
    data : np.ndarray
        The data.
    error : np.ndarray or None
        The errors.
    mask : np.ndarray
        The mask: 1 where valid, 0 elsewhere.
    header : fits.Header
        The header of the data.
    """
    data: np.ndarray
    error: np.ndarray | None
    mask: np.ndarray
    header: fits.Header

    @classmethod
    def read(
            cls,
            data: str,
            error: str | None = None,
            mask: str | None = None,
            velocity: str | None = None
    ) -> 'FitsArrays':
        """
        Read the data, errors and mask from FITS files (their first HDU
        with data), without their axes of length 1, in the axis order of
        the model. Values that are not finite, errors that are not finite
        or are 0 (e.g. blank pixels of an error map), and pixels the mask
        file marks with 0 are invalid; negative errors are an error. A
        spectral axis of velocities of the AIPS convention without VELREF
        (e.g. VELO-HEL) needs the convention of its velocities (see
        fitsutils.with_velocity_convention).

        Parameters
        ----------
        data : str
            The file of the data (with the world coordinates).
        error, mask : str, optional
            The files of the errors and the mask, of the shape of the data.
        velocity : str, optional
            The convention of the velocities of the spectral axis, 'radio'
            or 'optical', where the header does not state it.

        Returns
        -------
        FitsArrays
            The arrays.

        Raises
        ------
        ConfigError
            If a file has no data, the shapes differ, an error is
            negative, or the convention of the velocities is ambiguous and
            not given, or given and stated by the header.
        """
        data_d, header = _read_fits(data, velocity)
        data_e = _read_array(error) if error else None
        data_m = _read_array(mask) if mask else None
        for name, array in (('error', data_e), ('mask', data_m)):
            if array is not None and array.shape != data_d.shape:
                raise ConfigError(
                    f"the {name} must have the shape of the data "
                    f"{data_d.shape}; it has {array.shape}")
        # (the error and the mask follow the axes of the data, whose world
        # coordinates they may not have)
        if _has_wcs(header):
            order = fitsutils.model_axis_order(astropy.wcs.WCS(header))
            if order != tuple(range(data_d.ndim)):
                _log.info("reordering the axes in the order of the model")
                wcs = fitsutils.reorder_wcs(astropy.wcs.WCS(header), order)
                header = _with_wcs(header, wcs)
                data_d = fitsutils.reorder_data(data_d, order)
                if data_e is not None:
                    data_e = fitsutils.reorder_data(data_e, order)
                if data_m is not None:
                    data_m = fitsutils.reorder_data(data_m, order)
        valid = np.isfinite(data_d)
        if data_e is not None:
            if np.any(data_e < 0):
                raise ConfigError(
                    f"errors cannot be negative; "
                    f"{np.count_nonzero(data_e < 0)} are, down to "
                    f"{np.nanmin(data_e):g}")
            valid &= np.isfinite(data_e) & (data_e > 0)
        if data_m is not None:
            valid &= np.isfinite(data_m) & (data_m != 0)
        return cls(
            np.ascontiguousarray(data_d),
            None if data_e is None else np.ascontiguousarray(data_e),
            np.ones(data_d.shape), header).masked(valid)

    def masked(self, mask: npt.ArrayLike) -> 'FitsArrays':
        """
        Return the arrays with the pixels a mask marks invalid (0 or
        False) also invalid.
        """
        valid = (self.mask != 0) & (np.asarray(mask) != 0)
        data = np.where(valid, self.data, np.nan)
        error = None if self.error is None \
            else np.where(valid, self.error, np.nan)
        return FitsArrays(data, error, valid.astype(float), self.header)

    def crop(self, slices: Sequence[slice]) -> 'FitsArrays':
        """
        Return the arrays cropped to ranges of pixels (one slice per numpy
        axis, of step 1).
        """
        slices = tuple(slices)
        offset = [s.indices(n)[0] for s, n in zip(slices, self.data.shape)]
        return FitsArrays(
            self.data[slices],
            None if self.error is None else self.error[slices],
            self.mask[slices], _shift_axes(self.header, offset))

    def minified(self) -> 'FitsArrays':
        """
        Return the arrays cropped to their valid pixels (see
        bounding_box).

        Raises
        ------
        ConfigError
            If no pixel is valid.
        """
        return self.crop(bounding_box(self.mask))

    def padded(self, size: int) -> 'FitsArrays':
        """Return the arrays padded with invalid pixels on every side."""
        def pad(array: np.ndarray, value: float) -> np.ndarray:
            return np.pad(array, size, 'constant', constant_values=value)
        return FitsArrays(
            pad(self.data, np.nan),
            None if self.error is None else pad(self.error, np.nan),
            pad(self.mask, 0),
            _shift_axes(self.header, [-size] * self.data.ndim))

    def with_velocity_axis(
            self, rest: str | astropy.units.Quantity
    ) -> 'FitsArrays':
        """
        Return the arrays with their spectral axis in velocities of a rest
        (a wavelength or frequency; see gridutils.make_rest): optical
        velocities from wavelengths, radio velocities from frequencies.

        Raises
        ------
        ConfigError
            If there is no linear spectral axis, or its velocities refer to
            another rest.
        """
        return dataclasses.replace(
            self, header=_spectral_to_velocity(self.header, rest))

    def write(
            self,
            data: str,
            error: str | None = None,
            mask: str | None = None,
            dtype: npt.DTypeLike = np.float32
    ) -> None:
        """
        Write the data, errors and mask (if given filenames) as FITS
        files, with the header of the data. The axes that the model needs
        reversed (see fitsutils.model_axis_flips: that of a mirrored
        image, and a spectral axis of decreasing velocities) are reversed.

        Parameters
        ----------
        data : str
            The file of the data.
        error, mask : str, optional
            The files of the errors and the mask.
        dtype : DTypeLike, optional
            The type of the values.
        """
        arrays = [self.data, self.error, self.mask]
        header = self.header
        flips = fitsutils.model_axis_flips(astropy.wcs.WCS(header)) \
            if _has_wcs(header) else ()
        for axis in flips:
            _log.info(
                f"reversing the axis {axis + 1}: the image is mirrored, or "
                f"the velocity decreases along it")
            size = self.data.shape[self.data.ndim - 1 - axis]
            header = _with_wcs(header, fitsutils.flip_wcs(
                astropy.wcs.WCS(header), axis, size))
            arrays = [
                None if a is None else fitsutils.flip_data(a, axis)
                for a in arrays]
        for filename, array in zip((data, error, mask), arrays):
            if filename is not None and array is not None:
                fits.writeto(
                    filename, array.astype(dtype), header, overwrite=True)


def range_mask(
        data: np.ndarray,
        minimum: float | None = None,
        maximum: float | None = None
) -> np.ndarray:
    """
    Return the mask of the values in a range (True where kept); NaN values
    are left to the other masks (kept here).
    """
    with np.errstate(invalid='ignore'):
        kept = np.ones(data.shape, bool)
        if minimum is not None:
            kept &= ~(data < minimum)
        if maximum is not None:
            kept &= ~(data > maximum)
    return kept


def sigma_clip_mask(
        data: np.ndarray,
        mask: np.ndarray,
        sigma: float,
        maxiters: int | None = 5
) -> np.ndarray:
    """
    Return the mask of the values that sigma clipping keeps (True where
    kept), of those a mask keeps (the others do not count in its
    statistics, and are not kept).
    """
    kept = np.ma.masked_array(data, mask=(mask == 0))
    return ~astropy.stats.sigma_clip(
        kept, sigma=sigma, maxiters=maxiters).mask


def noise_errors(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Return the errors of the values of a cube from its noise: for each
    channel, the robust dispersion of its negative values (of those the
    mask keeps), 1.4826 times the median of their absolute values, which
    emission (positive) does not bias, unlike the dispersion of all the
    values, even sigma-clipped (its faint wings stay); NaN for channels
    with fewer than 5 negative values. The noise must be of mean 0 (e.g.
    of a cube without continuum, as the models are).

    Parameters
    ----------
    data : np.ndarray
        The cube, of shape (nz, ny, nx).
    mask : np.ndarray
        The mask of the values that count (nonzero).

    Returns
    -------
    np.ndarray
        The errors, of the shape of the cube: the same at every pixel of
        a channel.
    """
    errors = np.full(data.shape[0], np.nan)
    for channel, (values, kept) in enumerate(zip(data, mask)):
        negative = values[(kept != 0) & (values < 0)]
        if negative.size >= 5:
            errors[channel] = 1.4826 * np.median(-negative)
    return np.broadcast_to(errors[:, None, None], data.shape).copy()


def grown_mask(
        snr: np.ndarray,
        seed: float,
        grow: float
) -> np.ndarray:
    """
    Return the mask (True where kept) grown from seeds: the values whose
    signal-to-noise reaches grow and that connect (through neighbours
    along the axes) to one that reaches seed (seed >= grow), as the masks
    of SoFiA, 3D Barolo or qubefit keep faint emission next to bright
    emission and drop isolated noise. Of a map or a cube (or any number of
    axes); NaN values are not kept.

    Raises
    ------
    ValueError
        If seed is below grow.
    """
    if seed < grow:
        raise ValueError(
            f"the seed threshold ({seed}) must be at least the threshold "
            f"of growth ({grow})")
    with np.errstate(invalid='ignore'):
        seeds, allowed = snr >= seed, snr >= grow
    return scipy.ndimage.binary_propagation(seeds, mask=allowed)


def subtract_continuum(
        data: np.ndarray,
        line: np.ndarray,
        degree: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a cube less its continuum, and the continuum: of each spectrum,
    the polynomial of the given degree in the channel fitted by least
    squares to its channels outside the line (and with values; NaN where
    a spectrum has fewer of them than the degree needs).

    Parameters
    ----------
    data : np.ndarray
        The cube, of shape (nz, ny, nx).
    line : np.ndarray
        The channels of the line (True; nz), which the fit leaves out.
    degree : int, optional
        The degree of the polynomial (0: a constant).

    Returns
    -------
    tuple of np.ndarray
        The cube less its continuum, and the continuum (both of the shape
        of the cube).

    Raises
    ------
    ValueError
        If line is not one value per channel, or the degree is negative.
    """
    line = np.asarray(line, dtype=bool)
    if line.shape != data.shape[:1] or degree < 0:
        raise ValueError(
            "line must have one value per channel, and the degree be at "
            "least 0")
    nz = data.shape[0]
    spectra = data.reshape(nz, -1)
    # (the channels about the middle, for a well-conditioned fit)
    z = np.arange(nz) - (nz - 1) / 2
    design = np.vander(z, degree + 1)
    continuum = np.full(spectra.shape, np.nan)
    usable = ~line[:, None] & np.isfinite(spectra)
    # (the spectra of the same usable channels share one fit)
    patterns, which = np.unique(usable.T, axis=0, return_inverse=True)
    for k, pattern in enumerate(patterns):
        columns = np.flatnonzero(which.ravel() == k)
        if pattern.sum() <= degree:
            continue
        coefficients, *_ = np.linalg.lstsq(
            design[pattern], spectra[pattern][:, columns], rcond=None)
        continuum[:, columns] = design @ coefficients
    continuum = continuum.reshape(data.shape)
    return data - continuum, continuum


def largest_regions_mask(
        mask: np.ndarray,
        count: int | None = None,
        min_pixels: int | None = None,
        min_ratio: float | None = None
) -> np.ndarray:
    """
    Return the mask (True where kept) of the largest connected regions of
    the pixels a mask keeps: the count largest, those of at least
    min_pixels pixels, and those at least min_ratio times the largest
    (each criterion if given).
    """
    regions = skimage.measure.regionprops(skimage.measure.label(mask != 0))
    regions.sort(key=lambda r: r.area, reverse=True)
    if count is not None:
        regions = regions[:count]
    if min_pixels is not None:
        regions = [r for r in regions if r.area >= min_pixels]
    if min_ratio is not None and regions:
        regions = [r for r in regions if r.area / regions[0].area >= min_ratio]
    kept = np.zeros(mask.shape, bool)
    for region in regions:
        kept[tuple(region.coords.T)] = True
    return kept


def bounding_box(mask: np.ndarray) -> tuple[slice, ...]:
    """
    Return the smallest ranges of pixels (one slice per axis) that hold
    the pixels a mask keeps.

    Raises
    ------
    ConfigError
        If the mask keeps no pixel.
    """
    nonzero = np.nonzero(mask)
    if not nonzero[0].size:
        raise ConfigError(
            "no valid pixels are left to crop the data to; relax the "
            "clipping, or do not minify")
    return tuple(slice(i.min(), i.max() + 1) for i in nonzero)


def voronoi_bins(
        signal: np.ndarray,
        noise: np.ndarray,
        target: float,
        tessellation: Literal['wvt', 'cvt'] | None = 'wvt'
) -> np.ndarray:
    """
    Return Voronoi bins of a map of about a target signal-to-noise each
    (Cappellari & Copin 2003): the bin of each pixel, as the bins of
    region datasets (see bin_index and the regions 'bins'). The pixels of
    finite signal and finite positive noise are binned; the S/N of a bin
    is the sum of the signal of its pixels over the sum of their noise in
    quadrature (of uncorrelated noise).

    Bin accretion makes the bins, from the pixel of highest S/N: a bin
    grows by the pixel nearest its centroid while that pixel is next to
    it, the bin stays round (roundness at most 0.3) and its S/N gets
    closer to the target; each next bin starts at the pixel nearest the
    centroid of the pixels binned so far. The pixels of the bins below
    80% of the target join the nearest bin kept. The iterations of a
    tessellation then even out the S/N of the bins and their shapes: of a
    weighted Voronoi tessellation ('wvt', Diehl & Statler 2006), or of a
    centroidal one whose bins hold about the same (S/N)^2 ('cvt',
    Cappellari & Copin 2003); None keeps the bins of the accretion.

    Parameters
    ----------
    signal : np.ndarray
        The signal of each pixel (a map).
    noise : np.ndarray
        The noise of each pixel (its standard deviation).
    target : float
        The target S/N of the bins.
    tessellation : {'wvt', 'cvt'} or None, optional
        The tessellation of the iterations.

    Returns
    -------
    np.ndarray
        The bin of each pixel (int32), from 0; -1 for no bin.

    Raises
    ------
    ValueError
        If the signal and the noise are not maps of the same shape, the
        tessellation is unknown, or no bin reaches the target.
    """
    signal = np.asarray(signal, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if signal.ndim != 2 or signal.shape != noise.shape:
        raise ValueError(
            f"the signal and the noise must be maps of the same shape; "
            f"they have shapes {signal.shape} and {noise.shape}")
    if tessellation not in ('wvt', 'cvt', None):
        raise ValueError(
            f"the tessellation must be 'wvt', 'cvt' or None, not "
            f"{tessellation!r}")
    with np.errstate(invalid='ignore'):
        valid = np.isfinite(signal) & np.isfinite(noise) & (noise > 0)
    total = signal[valid].sum() / np.sqrt(np.sum(noise[valid] ** 2))
    if not total >= target:
        raise ValueError(
            f"all the pixels together have a S/N of {total:.3g}, below the "
            f"target ({target})")
    labels, kept = _voronoi_accretion(signal, noise, valid, target)
    if not kept.any():
        raise ValueError(
            f"no bin of the accretion reaches "
            f"{_VORONOI_KEPT_FRACTION:.0%} of the target ({target})")
    # (np.argwhere goes through the pixels in the order of labels[valid])
    points = np.argwhere(valid).astype(float)
    labels = labels[valid]
    lost = ~kept[labels]
    labels = (np.cumsum(kept) - 1)[labels]
    generators = _centroids(points[~lost], labels[~lost])
    labels[lost] = _nearest_generators(
        points[lost], generators, np.ones(len(generators)))
    signal, noise2 = signal[valid], noise[valid] ** 2
    if tessellation is not None:
        labels = _voronoi_tessellation(
            points, signal, noise2, labels, tessellation)
    bins = np.full(valid.shape, -1, dtype=np.int32)
    bins[valid] = labels
    _log.info(f"{labels.max() + 1} Voronoi bins")
    return bins


def _voronoi_accretion(
        signal: np.ndarray,
        noise: np.ndarray,
        valid: np.ndarray,
        target: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the bins of the bin accretion of voronoi_bins: the bin of each
    pixel (from 0; -2 for the invalid pixels), and whether each bin is
    kept.
    """
    labels = np.where(valid, -1, -2)
    members = np.empty((np.count_nonzero(valid), 2))
    snr = np.where(valid, signal / np.where(valid, noise, 1), -np.inf)
    start = np.array(np.unravel_index(np.argmax(snr), snr.shape))
    binned, nbinned = np.zeros(2), 0
    kept = []
    while True:
        label, size = len(kept), 1
        members[0] = start
        labels[tuple(start)] = label
        sum_signal = signal[tuple(start)]
        sum_noise2 = noise[tuple(start)] ** 2
        sn = sum_signal / np.sqrt(sum_noise2)
        centre = members[0].copy()
        # (no pixel brings a bin at the target closer to it)
        while sn < target:
            # (a pixel next to the bin is within r_max + 1 of its centre)
            radius = np.sqrt(np.max(
                np.sum((members[:size] - centre) ** 2, axis=1))) + 1.5
            pixel = _nearest_free(labels, centre, radius)
            if pixel is None or not _touches(labels, pixel, label):
                break
            members[size] = pixel
            grown = members[:size + 1]
            grown_centre = grown.mean(axis=0)
            r_max = np.sqrt(np.max(np.sum((grown - grown_centre) ** 2, axis=1)))
            roundness = r_max / np.sqrt((size + 1) / np.pi) - 1
            pixel = tuple(pixel)
            grown_sn = (sum_signal + signal[pixel]) / np.sqrt(
                sum_noise2 + noise[pixel] ** 2)
            if (roundness > _VORONOI_MAX_ROUNDNESS
                    or abs(grown_sn - target) >= abs(sn - target)):
                break
            labels[pixel] = label
            size += 1
            sum_signal += signal[pixel]
            sum_noise2 += noise[pixel] ** 2
            sn, centre = grown_sn, grown_centre
        kept.append(sn >= _VORONOI_KEPT_FRACTION * target)
        binned += members[:size].sum(axis=0)
        nbinned += size
        free = np.argwhere(labels == -1)
        if not free.size:
            return labels, np.array(kept)
        start = free[np.argmin(
            np.sum((free - binned / nbinned) ** 2, axis=1))]


def _nearest_free(
        labels: np.ndarray, point: np.ndarray, radius: float
) -> np.ndarray | None:
    """
    Return the pixel without a bin yet (label -1) nearest a point, if one
    is within a radius of it.
    """
    lower = np.maximum(np.floor(point - radius).astype(int), 0)
    upper = np.minimum(
        np.ceil(point + radius).astype(int) + 1, labels.shape)
    window = labels[lower[0]:upper[0], lower[1]:upper[1]]
    free = np.argwhere(window == -1) + lower
    if not free.size:
        return None
    distances = np.sum((free - point) ** 2, axis=1)
    nearest = np.argmin(distances)
    return free[nearest] if distances[nearest] <= radius ** 2 else None


def _touches(labels: np.ndarray, pixel: np.ndarray, label: int) -> bool:
    """Return whether a pixel is next to one of a bin, along an axis."""
    y, x = pixel
    ny, nx = labels.shape
    return ((y > 0 and labels[y - 1, x] == label)
            or (y + 1 < ny and labels[y + 1, x] == label)
            or (x > 0 and labels[y, x - 1] == label)
            or (x + 1 < nx and labels[y, x + 1] == label))


def _voronoi_tessellation(
        points: np.ndarray,
        signal: np.ndarray,
        noise2: np.ndarray,
        labels: np.ndarray,
        tessellation: Literal['wvt', 'cvt']
) -> np.ndarray:
    """
    Return the bins of points (from 0) after the iterations of the
    tessellation of voronoi_bins from bins, until the bins stop changing;
    of the signal and noise squared of the points.
    """
    tiny = np.finfo(float).tiny
    # (of a centroidal tessellation of density rho, rho A^2 is about the
    # same for every bin of area A; so with rho = (S/N)^4 per pixel, so is
    # the (S/N)^2 of the bins, Cappellari & Copin 2003, Sect. 4)
    density = (np.maximum(signal ** 4 / noise2 ** 2, tiny)
               if tessellation == 'cvt' else np.ones(len(points)))
    for iteration in range(1, _VORONOI_MAX_ITERATIONS + 1):
        generators = _centroids(points, labels, density)
        scales = np.ones(len(generators))
        if tessellation == 'wvt':
            # (Diehl & Statler 2006, eq. 4)
            sn = (np.bincount(labels, signal)
                  / np.sqrt(np.bincount(labels, noise2)))
            scales = np.sqrt(np.bincount(labels) / np.maximum(sn, tiny))
        nearest = _nearest_generators(points, generators, scales)
        # (without the bins left empty)
        nearest = np.unique(nearest, return_inverse=True)[1]
        if np.array_equal(nearest, labels):
            _log.info(f"the Voronoi bins converged in {iteration} iterations")
            return labels
        labels = nearest
    _log.warning(
        f"the Voronoi bins did not converge in {_VORONOI_MAX_ITERATIONS} "
        f"iterations")
    return labels


def _centroids(
        points: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray | None = None
) -> np.ndarray:
    """Return the centroid of the points of each label, from 0, weighted."""
    weights = np.ones(len(points)) if weights is None else weights
    return np.column_stack([
        np.bincount(labels, weights * p) for p in points.T]) / np.bincount(
            labels, weights)[:, None]


def _nearest_generators(
        points: np.ndarray, generators: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """
    Return the generator nearest each point in distance over the scale of
    the generator (of a weighted Voronoi tessellation; of a Voronoi one
    with scales of 1).
    """
    tree = scipy.spatial.cKDTree(generators)
    which = np.empty(len(points), dtype=int)
    todo = np.arange(len(points))
    k = 16
    while todo.size:
        k = min(k, len(generators))
        distances, nearest = tree.query(points[todo], k)
        distances = distances.reshape(todo.size, k)
        nearest = nearest.reshape(todo.size, k)
        scaled = distances / scales[nearest]
        rows = np.arange(todo.size)
        best = np.argmin(scaled, axis=1)
        which[todo] = nearest[rows, best]
        if k == len(generators):
            break
        # (the others are at least the kth distance away: in distance over
        # scale, at least that over the largest scale; else, search more)
        sure = scaled[rows, best] <= distances[:, -1] / scales.max()
        todo = todo[~sure]
        k *= 4
    return which


def bin_index(bins: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Return the bins of a map numbered from 0 (in the order of their
    numbers; -1 for no bin), and their number.

    Parameters
    ----------
    bins : np.ndarray
        The number of the bin of each pixel (negative or NaN for no bin).

    Raises
    ------
    ConfigError
        If the bins are not integers.
    """
    valid = np.isfinite(bins) & (bins >= 0)
    if np.any(bins[valid] != np.round(bins[valid])):
        raise ConfigError("the bins must be integers")
    numbers = np.unique(bins[valid])
    index = np.full(bins.shape, -1, dtype=np.int32)
    index[valid] = np.searchsorted(numbers, bins[valid])
    return index, len(numbers)


def bin_values(index: np.ndarray, nbins: int, data: np.ndarray) -> np.ndarray:
    """
    Return the value of each bin of a map whose pixels hold the value of
    their bin (NaN for a bin without finite pixels).

    Raises
    ------
    ConfigError
        If the finite pixels of a bin hold different values.
    """
    values = np.full(nbins, np.nan)
    for i in range(nbins):
        pixels = data[(index == i) & np.isfinite(data)]
        if pixels.size == 0:
            continue
        if not np.allclose(pixels, pixels[0], rtol=1e-6, atol=0):
            raise ConfigError(
                f"the pixels of bin {i} hold different values; the map must "
                f"hold the value of each bin on its pixels")
        values[i] = pixels[0]
    return values


def pixel_fluxes(
        header: fits.Header, filename: str, nchannels: int
) -> tuple[float | np.ndarray, astropy.units.UnitBase] | None:
    """
    Return how the values of the pixels of data of a header (e.g. that of
    FitsArrays, of the file of the given name; in its unit, BUNIT) become
    their fluxes: a factor, and the unit of the fluxes. Values per pixel
    (e.g. of a spaxel, or per pix) are fluxes (factor 1); per solid
    angle, they are times that of a pixel; per beam (e.g. Jy/beam), times
    that of a pixel over that of the Gaussian beam, pi / (4 ln 2) bmaj
    bmin, of the header (BMAJ and BMIN), or of each channel of a cube of
    CASA (the table BEAMS of the file; a factor for each of its nchannels
    channels).

    Parameters
    ----------
    header : fits.Header
        The header of the data.
    filename : str
        The file of the data (for its table BEAMS, if any).
    nchannels : int
        The number of channels of the data (of a cube).

    Returns
    -------
    tuple of float or np.ndarray and astropy.units.UnitBase, or None
        The factor (one, or one for each channel) and the unit of the
        fluxes; None if the header has no unit.

    Raises
    ------
    ConfigError
        If the unit is not a unit, is a brightness temperature (whose
        sums over pixels are not fluxes), or is per beam and the data
        have no beam, or a beam for another number of channels.
    """
    u = astropy.units
    text = str(header.get('BUNIT', '')).strip()
    if not text:
        return None
    unit = unitutils.parse_unit(text)
    if unit.is_equivalent(u.K):
        raise ConfigError(
            f"{filename}: the data are a brightness temperature ({text!r}), "
            f"whose sums over pixels are not fluxes: convert them to flux "
            f"density (e.g. Jy/beam) first")
    spatial = unitutils.spatial_unit(unit)
    if spatial in (u.dimensionless_unscaled, u.pix):
        return 1.0, unit * spatial
    wcs = astropy.wcs.WCS(header).celestial
    pixel = astropy.wcs.utils.proj_plane_pixel_area(wcs) * u.deg ** 2
    if spatial == u.sr:
        # (the angles of the unit, e.g. 1/arcsec^2)
        angles = u.dimensionless_unscaled
        for base, power in zip(unit.bases, unit.powers):
            if base.physical_type in ('solid angle', 'angle'):
                angles = angles * base ** power
        return float((pixel * angles).decompose()), unit / angles
    if 'BMAJ' in header and 'BMIN' in header:
        beam = dict(
            bmaj=float(header['BMAJ']) * 3600,
            bmin=float(header['BMIN']) * 3600)
    else:
        beam = fitsutils.read_beam(filename)
    if beam is None:
        raise ConfigError(
            f"{filename}: the data are per beam, but have no beam (BMAJ and "
            f"BMIN in the header, or a table BEAMS)")
    area = np.pi / (4 * np.log(2)) * np.multiply(beam['bmaj'], beam['bmin'])
    if np.ndim(area) and len(area) != nchannels:
        raise ConfigError(
            f"{filename}: the data have {nchannels} channels, but beams for "
            f"{len(area)}")
    return pixel.to_value(u.arcsec ** 2) / area, unit * u.beam


def bin_spectra(
        index: np.ndarray,
        nbins: int,
        cube: FitsArrays
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Return the spectrum of each bin of a cube: the sum of the valid
    spectra of its pixels (a flux, as the region datasets compare; in
    float64), and its errors (added in quadrature). A channel of a bin
    without valid pixels is NaN.

    Parameters
    ----------
    index : np.ndarray
        The bin of each pixel, from 0 (-1 for none; see bin_index).
    nbins : int
        The number of bins.
    cube : FitsArrays
        The cube (channels first).

    Returns
    -------
    np.ndarray
        The spectra, of shape (nchannels, nbins).
    np.ndarray or None
        Their errors, if the cube has some.
    """
    nchannels = cube.data.shape[0]
    spectra = np.full((nchannels, nbins), np.nan)
    errors = None if cube.error is None else spectra.copy()
    valid = cube.mask != 0
    for i in range(nbins):
        pixels = index == i
        counts = valid[:, pixels].sum(axis=1)
        sums = np.nansum(cube.data[:, pixels], axis=1, dtype=np.float64)
        spectra[:, i] = np.where(counts > 0, sums, np.nan)
        if errors is not None:
            variances = np.nansum(
                cube.error[:, pixels].astype(np.float64) ** 2, axis=1)
            errors[:, i] = np.where(counts > 0, np.sqrt(variances), np.nan)
    return spectra, errors


def muse_psf_table(header: fits.Header) -> astropy.table.QTable:
    """
    Return the PSF of a MUSE cube (from the keywords of its header: the
    MOFFAT1 and MOFFAT2 modes of MPDAF, its first field) as a table of the
    options alpha and beta of a Moffat PSF against wavelength.

    Raises
    ------
    ConfigError
        If the header has no PSF of MUSE.
    """
    wavelengths = _MUSE_WAVELENGTHS
    mode = header.get('FSFMODE')
    fields = [f for f in (*range(10), 99) if f'FSF{f:02d}BET' in header
              or f'FSF{f:02d}FNC' in header]
    if mode not in ('MOFFAT1', 'MOFFAT2') or not fields:
        raise ConfigError(
            f"the header has no PSF of MUSE (FSFMODE MOFFAT1 or MOFFAT2); "
            f"FSFMODE is {mode!r}")
    key = f'FSF{fields[0]:02d}'
    if mode == 'MOFFAT1':
        fwhm = header[f'{key}FWA'] + header[f'{key}FWB'] * wavelengths
        beta = np.full_like(wavelengths, header[f'{key}BET'])
    else:
        # (polynomials of the wavelength normalised to [-0.5, 0.5] in the
        # range FSFLB1 to FSFLB2, the coefficient of the highest power
        # first)
        lb1, lb2 = header['FSFLB1'], header['FSFLB2']
        x = (wavelengths - lb1) / (lb2 - lb1) - 0.5
        fwhm = np.polyval([
            header[f'{key}F{k:02d}']
            for k in range(header[f'{key}FNC'])], x)
        beta = np.polyval([
            header[f'{key}B{k:02d}']
            for k in range(header[f'{key}BNC'])], x)
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    u = astropy.units
    return astropy.table.QTable(dict(
        wavelength=wavelengths * u.Angstrom, alpha=alpha * u.arcsec,
        beta=beta * u.dimensionless_unscaled))


def muse_lsf_table() -> astropy.table.QTable:
    """
    Return the LSF of MUSE (a Gaussian whose FWHM is that of Bacon et al.
    2017, eq. 8) as a table of its sigma against wavelength.
    """
    w = _MUSE_WAVELENGTHS
    fwhm = 5.866e-8 * w ** 2 - 9.187e-4 * w + 6.040
    u = astropy.units
    return astropy.table.QTable(dict(
        wavelength=w * u.Angstrom,
        sigma=fwhm / np.sqrt(8 * np.log(2)) * u.Angstrom))




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
    resampled (needs the package jwst).

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
    from jwst import datamodels
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
    file of the source (see nirspec_msa_spectra; needs the package jwst).

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
    from jwst import datamodels
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
    from gbkfit.observation.observables.apertures import ApertureShutters
    from gbkfit.observation.observables.slitless_spectra import Exposure
    from gbkfit.observation.observables.traces import TraceTable
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
