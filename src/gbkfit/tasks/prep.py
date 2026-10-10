
import logging
import os
import re

import astropy.constants
import astropy.io.fits as fits
import astropy.stats as stats
import astropy.units
import astropy.wcs
import numpy as np
import skimage.measure

from gbkfit.utils import fitsutils, gridutils
from gbkfit.utils.parseutils import ConfigError


_log = logging.getLogger(__name__)

# The keywords of the world coordinates of each axis, or pair of axes
_WCS_KEYWORDS = re.compile(
    r'(WCSAXES'
    r'|(CRPIX|CRVAL|CDELT|CUNIT|CTYPE|CROTA)\d+'
    r'|(PC|CD|PV|PS)\d+_\d+)$')

# The keywords of the rest of the spectral axis, also in their old forms
_REST_KEYWORDS = ('RESTWAV', 'RESTFRQ', 'RESTWAVE', 'RESTFREQ')


def _has_wcs(header):
    return any(_WCS_KEYWORDS.match(key) for key in header)


def _with_wcs(header, wcs):
    """
    A copy of the header with the world coordinates of wcs, and its rest
    (if any).
    """
    header = header.copy()
    for key in [
            key for key in header
            if _WCS_KEYWORDS.match(key) or key in _REST_KEYWORDS]:
        del header[key]
    header.update(wcs.to_header())
    return header


def _drop_axes(header, axes):
    """
    The header of the data without the given axes (numpy axes, of length
    1), without the world coordinates of these axes.
    """
    if not _has_wcs(header):
        return header
    wcs = astropy.wcs.WCS(header)
    for axis in sorted(wcs.naxis - 1 - axis for axis in axes)[::-1]:
        wcs = wcs.dropaxis(axis)
    return _with_wcs(header, wcs)


def _shift_axes(header, offset):
    """
    The header of the data whose first pixel is the pixel at the given
    offset (on each numpy axis) of the original data: positive where the
    data was cropped, negative where it was padded.
    """
    if not _has_wcs(header):
        return header
    wcs = astropy.wcs.WCS(header)
    wcs.wcs.crpix -= offset[::-1]
    return _with_wcs(header, wcs)


def _spectral_to_velocity(header, rest):
    """
    The header with its spectral axis converted to the velocities of the
    given rest (see gridutils.make_rest; None leaves it unchanged): a
    linear wavelength axis (WAVE, or AWAV with a rest in air too) to the
    optical velocities c (w / rest - 1), a linear frequency axis (FREQ) to
    the radio velocities c (1 - f / rest). Both are linear in the pixels,
    so the conversion is exact; the header gets the rest (RESTWAV or
    RESTFRQ). A velocity axis (VOPT, VRAD) gets the rest only, and if it
    has one, it must be the same: its velocities refer to it.
    """
    if rest is None:
        return header
    rest = gridutils.make_rest(rest)
    wcs = astropy.wcs.WCS(header)
    s = wcs.wcs.spec
    if s < 0:
        raise ConfigError("the data have no spectral axis to convert")
    ctype = wcs.wcs.ctype[s]
    kind = ctype[:4]
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
    return _with_wcs(header, wcs)


def _check_same_rest(wcs, rest):
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


def _reorder(data, header, order):
    """
    Return the data and its header with their axes in the given order
    (FITS order, from 0; see fitsutils.model_axis_order). A header without
    world coordinates stays as it is.
    """
    data = fitsutils.reorder_data(data, order)
    if not _has_wcs(header):
        return data, header
    wcs = fitsutils.reorder_wcs(astropy.wcs.WCS(header), order)
    return data, _with_wcs(header, wcs)


def _flip(data, header, axis):
    """
    Return the data and its header reversed along an axis (FITS order,
    from 0): each pixel keeps its world coordinates. A header without
    world coordinates stays as it is.
    """
    size = data.shape[data.ndim - 1 - axis]
    data = fitsutils.flip_data(data, axis)
    if not _has_wcs(header):
        return data, header
    wcs = fitsutils.flip_wcs(astropy.wcs.WCS(header), axis, size)
    return data, _with_wcs(header, wcs)


def _read_fits(filename):
    """
    The data and the header of a file, without the axes of length 1: those
    of its first HDU with data (e.g. the extension after an empty primary
    HDU, as in MUSE and JWST cubes). Integer data become floats, which
    can hold the NaN of invalid pixels.
    """
    with fits.open(filename) as hdus:
        hdu = next((hdu for hdu in hdus if hdu.data is not None), None)
        if hdu is None:
            raise ConfigError(f"{filename}: the file has no data")
        data = np.array(hdu.data)
        header = hdu.header.copy()
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(float)
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


def _read_data(file_d, file_e, file_m):
    # Load data and headers, without their axes of length 1
    data_d, header_d = _read_fits(file_d)
    data_e, header_e = _read_fits(file_e) if file_e else (None, None)
    data_m, header_m = _read_fits(file_m) if file_m else (None, None)
    # The axes in the order of the model (e.g. RA before Dec), as those of
    # the data, also for an error and a mask without world coordinates.
    # (Reversed axes are reversed when saved, so that the regions given
    # are in the pixels of the files.)
    if _has_wcs(header_d):
        order = fitsutils.model_axis_order(astropy.wcs.WCS(header_d))
        if order != tuple(range(data_d.ndim)):
            _log.info("reordering the axes in the order of the model")
            data_d, header_d = _reorder(data_d, header_d, order)
            if data_e is not None:
                data_e, header_e = _reorder(data_e, header_e, order)
            if data_m is not None:
                data_m, header_m = _reorder(data_m, header_m, order)
    data_d = np.ascontiguousarray(data_d)
    data_e = np.ascontiguousarray(data_e) if data_e is not None else None
    data_m = np.ascontiguousarray(data_m) if data_m is not None else None
    # Deal with invalid or non-sensible pixel values
    data_d[~np.isfinite(data_d)] = np.nan
    if data_e is not None:
        data_e[~np.isfinite(data_e)] = np.nan
        data_e[data_e <= 0] = np.nan
    if data_m is not None:
        data_m[~np.isfinite(data_m)] = 0
        data_m[np.nonzero(data_m)] = 1
    return data_d, header_d, data_e, header_e, data_m, header_m


def _save_data(
        file_d, data_d, header_d,
        file_e, data_e, header_e,
        file_m, data_m, header_m,
        offset, dtype, velocity_rest=None, output_dir='.'):
    """
    Save the data to output_dir, whose first pixel is the pixel at the
    given offset (on each numpy axis) of the data read, with its spectral
    axis converted to the velocities of velocity_rest (see
    _spectral_to_velocity; the error and mask files are converted if they
    have world coordinates).
    The axes that the model needs reversed (see fitsutils.model_axis_flips:
    that of a mirrored image, and a spectral axis of decreasing
    velocities) are reversed in the data, error and mask (with or without
    world coordinates).
    """
    basename = os.path.basename
    splitext = os.path.splitext

    def prepare(header, has_wcs):
        header = _shift_axes(header, offset)
        if has_wcs:
            header = _spectral_to_velocity(header, velocity_rest)
        return header
    header_d = prepare(header_d, True)
    if header_e is not None:
        header_e = prepare(header_e, _has_wcs(header_e))
    if header_m is not None:
        header_m = prepare(header_m, _has_wcs(header_m))
    flips = ()
    if _has_wcs(header_d):
        flips = fitsutils.model_axis_flips(astropy.wcs.WCS(header_d))
    for axis in flips:
        _log.info(
            f"reversing the axis {axis + 1}: the image is mirrored, or the "
            f"velocity decreases along it")
        data_d, header_d = _flip(data_d, header_d, axis)
        if data_e is not None:
            data_e, header_e = _flip(data_e, header_e, axis)
        if data_m is not None:
            data_m, header_m = _flip(data_m, header_m, axis)
    def write(name, data, header):
        filename = os.path.join(output_dir, f'prep_{name}.fits')
        fits.writeto(filename, data.astype(dtype), header, overwrite=True)
    file_d = splitext(basename(file_d))[0]
    write(file_d, data_d, header_d)
    if data_e is not None:
        write(splitext(basename(file_e))[0], data_e, header_e)
    if data_m is not None:
        file_m = splitext(basename(file_m))[0] if file_m else file_d + '_mask'
        write(file_m, data_m, header_m)


def _crop_data(data_d, data_e, data_m, axis, range_):
    s = [slice(None), ] * data_d.ndim
    s[axis] = slice(range_[0], range_[1])
    data_d = data_d[tuple(s)]
    if data_e is not None:
        data_e = data_e[tuple(s)]
    if data_m is not None:
        data_m = data_m[tuple(s)]
    return data_d, data_e, data_m


def _make_mask(data_d, data_e, data_m):
    """
    The pixels with finite data and error, and not masked (data_m is 0
    or 1, see _read_data).
    """
    mask = np.ones_like(data_d)
    mask *= np.isfinite(data_d)
    if data_e is not None:
        mask *= np.isfinite(data_e)
    if data_m is not None:
        mask *= data_m != 0
    return mask


def _compare_nan_array(func, ary, threshold):
    out = ~np.isnan(ary)
    out[out] = func(ary[out], threshold)
    return out


def _make_mask_clip_min(data, min_value):
    return ~_compare_nan_array(np.less, data, min_value)


def _make_mask_clip_max(data, max_value):
    return ~_compare_nan_array(np.greater, data, max_value)


def _make_mask_clip_sig(data, mask, sigma, maxiters):
    """
    The pixels that sigma clipping keeps (its mask marks the others), of
    those that the mask keeps (the others do not count in its statistics).
    """
    kept = np.ma.masked_array(data, mask=(mask == 0))
    return ~stats.sigma_clip(kept, sigma=sigma, maxiters=maxiters).mask


def _make_mask_clip_ccl(data, lcount, pcount, lratio):
    """
    The largest connected regions of the pixels of data that are not 0:
    the lcount largest, those of at least pcount pixels, and those at
    least lratio times the largest.
    """
    labels = skimage.measure.label(data)
    props = skimage.measure.regionprops(labels)
    props.sort(key=lambda x: x.area, reverse=True)
    if lcount is not None:
        props = props[0:lcount]
    if pcount is not None:
        props = [p for p in props if p.area >= pcount]
    if lratio is not None and props:
        props = [p for p in props if p.area / props[0].area >= lratio]
    mask = np.zeros_like(data)
    for p in props:
        mask[tuple(p.coords.T)] = 1
    return mask


def _apply_mask(data_d, data_e, data_m, mask):
    data_d[mask == 0] = np.nan
    if data_e is not None:
        data_e[mask == 0] = np.nan
    if data_m is not None:
        data_m[mask == 0] = 0


def _valid_pixels(mask):
    """
    The indices of the pixels the mask keeps (see numpy.nonzero); raise
    ConfigError if there are none (there is nothing to minify to).
    """
    nonzero = mask.nonzero()
    if not nonzero[0].size:
        raise ConfigError(
            "no valid pixels are left to minify the data to; relax the "
            "clipping, or do not minify")
    return nonzero


def _minify_data(data_d, data_e, data_m, mask):
    """
    Crop the data to the masked pixels. Return it, and the index of its
    first pixel on each axis.
    """
    nonzero = _valid_pixels(mask)
    start = np.array([indices.min() for indices in nonzero])
    slices = tuple(
        slice(indices.min(), indices.max() + 1) for indices in nonzero)
    data_d = data_d[slices]
    if data_e is not None:
        data_e = data_e[slices]
    if data_m is not None:
        data_m = data_m[slices]
    return data_d, data_e, data_m, start


def _pad_data(data_d, data_e, data_m, size, value_d, value_e, value_m):
    data_d = np.pad(data_d, size, 'constant', constant_values=value_d)
    if data_e is not None:
        data_e = np.pad(data_e, size, 'constant', constant_values=value_e)
    if data_m is not None:
        data_m = np.pad(data_m, size, 'constant', constant_values=value_m)
    return data_d, data_e, data_m


def prep_pixel_brightness(
        file_d, file_e, file_m,
        roi_spat, clip_min, clip_max, ccl_lcount, ccl_pcount, ccl_lratio,
        sclip_sigma, sclip_iters, minify, nanpad, dtype,
        output_dir='.'):

    (data_d, header_d,
     data_e, header_e,
     data_m, header_m) = _read_data(file_d, file_e, file_m)
    # The index of the first pixel of the prepared data in the data read
    offset = np.zeros(data_d.ndim, int)

    if roi_spat is not None:
        xrange = roi_spat[0:2]
        yrange = roi_spat[2:4]
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 1, xrange)
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 0, yrange)
        offset += [yrange[0], xrange[0]]

    mask = _make_mask(data_d, data_e, data_m)

    if clip_min is not None:
        mask *= _make_mask_clip_min(data_d, clip_min)
    if clip_max is not None:
        mask *= _make_mask_clip_max(data_d, clip_max)
    if sclip_sigma is not None:
        mask *= _make_mask_clip_sig(
            data_d, mask, sclip_sigma, sclip_iters)
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask *= _make_mask_clip_ccl(
            mask != 0, ccl_lcount, ccl_pcount, ccl_lratio)

    _apply_mask(data_d, data_e, data_m, mask)

    if minify:
        data_d, data_e, data_m, start = _minify_data(
            data_d, data_e, data_m, mask)
        offset += start

    if nanpad:
        data_d, data_e, data_m = _pad_data(
            data_d, data_e, data_m, nanpad, np.nan, np.nan, 0)
        offset -= nanpad

    _save_data(
        file_d, data_d, header_d,
        file_e, data_e, header_e,
        file_m, data_m, header_m,
        offset, dtype, output_dir=output_dir)


def prep_slit_spectra(
        file_d, file_e, file_m,
        roi_spat, roi_spec, clip_min, clip_max,
        ccl_lcount, ccl_pcount, ccl_lratio,
        sclip_sigma, sclip_iters, minify, nanpad, dtype,
        velocity_rest=None, output_dir='.'):

    (data_d, header_d,
     data_e, header_e,
     data_m, header_m) = _read_data(file_d, file_e, file_m)
    # The index of the first pixel of the prepared data in the data read
    offset = np.zeros(data_d.ndim, int)

    # The data are (channels, positions) in numpy order
    if roi_spat is not None:
        xrange = roi_spat
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 1, xrange)
        offset[1] += xrange[0]
    if roi_spec is not None:
        srange = roi_spec
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 0, srange)
        offset[0] += srange[0]

    mask = _make_mask(data_d, data_e, data_m)

    if clip_min is not None:
        mask *= _make_mask_clip_min(data_d, clip_min)
    if clip_max is not None:
        mask *= _make_mask_clip_max(data_d, clip_max)
    if sclip_sigma is not None:
        mask *= _make_mask_clip_sig(
            data_d, mask, sclip_sigma, sclip_iters)
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask *= _make_mask_clip_ccl(
            mask != 0, ccl_lcount, ccl_pcount, ccl_lratio)

    _apply_mask(data_d, data_e, data_m, mask)

    if minify:
        data_d, data_e, data_m, start = _minify_data(
            data_d, data_e, data_m, mask)
        offset += start

    if nanpad:
        data_d, data_e, data_m = _pad_data(
            data_d, data_e, data_m, nanpad, np.nan, np.nan, 0)
        offset -= nanpad

    _save_data(
        file_d, data_d, header_d,
        file_e, data_e, header_e,
        file_m, data_m, header_m,
        offset, dtype, velocity_rest, output_dir)


def prep_pixel_moments(
        file_d, file_e, file_m,
        roi_spat, clip_min, clip_max, ccl_lcount, ccl_pcount, ccl_lratio,
        sclip_sigma, sclip_iters, minify, nanpad, dtype,
        output_dir='.'):
    """
    The options of the clipping (clip_min, clip_max, sclip_sigma and
    sclip_iters) have one value for each map; sclip_iters can also be
    one value for all maps.
    """

    nmoments = len(file_d)
    if file_e is None:
        file_e = [None] * nmoments
    if file_m is None:
        file_m = [None] * nmoments
    if np.ndim(sclip_iters) == 0:
        sclip_iters = [sclip_iters] * nmoments

    data_d = []
    data_e = []
    data_m = []
    header_d = []
    header_e = []
    header_m = []
    # The index of the first pixel of the prepared data in the data read
    offset = np.zeros(2, int)
    if roi_spat is not None:
        offset += [roi_spat[2], roi_spat[0]]
    for i in range(nmoments):
        (data_d_, header_d_,
         data_e_, header_e_,
         data_m_, header_m_) = _read_data(file_d[i], file_e[i], file_m[i])
        if roi_spat is not None:
            xrange = roi_spat[0:2]
            yrange = roi_spat[2:4]
            data_d_, data_e_, data_m_ = _crop_data(
                data_d_, data_e_, data_m_, 1, xrange)
            data_d_, data_e_, data_m_ = _crop_data(
                data_d_, data_e_, data_m_, 0, yrange)
        data_d.append(data_d_)
        data_e.append(data_e_)
        data_m.append(data_m_)
        header_d.append(header_d_)
        header_e.append(header_e_)
        header_m.append(header_m_)

    mask = np.ones_like(data_d[0])

    for i in range(nmoments):
        mask *= _make_mask(data_d[i], data_e[i], data_m[i])
        if clip_min is not None:
            mask *= _make_mask_clip_min(data_d[i], clip_min[i])
        if clip_max is not None:
            mask *= _make_mask_clip_max(data_d[i], clip_max[i])
        if sclip_sigma is not None:
            mask *= _make_mask_clip_sig(
                data_d[i], mask, sclip_sigma[i], sclip_iters[i])
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask *= _make_mask_clip_ccl(
            mask != 0, ccl_lcount, ccl_pcount, ccl_lratio)

    if minify:
        offset += [indices.min() for indices in _valid_pixels(mask)]
    if nanpad:
        offset -= nanpad

    for i in range(nmoments):

        _apply_mask(data_d[i], data_e[i], data_m[i], mask)

        if minify:
            data_d[i], data_e[i], data_m[i], _ = _minify_data(
                data_d[i], data_e[i], data_m[i], mask)

        if nanpad:
            data_d[i], data_e[i], data_m[i] = _pad_data(
                data_d[i], data_e[i], data_m[i], nanpad, np.nan, np.nan, 0)

        _save_data(
            file_d[i], data_d[i], header_d[i],
            file_e[i], data_e[i], header_e[i],
            file_m[i], data_m[i], header_m[i],
            offset, dtype, output_dir=output_dir)


def prep_pixel_spectra(
        file_d, file_e, file_m,
        roi_spat, roi_spec, clip_min, clip_max,
        ccl_lcount, ccl_pcount, ccl_lratio,
        sclip_sigma, sclip_iters, minify, nanpad, dtype,
        velocity_rest=None, output_dir='.'):

    (data_d, header_d,
     data_e, header_e,
     data_m, header_m) = _read_data(file_d, file_e, file_m)
    # The index of the first pixel of the prepared data in the data read
    offset = np.zeros(data_d.ndim, int)

    if roi_spat is not None:
        xrange = roi_spat[0:2]
        yrange = roi_spat[2:4]
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 2, xrange)
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 1, yrange)
        offset[1:] += [yrange[0], xrange[0]]
    if roi_spec is not None:
        srange = roi_spec
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 0, srange)
        offset[0] += srange[0]

    mask = _make_mask(data_d, data_e, data_m)

    if clip_min is not None:
        mask *= _make_mask_clip_min(data_d, clip_min)
    if clip_max is not None:
        mask *= _make_mask_clip_max(data_d, clip_max)
    if sclip_sigma is not None:
        mask *= _make_mask_clip_sig(
            data_d, mask, sclip_sigma, sclip_iters)
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask *= _make_mask_clip_ccl(
            mask != 0, ccl_lcount, ccl_pcount, ccl_lratio)

    _apply_mask(data_d, data_e, data_m, mask)

    if minify:
        data_d, data_e, data_m, start = _minify_data(
            data_d, data_e, data_m, mask)
        offset += start

    if nanpad:
        data_d, data_e, data_m = _pad_data(
            data_d, data_e, data_m, nanpad, np.nan, np.nan, 0)
        offset -= nanpad

    _save_data(
        file_d, data_d, header_d,
        file_e, data_e, header_e,
        file_m, data_m, header_m,
        offset, dtype, velocity_rest, output_dir)


def _bin_values(bins, nbins, data, name):
    """
    The value of each bin of a map whose pixels hold the value of their bin
    (NaN for a bin without finite pixels). Raise ConfigError if the finite
    pixels of a bin hold different values.
    """
    values = np.full(nbins, np.nan)
    for i in range(nbins):
        pixels = data[(bins == i) & np.isfinite(data)]
        if pixels.size == 0:
            continue
        if not np.allclose(pixels, pixels[0], rtol=1e-6, atol=0):
            raise ConfigError(
                f"{name}: the pixels of bin {i} hold different values; the "
                f"map must hold the value of each bin on its pixels")
        values[i] = pixels[0]
    return values


def prep_region_moments(
        file_bins, file_d, file_e, file_m, dtype, output_dir='.'):
    """
    Prepare binned moment maps (e.g. of MaNGA DAP or GIST) for
    region_moments: a map of the bin of each pixel (file_bins; negative or
    NaN for no bin),
    and moment maps (file_d, with optional errors file_e and masks file_m)
    whose pixels hold the value of their bin. Write the bins, numbered
    from 0 in the order of their numbers in the map
    (prep_<bins>.fits), and for each map a vector of the value of each
    bin (prep_<map>.fits), whose masked or empty bins are NaN.
    """
    nmoments = len(file_d)
    file_e = file_e or [None] * nmoments
    file_m = file_m or [None] * nmoments
    bins, header_bins = _read_fits(file_bins)
    if bins.ndim != 2:
        raise ConfigError(
            f"{file_bins}: the bins must be an image; they have "
            f"{bins.ndim} axes")
    valid = np.isfinite(bins) & (bins >= 0)
    if np.any(bins[valid] != np.round(bins[valid])):
        raise ConfigError(f"{file_bins}: the bins must be integers")
    numbers = np.unique(bins[valid])
    index = np.full(bins.shape, -1, dtype=np.int32)
    index[valid] = np.searchsorted(numbers, bins[valid])
    _log.info(f"{len(numbers)} bins")

    def save(filename, data, header=None):
        name = os.path.splitext(os.path.basename(filename))[0]
        fits.writeto(
            os.path.join(output_dir, f'prep_{name}.fits'), data, header,
            overwrite=True)

    save(file_bins, index, header_bins)
    for i in range(nmoments):
        data_d, _, data_e, _, data_m, _ = _read_data(
            file_d[i], file_e[i], file_m[i])
        for data in (data_d, data_e, data_m):
            if data is not None and data.shape != bins.shape:
                raise ConfigError(
                    f"{file_d[i]}: the maps must have the shape of the bins "
                    f"{bins.shape}; they have {data.shape}")
        mask = _make_mask(data_d, data_e, data_m)
        _apply_mask(data_d, data_e, None, mask)
        save(file_d[i], _bin_values(
            index, len(numbers), data_d, file_d[i]).astype(dtype))
        if data_e is not None:
            save(file_e[i], _bin_values(
                index, len(numbers), data_e, file_e[i]).astype(dtype))


# The PSFs and LSFs of instruments, as tables for the options of PSFs and
# LSFs that vary along the spectral axis (see gbkfit.instrument.varying)

# The wavelengths (Angstrom) of the tables of MUSE (its range, and more)
_MUSE_WAVELENGTHS = np.arange(4600.0, 9401.0, 50.0)


def _write_table(columns, units, name, output_dir):
    """Write a table (ECSV) of the given columns and units; return its path."""
    import astropy.table
    table = astropy.table.QTable({
        key: np.asarray(values) * astropy.units.Unit(units[key])
        for key, values in columns.items()})
    filename = os.path.join(output_dir, f'prep_{name}.ecsv')
    table.write(filename, overwrite=True)
    return filename


def _muse_fsf(header, wavelengths):
    """
    The FWHM (arcsec) and beta of the Moffat PSF of a MUSE cube at the
    given wavelengths (Angstrom), from the keywords of its header (the
    MOFFAT1 and MOFFAT2 modes of MPDAF; the first field).
    """
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
        return fwhm, beta
    # (polynomials of the wavelength normalised to [-0.5, 0.5] in the
    # range FSFLB1 to FSFLB2, the coefficient of the highest power first)
    lb1, lb2 = header['FSFLB1'], header['FSFLB2']
    x = (wavelengths - lb1) / (lb2 - lb1) - 0.5
    fwhm_coeffs = [header[f'{key}F{k:02d}']
                   for k in range(header[f'{key}FNC'])]
    beta_coeffs = [header[f'{key}B{k:02d}']
                   for k in range(header[f'{key}BNC'])]
    return np.polyval(fwhm_coeffs, x), np.polyval(beta_coeffs, x)


def prep_psf_muse(file, output_dir='.'):
    """
    Write the PSF of a MUSE cube (from its header) as a table of the
    options alpha and beta of a Moffat PSF against wavelength.
    """
    header = fits.getheader(file, 0)
    fwhm, beta = _muse_fsf(header, _MUSE_WAVELENGTHS)
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    filename = _write_table(
        dict(wavelength=_MUSE_WAVELENGTHS, alpha=alpha, beta=beta),
        dict(wavelength='Angstrom', alpha='arcsec', beta=''), 'psf',
        output_dir)
    _log.info(
        f"the PSF is in {filename}; use it as psf: {{type: moffat, "
        f"alpha: {{table: {filename}, column: alpha}}, "
        f"beta: {{table: {filename}, column: beta}}}}")


def prep_lsf_muse(output_dir='.'):
    """
    Write the LSF of MUSE (a Gaussian whose FWHM is that of Bacon et al.
    2017, eq. 8) as a table of its sigma against wavelength.
    """
    w = _MUSE_WAVELENGTHS
    fwhm = 5.866e-8 * w ** 2 - 9.187e-4 * w + 6.040
    filename = _write_table(
        dict(wavelength=w, sigma=fwhm / np.sqrt(8 * np.log(2))),
        dict(wavelength='Angstrom', sigma='Angstrom'), 'lsf', output_dir)
    _log.info(
        f"the LSF is in {filename}; use it as lsf: {{type: gauss, "
        f"sigma: {{table: {filename}, column: sigma}}}}")


def prep_psf_casa_beams(file, output_dir='.'):
    """
    Write the beams of the channels of a cube of CASA (its table BEAMS)
    as a table of the options of a Gaussian beam PSF against the points
    of its spectral axis (e.g. frequency). Only the first polarisation is
    used.
    """
    import astropy.table
    beams = astropy.table.Table.read(file, hdu='BEAMS')
    if 'POL' in beams.colnames:
        beams = beams[beams['POL'] == 0]
    wcs = astropy.wcs.WCS(fits.getheader(file, 0))
    if wcs.wcs.spec < 0:
        raise ConfigError(f"{file}: the cube has no spectral axis")
    spectral = wcs.sub([wcs.wcs.spec + 1])
    unit = astropy.units.Unit(spectral.wcs.cunit[0])
    points = spectral.pixel_to_world_values(np.asarray(beams['CHAN']))
    axis = ('frequency' if unit.is_equivalent(astropy.units.Hz) else
            'wavelength' if unit.is_equivalent(astropy.units.m) else
            'velocity')

    def column(name, to):
        return (beams[name].quantity if beams[name].unit else
                beams[name] * astropy.units.Unit(to)).to_value(to)
    filename = _write_table(
        {axis: points, 'bmaj': column('BMAJ', 'arcsec'),
         'bmin': column('BMIN', 'arcsec'), 'bpa': column('BPA', 'deg')},
        {axis: str(unit), 'bmaj': 'arcsec', 'bmin': 'arcsec', 'bpa': 'deg'},
        'psf', output_dir)
    _log.info(
        f"the beams are in {filename}; use them as psf: {{type: "
        f"gauss_beam, bmaj: {{table: {filename}, column: bmaj}}, ...}} "
        f"(and bmin and bpa)")
