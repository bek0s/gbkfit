
import os

import astropy.io.fits as fits
import astropy.stats as stats
import numpy as np
import skimage.measure


def _read_data(file_d, file_e, file_m):

    data_d = fits.getdata(file_d)
    data_e = fits.getdata(file_e) if file_e else None
    data_m = fits.getdata(file_m) if file_m else None
    header_d = fits.getheader(file_d)
    header_e = fits.getheader(file_e) if file_e else dict()

    data_d = np.squeeze(data_d)
    if data_e is not None:
        data_e = np.squeeze(data_e)
    if data_m is not None:
        data_m = np.squeeze(data_m)

    data_d[~np.isfinite(data_d)] = np.nan
    if data_e is not None:
        data_e[~np.isfinite(data_e)] = np.nan
        data_e[data_e <= 0] = np.nan
    if data_m is not None:
        data_m[np.nonzero(data_m)] = 1
        data_m[np.isnan(data_m)] = 0
    """
    wcs_d = astropy.wcs.WCS(header_d)
    wcs_e = astropy.wcs.WCS(header_e)
    if unit_d is None or not unit_d.strip() or unit_d.lower() == 'auto':
        unit_d = header_d.get('BUNIT', '')
    if unit_e is None or not unit_e.strip() or unit_e.lower() == 'auto':
        unit_e = header_e.get('BUNIT', unit_d)
    unit_d = units.Unit(unit_d)
    unit_e = units.Unit(unit_e)
    """
    return data_d, data_e, data_m, header_d, header_e


def _save_data(file_d, file_e, file_m, data_d, data_e, data_m, dtype):
    if dtype is not None:
        data_d = data_d.astype(dtype)
    base_file_d = os.path.splitext(os.path.basename(file_d))[0]
    fits.writeto(f'prep_{base_file_d}.fits', data_d, overwrite=True)
    if file_e:
        if dtype is not None:
            data_e = data_e.astype(dtype)
        base_file_e = os.path.splitext(os.path.basename(file_e))[0]
        fits.writeto(f'prep_{base_file_e}.fits', data_e, overwrite=True)
    if file_m:
        if dtype is not None:
            data_m = data_m.astype(dtype)
        base_file_m = os.path.splitext(os.path.basename(file_m))[0]
        fits.writeto(f'prep_{base_file_m}.fits', data_m, overwrite=True)


def _crop_data(data_d, data_e, data_m, axis, range):
    s = [slice(None), ] * data_d.ndim
    s[axis] = slice(range[0], range[1])
    data_d = data_d[tuple(s)]
    if data_e is not None:
        data_e = data_e[tuple(s)]
    if data_m is not None:
        data_m = data_m[tuple(s)]
    return data_d, data_e, data_m


def _make_mask(data_d, data_e, data_m):
    mask = np.ones_like(data_d)
    mask *= np.isfinite(data_d)
    mask *= np.isfinite(data_e) if data_e else 1
    mask *= np.isfinite(data_m) if data_m else 1
    return mask


def _compare_nan_array(func, ary, threshold):
    out = ~np.isnan(ary)
    out[out] = func(ary[out], threshold)
    return out


def _make_mask_clip_min(data, min_value):
    return ~_compare_nan_array(np.less, data, min_value)


def _make_mask_clip_max(data, max_value):
    return ~_compare_nan_array(np.greater, data, max_value)


def _make_mask_clip_sig(data, sigma, maxiters):
    return ~stats.sigma_clip(data, sigma=sigma, maxiters=maxiters).mask


def _make_mask_clip_ccl(data, lcount, pcount, lratio):
    labels = skimage.measure.label(data)
    props = skimage.measure.regionprops(labels)
    props.sort(key=lambda x: x.area, reverse=True)
    if lcount is not None:
        props = props[0:lcount]
    if pcount is not None:
        props = [p for p in props if p.area >= pcount]
    if lratio is not None:
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


def _minify_data(data_d, data_e, data_m, mask):
    slices = tuple([slice(
        indices.min(), indices.max() + 1) for indices in mask.nonzero()])
    data_d = data_d[slices]
    if data_e is not None:
        data_e = data_e[slices]
    if data_m is not None:
        data_m = data_m[slices]
    return data_d, data_e, data_m


def prep_image(
        file_d, file_e, file_m, unit_d, unit_e,
        roi_spat, clip_min, clip_max, ccl_lcount, ccl_pcount, ccl_lratio,
        sclip_sigma, sclip_iters, minify, dtype):

    data_d, data_e, data_m, _, _ = _read_data(
        file_d, file_e, file_m)

    if roi_spat is not None:
        xrange = roi_spat[0:2]
        yrange = roi_spat[2:4]
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 2, xrange)
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 1, yrange)

    mask = _make_mask(data_d, data_e, data_m)

    if clip_min is not None:
        mask *= _make_mask_clip_min(data_d, clip_min)
    if clip_max is not None:
        mask *= _make_mask_clip_max(data_d, clip_max)
    if sclip_sigma is not None:
        mask *= _make_mask_clip_sig(data_d, sclip_sigma, sclip_iters)
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask *= _make_mask_clip_ccl(
            np.isfinite(data_d), ccl_lcount, ccl_pcount, ccl_lratio)

    _apply_mask(data_d, data_e, data_m, mask)

    if minify:
        data_d, data_e, data_m = _minify_data(
            data_d, data_e, data_m, mask)

    _save_data(file_d, file_e, file_m, data_d, data_e, data_m, dtype)


def prep_lslit(
        file_d, file_e, file_m, unit_d, unit_e,
        roi_spat, roi_spec, clip_min, clip_max,
        ccl_lcount, ccl_pcount, ccl_lratio,
        sclip_sigma, sclip_iters, minify, dtype):

    data_d, data_e, data_m, _, _ = _read_data(
        file_d, file_e, file_m)

    if roi_spat is not None:
        xrange = roi_spat
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 0, xrange)
    if roi_spec is not None:
        srange = roi_spec
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 1, srange)

    mask = _make_mask(data_d, data_e, data_m)

    if clip_min is not None:
        mask *= _make_mask_clip_min(data_d, clip_min)
    if clip_max is not None:
        mask *= _make_mask_clip_max(data_d, clip_max)
    if sclip_sigma is not None:
        mask *= _make_mask_clip_sig(data_d, sclip_sigma, sclip_iters)
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask *= _make_mask_clip_ccl(
            np.isfinite(data_d), ccl_lcount, ccl_pcount, ccl_lratio)

    _apply_mask(data_d, data_e, data_m, mask)

    if minify:
        data_d, data_e, data_m = _minify_data(
            data_d, data_e, data_m, mask)

    _save_data(file_d, file_e, file_m, data_d, data_e, data_m, dtype)


def prep_mmaps(
        file_d, file_e, file_m, unit_d, unit_e,
        roi_spat, clip_min, clip_max, ccl_lcount, ccl_pcount, ccl_lratio,
        sclip_sigma, sclip_iters, minify, dtype):

    nmmaps = len(file_d)

    if file_e is None:
        file_e = [None] * nmmaps
    if file_m is None:
        file_m = [None] * nmmaps

    data_d = [None] * nmmaps
    data_e = [None] * nmmaps
    data_m = [None] * nmmaps

    for i in range(nmmaps):

        data_d[i], data_e[i], data_m[i], _, _ = _read_data(
            file_d[i], file_e[i], file_m[i])

        if roi_spat is not None:
            xrange = roi_spat[0:2]
            yrange = roi_spat[2:4]
            data_d[i], data_e[i], data_m[i] = _crop_data(
                data_d[i], data_e[i], data_m[i], 1, xrange)
            data_d[i], data_e[i], data_m[i] = _crop_data(
                data_d[i], data_e[i], data_m[i], 0, yrange)

    mask = np.ones_like(data_d[0])

    for i in range(nmmaps):

        mask *= _make_mask(data_d[i], data_e[i], data_m[i])
        if clip_min is not None:
            mask *= _make_mask_clip_min(data_d[i], clip_min[i])
        if clip_max is not None:
            mask *= _make_mask_clip_max(data_d[i], clip_max[i])
        if sclip_sigma is not None:
            mask *= _make_mask_clip_sig(data_d[i], sclip_sigma, sclip_iters)
        if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
            mask *= _make_mask_clip_ccl(
                np.isfinite(data_d[i]), ccl_lcount, ccl_pcount, ccl_lratio)

    for i in range(nmmaps):

        _apply_mask(data_d[i], data_e[i], data_m[i], mask)

        if minify:
            data_d[i], data_e[i], data_m[i] = _minify_data(
                data_d[i], data_e[i], data_m[i], mask)

        _save_data(file_d[i], file_e[i], file_m[i], data_d[i], data_e[i], data_m[i], dtype)


def prep_scube(
        file_d, file_e, file_m, unit_d, unit_e,
        roi_spat, roi_spec, clip_min, clip_max,
        ccl_lcount, ccl_pcount, ccl_lratio,
        sclip_sigma, sclip_iters, minify, dtype):

    data_d, data_e, data_m, _, _ = _read_data(
        file_d, file_e, file_m)

    if roi_spat is not None:
        xrange = roi_spat[0:2]
        yrange = roi_spat[2:4]
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 2, xrange)
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 1, yrange)
    if roi_spec is not None:
        srange = roi_spec
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 0, srange)

    mask = _make_mask(data_d, data_e, data_m)

    if clip_min is not None:
        mask *= _make_mask_clip_min(data_d, clip_min)
    if clip_max is not None:
        mask *= _make_mask_clip_max(data_d, clip_max)
    if sclip_sigma is not None:
        mask *= _make_mask_clip_sig(data_d, sclip_sigma, sclip_iters)
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask *= _make_mask_clip_ccl(
            np.isfinite(data_d), ccl_lcount, ccl_pcount, ccl_lratio)

    _apply_mask(data_d, data_e, data_m, mask)

    if minify:
        data_d, data_e, data_m = _minify_data(
            data_d, data_e, data_m, mask)

    _save_data(file_d, file_e, file_m, data_d, data_e, data_m, dtype)
