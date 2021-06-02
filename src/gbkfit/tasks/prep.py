
import os

import astropy.io.fits as fits
import astropy.stats as stats
import numpy as np
import skimage.measure


def _read_data(file_d, file_e, file_m):
    # Load data and headers.
    data_d = fits.getdata(file_d)
    data_e = fits.getdata(file_e) if file_e else None
    data_m = fits.getdata(file_m) if file_m else None
    header_d = fits.getheader(file_d)
    header_e = fits.getheader(file_e) if file_e else None
    header_m = fits.getheader(file_m) if file_m else None
    # Discard dimensions of length = 1
    data_d = np.squeeze(data_d)
    if data_e is not None:
        data_e = np.squeeze(data_e)
    if data_m is not None:
        data_m = np.squeeze(data_m)
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
        dtype):
    basename = os.path.basename
    splitext = os.path.splitext
    data_d = data_d.astype(dtype)
    file_d = splitext(basename(file_d))[0]
    fits.writeto(f'prep_{file_d}.fits', data_d, header_d, overwrite=True)
    if data_e is not None:
        data_e = data_e.astype(dtype)
        file_e = splitext(basename(file_e))[0]
        fits.writeto(f'prep_{file_e}.fits', data_e, header_e, overwrite=True)
    if data_m is not None:
        data_m = data_m.astype(dtype)
        file_m = splitext(basename(file_m))[0] if file_m else file_d + '_mask'
        fits.writeto(f'prep_{file_m}.fits', data_m, header_m, overwrite=True)


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
    mask = np.ones_like(data_d)
    mask *= np.isfinite(data_d)
    if data_e is not None:
        mask *= np.isfinite(data_e)
    if data_m is not None:
        mask *= np.isfinite(data_m)
    return mask


def _compare_nan_array(func, ary, threshold):
    out = ~np.isnan(ary)
    out[out] = func(ary[out], threshold)
    return out


def _make_mask_clip_min(data, min_value):
    return ~_compare_nan_array(np.less, data, min_value)


def _make_mask_clip_max(data, max_value):
    return ~_compare_nan_array(np.greater, data, max_value)


def _make_mask_clip_sig(data, sigma, maxiters, invert):
    mask = stats.sigma_clip(data, sigma=sigma, maxiters=maxiters).mask
    return mask if not invert else ~mask


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


def _pad_data(data_d, data_e, data_m, size, value_d, value_e, value_m):
    data_d = np.pad(data_d, size, 'constant', constant_values=value_d)
    if data_e is not None:
        data_e = np.pad(data_e, size, 'constant', constant_values=value_e)
    if data_m is not None:
        data_m = np.pad(data_m, size, 'constant', constant_values=value_m)
    return data_d, data_e, data_m


def prep_image(
        file_d, file_e, file_m,
        roi_spat, clip_min, clip_max, ccl_lcount, ccl_pcount, ccl_lratio,
        sclip_sigma, sclip_iters, minify, nanpad, dtype):

    (data_d, header_d,
     data_e, header_e,
     data_m, header_m) = _read_data(file_d, file_e, file_m)

    if roi_spat is not None:
        xrange = roi_spat[0:2]
        yrange = roi_spat[2:4]
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 1, xrange)
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 0, yrange)

    mask = _make_mask(data_d, data_e, data_m)

    if clip_min is not None:
        mask *= _make_mask_clip_min(data_d, clip_min)
    if clip_max is not None:
        mask *= _make_mask_clip_max(data_d, clip_max)
    if sclip_sigma is not None:
        mask *= _make_mask_clip_sig(
            data_d, sclip_sigma, sclip_iters, False)
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask *= _make_mask_clip_ccl(
            np.isfinite(data_d), ccl_lcount, ccl_pcount, ccl_lratio)

    _apply_mask(data_d, data_e, data_m, mask)

    if minify:
        data_d, data_e, data_m = _minify_data(
            data_d, data_e, data_m, mask)

    if nanpad:
        _pad_data(data_d, data_e, data_m, nanpad, np.nan, np.nan, 0)

    _save_data(
        file_d, data_d, header_d,
        file_e, data_e, header_e,
        file_m, data_m, header_m,
        dtype)


def prep_lslit(
        file_d, file_e, file_m,
        roi_spat, roi_spec, clip_min, clip_max,
        ccl_lcount, ccl_pcount, ccl_lratio,
        sclip_sigma, sclip_iters, minify, nanpad, dtype):

    (data_d, header_d,
     data_e, header_e,
     data_m, header_m) = _read_data(file_d, file_e, file_m)

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
        mask *= _make_mask_clip_sig(
            data_d, sclip_sigma, sclip_iters, False)
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask *= _make_mask_clip_ccl(
            np.isfinite(data_d), ccl_lcount, ccl_pcount, ccl_lratio)

    _apply_mask(data_d, data_e, data_m, mask)

    if minify:
        data_d, data_e, data_m = _minify_data(
            data_d, data_e, data_m, mask)

    if nanpad:
        _pad_data(data_d, data_e, data_m, nanpad, np.nan, np.nan, 0)

    _save_data(
        file_d, data_d, header_d,
        file_e, data_e, header_e,
        file_m, data_m, header_m,
        dtype)


def prep_mmaps(
        orders, file_d, file_e, file_m,
        roi_spat, clip_min, clip_max, ccl_lcount, ccl_pcount, ccl_lratio,
        sclip_sigma, sclip_iters, minify, nanpad, dtype):

    nmmaps = len(file_d)
    if file_e is None:
        file_e = [None] * nmmaps
    if file_m is None:
        file_m = [None] * nmmaps

    data_d = []
    data_e = []
    data_m = []
    header_d = []
    header_e = []
    header_m = []
    for i in range(nmmaps):
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

    for i in range(nmmaps):
        mask *= _make_mask(data_d[i], data_e[i], data_m[i])
        if clip_min is not None:
            mask *= _make_mask_clip_min(data_d[i], clip_min[i])
        if clip_max is not None:
            mask *= _make_mask_clip_max(data_d[i], clip_max[i])
        if sclip_sigma is not None:
            mask *= _make_mask_clip_sig(
                data_d[i], sclip_sigma, sclip_iters, orders[i] == 0)
        if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
            mask *= _make_mask_clip_ccl(
                np.isfinite(data_d[i]), ccl_lcount, ccl_pcount, ccl_lratio)

    for i in range(nmmaps):

        _apply_mask(data_d[i], data_e[i], data_m[i], mask)

        if minify:
            data_d[i], data_e[i], data_m[i] = _minify_data(
                data_d[i], data_e[i], data_m[i], mask)

        if nanpad:
            _pad_data(data_d, data_e, data_m, nanpad, np.nan, np.nan, 0)

        _save_data(
            file_d[i], data_d[i], header_d[i],
            file_e[i], data_e[i], header_e[i],
            file_m[i], data_m[i], header_m[i],
            dtype)


def prep_scube(
        file_d, file_e, file_m,
        roi_spat, roi_spec, clip_min, clip_max,
        ccl_lcount, ccl_pcount, ccl_lratio,
        sclip_sigma, sclip_iters, minify, nanpad, dtype):

    (data_d, header_d,
     data_e, header_e,
     data_m, header_m) = _read_data(file_d, file_e, file_m)

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
        mask *= _make_mask_clip_sig(
            data_d, sclip_sigma, sclip_iters, False)
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask *= _make_mask_clip_ccl(
            np.isfinite(data_d), ccl_lcount, ccl_pcount, ccl_lratio)

    _apply_mask(data_d, data_e, data_m, mask)

    if minify:
        data_d, data_e, data_m = _minify_data(
            data_d, data_e, data_m, mask)

    if nanpad:
        _pad_data(data_d, data_e, data_m, nanpad, np.nan, np.nan, 0)

    _save_data(
        file_d, data_d, header_d,
        file_e, data_e, header_e,
        file_m, data_m, header_m,
        dtype)
