"""
Measurement sets of CASA: the interferometric visibilities of a field in
a spectral window, as Stokes I, with their errors and where they are
measured (see read_measurement_set). They are read by python-casacore,
which is optional: it is imported only when a measurement set is read.
"""

import dataclasses
import os.path
from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypeAlias

import astropy.units
import numpy as np

from gbkfit.config import Choices
from gbkfit.utils.numutils import MAD_TO_STD


__all__ = [
    'MSErrorSource',
    'MSVisibilities',
    'read_measurement_set'
]


# Where the errors of the visibilities of a measurement set come from
# (see read_measurement_set)
MSErrorSource: TypeAlias = Annotated[
    Literal['weights', 'scatter'],
    Choices(weights='From the weights', scatter='From the scatter')]

# The frames of the measurement sets, by their codes (MEAS_FREQ_REF)
_MS_FRAMES = {
    0: 'REST', 1: 'LSRK', 2: 'LSRD', 3: 'BARY', 4: 'GEO', 5: 'TOPO',
    6: 'GALACTO', 7: 'LGROUP', 8: 'CMB'}

# The codes of the Stokes I and of the pairs of parallel hands (I = (a +
# b) / 2) of the measurement sets (CORR_TYPE)
_MS_STOKES_I = 1
_MS_PARALLEL_HANDS = ((5, 8), (9, 12))


@dataclasses.dataclass(frozen=True)
class MSVisibilities:
    """
    The visibilities of a field of a measurement set in a spectral window,
    as Stokes I, and where they are measured (see read_measurement_set).

    Attributes
    ----------
    data : np.ndarray
        The complex visibilities (Jy), of shape (nchannels, nrows).
    mask : np.ndarray
        True where they are valid (flagged in none of the products of
        which they are made), of their shape.
    error_re, error_im : np.ndarray
        The errors of their real and imaginary parts, of their shape
        (inf where the weight is 0; NaN where they cannot be estimated
        from the scatter).
    uvw : np.ndarray
        The baselines b of the rows (m), of shape (nrows, 3): -UVW, as
        gbkfit takes them (see dataset.UVCoverage).
    frequencies : np.ndarray
        The frequencies of the channels (Hz).
    phase_centre : np.ndarray
        The RA and Dec of the phase centre of the field (degrees).
    frame : str
        The rest frame of the frequencies (e.g. 'LSRK'; 'unknown' if the
        set gives a code of none).
    rest : str or Quantity
        The rest frequency of the line: that given, or that of the source
        of the field (see read_measurement_set).
    """
    data: np.ndarray
    mask: np.ndarray
    error_re: np.ndarray
    error_im: np.ndarray
    uvw: np.ndarray
    frequencies: np.ndarray
    phase_centre: np.ndarray
    frame: str
    rest: str | astropy.units.Quantity


def read_measurement_set(
        ms: str,
        field: int | str | None = None,
        spw: int = 0,
        column: str = 'DATA',
        rest: str | astropy.units.Quantity | None = None,
        error: MSErrorSource = 'weights',
        frames: Sequence[str] | None = None
) -> MSVisibilities:
    """
    Read the cross-correlations of a field of a measurement set in a
    spectral window (python-casacore), as Stokes I: its own visibilities,
    or the mean of those of the parallel hands (RR and LL, or XX and YY),
    left out where either is flagged. The visibilities of a measurement
    set are sums of exp(+2 pi i UVW . (s - p)), so their baselines b are
    -UVW (checked against those that CASA predicts; see
    tests/data/visibilities).

    Not supported yet: several fields or spectral windows at once.

    Parameters
    ----------
    ms : str
        The path of the measurement set.
    field : int or str, optional
        The field, by its index or name; may be left out if there is
        one.
    spw : int, optional
        The spectral window; by default the first.
    column : str, optional
        The column of the visibilities (e.g. 'CORRECTED_DATA').
    rest : str or Quantity, optional
        The rest frequency of the line; by default, that of the source of
        the field (REST_FREQUENCY of the table SOURCE).
    error : str, optional
        The errors of each part: 'weights' (by default), 1 / sqrt(weight),
        of WEIGHT_SPECTRUM if given, else of WEIGHT; or 'scatter',
        estimated from the visibilities themselves (see
        _scatter_errors).
    frames : Sequence of str, optional
        The rest frames of the frequencies that are accepted, if not any
        (checked before the visibilities are read).

    Returns
    -------
    MSVisibilities
        The visibilities.

    Raises
    ------
    ValueError
        If the field, the spectral window or the column are not in the
        measurement set, it has neither Stokes I nor a pair of parallel
        hands, its frequencies are not in one of the frames, or it has no
        rest frequency and none is given.
    ModuleNotFoundError
        If python-casacore is not installed.
    """
    try:
        from casacore import tables
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "reading measurement sets needs python-casacore: pip "
            "install 'gbkfit[casacore]'") from e

    # TODO: several fields or spectral windows at once (for now, one
    # read of each)
    def subtable(name: str) -> Any:
        return tables.table(os.path.join(ms, name), ack=False)
    with subtable('FIELD') as fields:
        names = list(fields.getcol('NAME'))
        index = _ms_field(field, names)
        phase_centre = np.degrees(fields.getcol('PHASE_DIR')[index, 0])
        source = fields.getcol('SOURCE_ID')[index]
    with subtable('SPECTRAL_WINDOW') as windows:
        if not 0 <= spw < windows.nrows():
            raise ValueError(
                f"the measurement set has {windows.nrows()} spectral "
                f"windows; spw {spw} is not one")
        frequencies = windows.getcell('CHAN_FREQ', spw)
        frame = _MS_FRAMES.get(
            int(windows.getcell('MEAS_FREQ_REF', spw)), 'unknown')
    if frames is not None and frame not in frames:
        raise ValueError(
            f"the frequencies must be in the rest frame "
            f"{' or '.join(frames)}; they are in {frame} (regrid them, "
            f"e.g. by CASA mstransform with outframe='{frames[0]}')")
    with subtable('DATA_DESCRIPTION') as descriptions:
        description = list(descriptions.getcol('SPECTRAL_WINDOW_ID')) \
            .index(spw)
        polarization = descriptions.getcell('POLARIZATION_ID', description)
    with subtable('POLARIZATION') as polarizations:
        products = list(polarizations.getcell('CORR_TYPE', polarization))
    if rest is None:
        rest = _ms_rest(subtable, source)
    with tables.table(ms, ack=False) as main:
        if column not in main.colnames():
            raise ValueError(
                f"the measurement set has no column {column}")
        rows = main.query(
            f'FIELD_ID == {index} && DATA_DESC_ID == {description} '
            f'&& ANTENNA1 != ANTENNA2')
        uvw = -rows.getcol('UVW')
        visibilities = rows.getcol(column)
        flags = rows.getcol('FLAG') \
            | rows.getcol('FLAG_ROW')[:, None, None]
        weights = rows.getcol('WEIGHT_SPECTRUM') \
            if 'WEIGHT_SPECTRUM' in rows.colnames() \
            and rows.iscelldefined('WEIGHT_SPECTRUM', 0) \
            else rows.getcol('WEIGHT')[:, None, :]
        baselines = rows.getcol('ANTENNA1'), rows.getcol('ANTENNA2')
        times = rows.getcol('TIME'), rows.getcol('INTERVAL')
        rows.close()
    stokes, sigma, mask = _ms_stokes_i(
        visibilities, flags, weights, products)
    sigma_re = sigma_im = sigma
    if error == 'scatter':
        sigma_re, sigma_im = _scatter_errors(
            np.where(mask, stokes, np.nan), *baselines, *times)
    # (read by row, as the set stores them; returned by channel, of shape
    # (nchannels, nrows), as gbkfit holds visibilities)
    return MSVisibilities(
        stokes.T, mask.T, sigma_re.T, sigma_im.T, uvw, frequencies,
        phase_centre, frame, rest)


def _ms_field(field: int | str | None, names: list[str]) -> int:
    """Return the index of a field of a measurement set of the names."""
    if field is None:
        if len(names) != 1:
            raise ValueError(
                f"the measurement set has the fields {names}; give one")
        return 0
    if isinstance(field, str):
        if field not in names:
            raise ValueError(
                f"the measurement set has the fields {names}, not {field!r}")
        return names.index(field)
    if not 0 <= field < len(names):
        raise ValueError(
            f"the measurement set has {len(names)} fields; field {field} is "
            f"not one")
    return field


def _ms_rest(subtable: Any, source: int) -> astropy.units.Quantity:
    """
    Return the rest frequency of a source of a measurement set (the first
    of the table SOURCE, if it has one).
    """
    error = ValueError(
        "the measurement set has no rest frequency for the field; give it "
        "(option rest)")
    try:
        with subtable('SOURCE') as sources:
            rows = [i for i, x in enumerate(sources.getcol('SOURCE_ID'))
                    if x == source]
            if not rows or 'REST_FREQUENCY' not in sources.colnames() \
                    or not sources.iscelldefined('REST_FREQUENCY', rows[0]):
                raise error
            values = sources.getcell('REST_FREQUENCY', rows[0])
    except RuntimeError as e:
        raise error from e
    if len(values) == 0:
        raise error
    return float(values[0]) * astropy.units.Hz


def _scatter_errors(
        visibilities: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        time: np.ndarray,
        interval: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the errors of the real and the imaginary parts of visibilities
    of shape (nrows, nchannels) (NaN where flagged), estimated from the
    second differences of those of each baseline at consecutive times.

    Those of a baseline at three consecutive times (in one scan: at most
    1.5 integrations apart) see nearly the same sky, so V(t - 1) - 2 V(t)
    + V(t + 1) is noise of 6 times the variance, without the part of the
    sky that changes linearly in time. The error of each part, of each
    baseline, is the robust dispersion (1.4826 times the median absolute
    deviation) of these over all its times and channels, over sqrt(6);
    that of baselines with fewer than 5 is missing (NaN).
    """
    errors = np.full((2,) + visibilities.shape, np.nan)
    baselines = antenna1 * (antenna2.max() + 1) + antenna2
    for baseline in np.unique(baselines):
        rows = np.flatnonzero(baselines == baseline)
        rows = rows[np.argsort(time[rows])]
        # The triples of consecutive times in one scan
        apart = np.diff(time[rows]) <= 1.5 * np.maximum(
            interval[rows[1:]], interval[rows[:-1]])
        middle = np.flatnonzero(apart[:-1] & apart[1:]) + 1
        differences = visibilities[rows[middle - 1]] \
            - 2 * visibilities[rows[middle]] + visibilities[rows[middle + 1]]
        for part, values in enumerate((differences.real, differences.imag)):
            values = values[np.isfinite(values)]
            if values.size >= 5:
                deviation = np.median(np.abs(values - np.median(values)))
                errors[part][rows] = MAD_TO_STD * deviation / np.sqrt(6)
    return errors[0], errors[1]


def _ms_stokes_i(
        visibilities: np.ndarray,
        flags: np.ndarray,
        weights: np.ndarray,
        products: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the Stokes I of the visibilities of a measurement set, of shape
    (nrows, nchannels, nproducts), its errors (of each part), and its
    mask, of shape (nrows, nchannels).
    """
    with np.errstate(divide='ignore'):
        sigmas = np.where(weights > 0, 1 / np.sqrt(weights), np.inf)
    sigmas = np.broadcast_to(sigmas, visibilities.shape)
    if _MS_STOKES_I in products:
        i = products.index(_MS_STOKES_I)
        return visibilities[..., i], sigmas[..., i], ~flags[..., i]
    for a, b in _MS_PARALLEL_HANDS:
        if a in products and b in products:
            a, b = products.index(a), products.index(b)
            stokes = (visibilities[..., a] + visibilities[..., b]) / 2
            error = np.sqrt(sigmas[..., a] ** 2 + sigmas[..., b] ** 2) / 2
            return stokes, error, ~(flags[..., a] | flags[..., b])
    raise ValueError(
        f"the measurement set has neither Stokes I nor a pair of parallel "
        f"hands; its products are {products} (CASA codes)")
