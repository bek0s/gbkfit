import os.path
from collections.abc import Sequence
from typing import Any

import astropy.units
import numpy as np

from gbkfit.utils import gridutils, parseutils, unitutils
from gbkfit.utils.parseutils import ConfigError
from .base import Dataset
from .data import Data


__all__ = [
    'DatasetVisibilities'
]


# The rest frames of frequencies whose channels hold the velocities of a
# galaxy; topocentric ones move with the Earth (by up to 30 km/s in a
# year, and 0.5 km/s in a day)
_FRAMES = ('LSRK', 'BARY')

# The frames of the measurement sets, by their codes (MEAS_FREQ_REF)
_MS_FRAMES = {
    0: 'REST', 1: 'LSRK', 2: 'LSRD', 3: 'BARY', 4: 'GEO', 5: 'TOPO',
    6: 'GALACTO', 7: 'LGROUP', 8: 'CMB'}

# The codes of the Stokes I and of the pairs of parallel hands (I = (a +
# b) / 2) of the measurement sets (CORR_TYPE)
_MS_STOKES_I = 1
_MS_PARALLEL_HANDS = ((5, 8), (9, 12))


class DatasetVisibilities(Dataset):
    """
    Interferometric visibilities of Stokes I: their real and imaginary
    parts (the items re and im, of shape (nchannels, nrows), with the error
    of each part), at the baselines of the rows and the frequencies of the
    channels, about a phase centre.

    Its configuration is a file of the arrays of the dataset (option file;
    see from_file and dump), or a measurement set and the options of
    from_ms (option ms).

    Parameters
    ----------
    re, im : Data
        The real and imaginary parts, of shape (nchannels, nrows), with
        the error of each part (e.g. 1 / sqrt(weight) of a measurement
        set).
    uvw : numpy.ndarray
        The baselines (u, v, w) of the rows, in m, in the frame of the
        phase centre (u to the east, v to the north, w to it), of shape
        (nrows, 3).
    frequencies : Sequence of float
        The frequencies of the channels (Hz): uniform, in the rest frame
        given.
    phase_centre : Sequence of float
        The RA and Dec of the phase centre (degrees).
    frame : str
        The rest frame of the frequencies: 'LSRK' or 'BARY' (for TOPO
        ones, regrid them, e.g. by CASA mstransform with outframe='LSRK').
    rest : str or Quantity
        The rest frequency of the line, of the radio velocities of the
        channels (see gridutils.make_rest; e.g. '1420.405752 MHz').

    Raises
    ------
    ConfigError
        If the items are not re and im with errors, of one row of each
        baseline, the baselines are not finite, the frequencies are not at least two,
        positive and uniform, the phase centre is not an RA and a Dec, the
        frame is not one of those, or the rest is not a frequency.
    """

    ndim = 2

    @staticmethod
    def type() -> str:
        return 'visibilities'

    @classmethod
    def load(
            cls, info: dict[str, Any], prefix: str = ''
    ) -> 'DatasetVisibilities':
        options = parseutils.parse_options(
            info, required=set(), optional={'file', 'ms', 'field', 'spw',
                                            'column', 'rest'})
        if ('file' in options) == ('ms' in options):
            raise ConfigError(
                "the visibilities are read from a file (option file) or a "
                "measurement set (option ms), one of them")
        if 'file' in options:
            ms_options = set(options) - {'file'}
            if ms_options:
                raise ConfigError(
                    f"the options {sorted(ms_options)} are of measurement "
                    f"sets (option ms)")
            return cls.from_file(prefix + options['file'])
        return cls.from_ms(
            prefix + options.pop('ms'), **options)

    @classmethod
    def from_file(cls, file: str) -> 'DatasetVisibilities':
        """
        Read the dataset from a numpy file (npz) of its arrays: data, the
        complex visibilities (nchannels, nrows); error, the error of each
        of their parts (nchannels, nrows); optionally mask (0 for the
        visibilities left out) and unit (e.g. 'Jy'); and uvw, frequencies,
        phase_centre, frame and rest (Hz), as in the class.

        Parameters
        ----------
        file : str
            The filename.

        Returns
        -------
        DatasetVisibilities
            The dataset.

        Raises
        ------
        ConfigError
            If an array is missing (see the class).
        """
        with np.load(file) as arrays:
            arrays = dict(arrays)
        missing = {
            'data', 'error', 'uvw', 'frequencies', 'phase_centre', 'frame',
            'rest'} - set(arrays)
        if missing:
            raise ConfigError(
                f"the file of visibilities {file} has no arrays "
                f"{sorted(missing)}")
        data, mask = arrays['data'], arrays.get('mask')
        unit = str(arrays['unit']) if 'unit' in arrays else None
        return cls(
            Data(data.real, mask, arrays['error'], unit),
            Data(data.imag, mask, arrays['error'], unit),
            arrays['uvw'], arrays['frequencies'], arrays['phase_centre'],
            str(arrays['frame']), float(arrays['rest']) * astropy.units.Hz)

    @classmethod
    def from_ms(
            cls,
            ms: str,
            field: int | str | None = None,
            spw: int = 0,
            column: str = 'DATA',
            rest: str | astropy.units.Quantity | None = None
    ) -> 'DatasetVisibilities':
        """
        Read the dataset from a measurement set (python-casacore): the
        cross-correlations of a field in a spectral window, as Stokes I:
        its own visibilities, or the mean of those of the parallel hands
        (RR and LL, or XX and YY), left out where either is flagged. The
        errors of each part are 1 / sqrt(weight), of WEIGHT_SPECTRUM if
        given, else of WEIGHT.

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
            The rest frequency (see the class); by default, that of the
            source of the field (REST_FREQUENCY of the table SOURCE).

        Returns
        -------
        DatasetVisibilities
            The dataset, in Jy.

        Raises
        ------
        ConfigError
            If the field, the spectral window or the column are not in the
            measurement set, it has neither Stokes I nor a pair of parallel
            hands, or no rest frequency (see the class).
        ModuleNotFoundError
            If python-casacore is not installed.
        """
        try:
            from casacore import tables
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "reading measurement sets needs python-casacore: pip "
                "install 'gbkfit[casacore]'") from e

        def subtable(name: str) -> Any:
            return tables.table(os.path.join(ms, name), ack=False)
        with subtable('FIELD') as fields:
            names = list(fields.getcol('NAME'))
            index = _ms_field(field, names)
            phase_centre = np.degrees(fields.getcol('PHASE_DIR')[index, 0])
            source = fields.getcol('SOURCE_ID')[index]
        with subtable('SPECTRAL_WINDOW') as windows:
            if not 0 <= spw < windows.nrows():
                raise ConfigError(
                    f"the measurement set has {windows.nrows()} spectral "
                    f"windows; spw {spw} is not one")
            frequencies = windows.getcell('CHAN_FREQ', spw)
            frame = _MS_FRAMES.get(
                int(windows.getcell('MEAS_FREQ_REF', spw)), 'unknown')
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
                raise ConfigError(
                    f"the measurement set has no column {column}")
            rows = main.query(
                f'FIELD_ID == {index} && DATA_DESC_ID == {description} '
                f'&& ANTENNA1 != ANTENNA2')
            uvw = rows.getcol('UVW')
            visibilities = rows.getcol(column)
            flags = rows.getcol('FLAG') | rows.getcol('FLAG_ROW')[:, None, None]
            weights = rows.getcol('WEIGHT_SPECTRUM') \
                if 'WEIGHT_SPECTRUM' in rows.colnames() \
                and rows.iscelldefined('WEIGHT_SPECTRUM', 0) \
                else rows.getcol('WEIGHT')[:, None, :]
            rows.close()
        stokes, error, mask = _ms_stokes_i(
            visibilities, flags, weights, products)
        return cls(
            Data(stokes.real.T, mask.T, error.T, 'Jy'),
            Data(stokes.imag.T, mask.T, error.T, 'Jy'),
            uvw, frequencies, phase_centre, frame, rest)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        filename = f'{prefix}visibilities.npz'
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f"{filename} exists")
        re, im = self['re'], self['im']
        arrays = dict(
            data=re.data() + 1j * im.data(), error=re.error(), mask=re.mask(),
            uvw=self._uvw, frequencies=self._frequencies,
            phase_centre=np.array(self._phase_centre), frame=self._frame,
            rest=self._rest.to_value(astropy.units.Hz))
        if re.unit() is not None:
            arrays['unit'] = unitutils.format_unit(re.unit())
        with open(filename, 'wb') as file:
            np.savez(file, **arrays)
        return dict(
            type=self.type(),
            file=filename if dump_path else os.path.basename(filename))

    def __init__(
            self,
            re: Data,
            im: Data,
            uvw: np.ndarray,
            frequencies: Sequence[float],
            phase_centre: Sequence[float],
            frame: str,
            rest: str | astropy.units.Quantity
    ):
        super().__init__(dict(re=re, im=im))
        if re.error() is None or im.error() is None:
            raise ConfigError(
                "visibilities need the errors of their parts (e.g. 1 / "
                "sqrt(weight) of a measurement set)")
        nchannels, nrows = re.shape()
        uvw = np.array(uvw, float)
        frequencies = np.array(frequencies, float)
        if uvw.shape != (nrows, 3) or not np.isfinite(uvw).all():
            raise ConfigError(
                f"the baselines must be finite, one (u, v, w) for each of "
                f"the {nrows} rows; their shape is {uvw.shape}")
        if frequencies.shape != (nchannels,):
            raise ConfigError(
                f"there must be a frequency for each of the {nchannels} "
                f"channels; there are {frequencies.size}")
        steps = np.diff(frequencies)
        if nchannels < 2 or not np.all(frequencies > 0) or not np.allclose(
                steps, steps[0], rtol=1e-6, atol=0) or steps[0] == 0:
            raise ConfigError(
                "the frequencies must be at least two, positive and uniform "
                "(e.g. regridded by CASA mstransform)")
        if np.shape(phase_centre) != (2,):
            raise ConfigError(
                f"the phase centre must be an RA and a Dec; it is "
                f"{phase_centre}")
        if frame not in _FRAMES:
            raise ConfigError(
                f"the frequencies must be in the rest frame {' or '.join(_FRAMES)}"
                f"; they are in {frame} (regrid them, e.g. by CASA mstransform "
                f"with outframe='LSRK')")
        rest = gridutils.make_rest(rest)
        if rest is None or rest.unit != astropy.units.Hz:
            raise ConfigError(
                f"the rest of visibilities must be a frequency; it is {rest}")
        self._uvw = uvw
        self._frequencies = frequencies
        self._phase_centre = tuple(float(x) for x in phase_centre)
        self._frame = frame
        self._rest = rest

    def uvw(self) -> np.ndarray:
        """Return the baselines of the rows (m), of shape (nrows, 3)."""
        return self._uvw

    def frequencies(self) -> np.ndarray:
        """Return the frequencies of the channels (Hz)."""
        return self._frequencies

    def phase_centre(self) -> tuple[float, float]:
        """Return the RA and Dec of the phase centre (degrees)."""
        return self._phase_centre

    def frame(self) -> str:
        """Return the rest frame of the frequencies."""
        return self._frame

    def rest(self) -> astropy.units.Quantity:
        """Return the rest frequency of the line (Hz)."""
        return self._rest


def _ms_field(field: int | str | None, names: list[str]) -> int:
    """Return the index of a field of a measurement set of the names."""
    if field is None:
        if len(names) != 1:
            raise ConfigError(
                f"the measurement set has the fields {names}; give one")
        return 0
    if isinstance(field, str):
        if field not in names:
            raise ConfigError(
                f"the measurement set has the fields {names}, not {field!r}")
        return names.index(field)
    if not 0 <= field < len(names):
        raise ConfigError(
            f"the measurement set has {len(names)} fields; field {field} is "
            f"not one")
    return field


def _ms_rest(subtable: Any, source: int) -> astropy.units.Quantity:
    """
    Return the rest frequency of a source of a measurement set (the first
    of the table SOURCE, if it has one).
    """
    error = ConfigError(
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
    raise ConfigError(
        f"the measurement set has neither Stokes I nor a pair of parallel "
        f"hands; its products are {products} (CASA codes)")
