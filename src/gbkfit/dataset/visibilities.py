import os.path
from collections.abc import Mapping, Sequence
from typing import Any

import astropy.units
import numpy as np

from gbkfit.utils import gridutils, parseutils, unitutils
from gbkfit.utils.parseutils import ConfigError
from .base import Dataset
from .data import Data


__all__ = [
    'DatasetVisibilities',
    'UVCoverage'
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


class UVCoverage:
    """
    Where interferometric visibilities are measured: the baselines of
    their rows, the frequencies of their channels, their phase centre, and
    the rest frequency of their line.

    Parameters
    ----------
    uvw : numpy.ndarray
        The baselines b = (u, v, w) of the rows, in m, in the frame of the
        phase centre p (u to the east, v to the north, w to it), of shape
        (nrows, 3): of the visibilities V(b) = sum of the flux at each
        direction s times exp(-2 pi i b . (s - p)) (b in wavelengths);
        those of a measurement set are -UVW (see
        DatasetVisibilities.from_ms).
    frequencies : Sequence of float
        The frequencies of the channels (Hz): at least two, uniform, in
        the rest frame given (not supported yet: the one channel of
        continuum visibilities).
    phase_centre : Sequence of float
        The RA and Dec of the phase centre (degrees).
    frame : str
        The rest frame of the frequencies: 'LSRK' or 'BARY' (for TOPO
        ones, regrid them, e.g. by CASA mstransform with outframe='LSRK';
        not supported yet: TOPO frequencies with the Doppler shift of the
        time of each row).
    rest : str or Quantity
        The rest frequency of the line, of the radio velocities of the
        channels (see gridutils.make_rest; e.g. '1420.405752 MHz').

    Raises
    ------
    ConfigError
        If the baselines are not finite, of shape (nrows, 3), the
        frequencies are not at least two, positive and uniform, the phase
        centre is not an RA and a Dec, the frame is not one of those, or
        the rest is not a frequency.
    """

    # The names of its arrays in files (see arrays)
    _ARRAYS = ('uvw', 'frequencies', 'phase_centre', 'frame', 'rest')

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> 'UVCoverage':
        """
        Make the coverage of its arrays (see arrays; e.g. those of a file
        of visibilities).

        Raises
        ------
        ConfigError
            If an array is missing, or they are invalid (see the class).
        """
        missing = set(cls._ARRAYS) - set(arrays)
        if missing:
            raise ConfigError(
                f"the coverage of visibilities needs the arrays "
                f"{sorted(missing)}")
        return cls(
            arrays['uvw'], arrays['frequencies'], arrays['phase_centre'],
            str(arrays['frame']), float(arrays['rest']) * astropy.units.Hz)

    @classmethod
    def from_file(cls, file: str) -> 'UVCoverage':
        """
        Read the coverage from a numpy file (npz) of its arrays (see
        arrays; e.g. a file of visibilities, see
        DatasetVisibilities.from_file).

        Raises
        ------
        ConfigError
            If an array is missing, or they are invalid (see the class).
        """
        with np.load(file) as arrays:
            return cls.from_arrays(dict(arrays))

    def dump(
            self, filename: str, dump_path: bool = True,
            overwrite: bool = False
    ) -> str:
        """
        Write its arrays to a numpy file (npz), and return its filename
        (the path, or its name only without dump_path).

        Raises
        ------
        FileExistsError
            If the file exists and overwrite is false.
        """
        _savez(filename, self.arrays(), overwrite)
        return filename if dump_path else os.path.basename(filename)

    def __init__(
            self,
            uvw: np.ndarray,
            frequencies: Sequence[float],
            phase_centre: Sequence[float],
            frame: str,
            rest: str | astropy.units.Quantity
    ):
        uvw = np.array(uvw, float)
        frequencies = np.array(frequencies, float)
        if uvw.ndim != 2 or uvw.shape[1] != 3 or not np.isfinite(uvw).all():
            raise ConfigError(
                f"the baselines must be finite, one (u, v, w) for each row; "
                f"their shape is {uvw.shape}")
        steps = np.diff(frequencies)
        if frequencies.ndim != 1 or frequencies.size < 2 \
                or not np.all(frequencies > 0) or steps[0] == 0 \
                or not np.allclose(steps, steps[0], rtol=1e-6, atol=0):
            raise ConfigError(
                "the frequencies must be at least two, positive and uniform "
                "(e.g. regridded by CASA mstransform)")
        if np.shape(phase_centre) != (2,):
            raise ConfigError(
                f"the phase centre must be an RA and a Dec; it is "
                f"{phase_centre}")
        if frame not in _FRAMES:
            raise ConfigError(
                f"the frequencies must be in the rest frame "
                f"{' or '.join(_FRAMES)}; they are in {frame} (regrid them, "
                f"e.g. by CASA mstransform with outframe='LSRK')")
        rest = gridutils.make_rest(rest)
        if rest is None or rest.unit != astropy.units.Hz:
            raise ConfigError(
                f"the rest of visibilities must be a frequency; it is {rest}")
        self._uvw = uvw
        self._frequencies = frequencies
        self._phase_centre = tuple(float(x) for x in phase_centre)
        self._frame = frame
        self._rest = rest

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UVCoverage) and all(
            np.array_equal(a, b) for a, b in zip(
                self.arrays().values(), other.arrays().values()))

    def arrays(self) -> dict[str, np.ndarray]:
        """
        Return its arrays, by name: uvw, frequencies, phase_centre, frame
        and rest (Hz).
        """
        return dict(
            uvw=self._uvw, frequencies=self._frequencies,
            phase_centre=np.array(self._phase_centre),
            frame=np.array(self._frame),
            rest=np.array(self._rest.to_value(astropy.units.Hz)))

    def nrows(self) -> int:
        """Return the number of rows."""
        return len(self._uvw)

    def nchannels(self) -> int:
        """Return the number of channels."""
        return len(self._frequencies)

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


class DatasetVisibilities(Dataset):
    """
    Interferometric visibilities of Stokes I: their real and imaginary
    parts (the items re and im, of shape (nchannels, nrows), with the error
    of each part), and where they are measured (see UVCoverage).

    Its configuration is a file of the arrays of the dataset (option file;
    see from_file and dump), or a measurement set and the options of
    from_ms (option ms).

    Parameters
    ----------
    re, im : Data
        The real and imaginary parts, of shape (nchannels, nrows), with
        the error of each part (e.g. 1 / sqrt(weight) of a measurement
        set).
    coverage : UVCoverage
        Where they are measured.

    Raises
    ------
    ConfigError
        If the items have no errors, or are not of the channels and rows
        of the coverage.
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
        visibilities left out) and unit (e.g. 'Jy'); and those of its
        coverage (see UVCoverage.arrays).

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
        missing = {'data', 'error'} - set(arrays)
        if missing:
            raise ConfigError(
                f"the file of visibilities {file} has no arrays "
                f"{sorted(missing)}")
        data, mask = arrays['data'], arrays.get('mask')
        unit = str(arrays['unit']) if 'unit' in arrays else None
        return cls(
            Data(data.real, mask, arrays['error'], unit),
            Data(data.imag, mask, arrays['error'], unit),
            UVCoverage.from_arrays(arrays))

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
        given, else of WEIGHT. The visibilities of a measurement set are
        sums of exp(+2 pi i UVW . (s - p)), so their baselines b (see
        UVCoverage) are -UVW (checked against those that CASA predicts;
        see tests/data/visibilities).

        Not supported yet: errors estimated from the scatter of the
        visibilities (for weights of the wrong scale, fit a noise scale);
        several fields or spectral windows in one dataset (one dataset
        each).

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
            The rest frequency (see UVCoverage); by default, that of the
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
            hands, or no rest frequency (see UVCoverage).
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
            uvw = -rows.getcol('UVW')
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
            UVCoverage(uvw, frequencies, phase_centre, frame, rest))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        filename = f'{prefix}visibilities.npz'
        re, im = self['re'], self['im']
        arrays = dict(
            data=re.data() + 1j * im.data(), error=re.error(), mask=re.mask(),
            **self._coverage.arrays())
        if re.unit() is not None:
            arrays['unit'] = unitutils.format_unit(re.unit())
        _savez(filename, arrays, overwrite)
        return dict(
            type=self.type(),
            file=filename if dump_path else os.path.basename(filename))

    def __init__(self, re: Data, im: Data, coverage: UVCoverage):
        super().__init__(dict(re=re, im=im))
        if re.error() is None or im.error() is None:
            raise ConfigError(
                "visibilities need the errors of their parts (e.g. 1 / "
                "sqrt(weight) of a measurement set)")
        shape = (coverage.nchannels(), coverage.nrows())
        if re.shape() != shape:
            raise ConfigError(
                f"the visibilities must be of the {shape[0]} channels and the "
                f"{shape[1]} rows of their coverage; their shape is "
                f"{re.shape()}")
        self._coverage = coverage

    def coverage(self) -> UVCoverage:
        """Return where the visibilities are measured."""
        return self._coverage


def _savez(
        filename: str, arrays: Mapping[str, np.ndarray], overwrite: bool
) -> None:
    """Write arrays to a numpy file (npz), unless it exists (but overwrite)."""
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(f"{filename} exists")
    with open(filename, 'wb') as file:
        np.savez(file, **arrays)


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
