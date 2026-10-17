import copy
from collections.abc import Mapping, Sequence
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from gbkfit.dataset import Dataset
from gbkfit.driver import Driver, driver_by_name
from gbkfit.instrument import Instrument, PSFGaussBeam, instrument_parser
from gbkfit.model import Band, BandEmission, LineCatalogue, band_parser
from gbkfit.model.components._units import DataUnit
from gbkfit.model.components.roles import load_band
from gbkfit.params import ParamScalarDesc
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from ._scene import Scene, check_component_names
from .foreground import Foreground, foreground_parser
from .likelihood import Likelihood, LikelihoodGaussian, likelihood_parser
from .line_of_sight import LineOfSight, line_of_sight_parser
from .observables import Observable, ObservablePlan, observable_parser
from .weights import Weights, weights_parser


__all__ = [
    'Observation',
    'observation_parser'
]


def _check_dtype(dtype: npt.DTypeLike, name: str = 'dtype') -> np.dtype:
    """
    Return a floating type of a model (the option name), or raise
    ConfigError if the native modules have no kernels for it.
    """
    supported = Driver.dtypes()
    try:
        result = np.dtype(dtype)
    except TypeError:
        result = None
    if result not in supported:
        raise ConfigError(
            f"{name} must be one of {[dt.name for dt in supported]}; it is "
            f"{dtype!r}")
    return result


class Observation(parseutils.Serializable):
    """
    Galaxies seen as an observable, along a line of sight, through a
    foreground (e.g. a gravitational lens) and an instrument, and
    optionally its data, which the model of the galaxies is compared with
    under a likelihood.

    In its configuration, the data have the form of the observable (no
    type), and give it the options of options_from_data (see
    Observable.from_data).

    Its model is in the unit of its data (see data_unit): the light of
    the components whose emission has a unit is turned into it (see
    Component.plan), and that of the others is in it. Without data, the
    model is in the unit of the components it sees, which must then have
    one (or none).

    Parameters
    ----------
    observable : Observable
        What the data measure.
    line_of_sight : LineOfSight
        How it treats the line of sight: Flat() (the plane of the sky) or
        Grid() (a depth axis).
    driver : Driver, optional
        Where the model is evaluated; by default, the host. (In its
        configuration, the name of a driver; see load.)
    foreground : Foreground, optional
        What the light meets before the telescope; nothing by default.
    instrument : Instrument, optional
        The telescope and the instrument; a perfect one by default.
    galaxies : Sequence of str, optional
        The names of the galaxies it sees; required only when there are
        several (see ObservationGroup).
    components : Sequence of str, optional
        The names of the components whose light it sees ('galaxy.component'
        when it sees several galaxies), which emit what it sees: in a band
        for an image, lines for a spectral cube; all of those by default.
        The opacity of every component applies.
    lines : Sequence of str, optional
        The names of the emission lines it sees (e.g. the line of the
        data); all by default.
    band : Band, optional
        The band of an image: it sees the components that emit in every
        band or in it (see BandEmission), and dust of an extinction law
        dims them at its rest wavelength (see Galaxy).
    weights : Weights, optional
        The spatial weights of its data (e.g. less weight near the minor
        axis of a disk); none by default.
    calibration : bool, optional
        Whether the flux of its model is times a parameter,
        '{name}_calibration' (e.g. a flux scale between the instruments
        of a joint fit); its name is then required. False by default.
    data : Dataset, optional
        The data, of the dataset class of the observable.
    likelihood : Likelihood, optional
        How the model is compared with the data; Gaussian by default when
        there are data, and none without data.
    scale : Sequence of int, optional
        How many times the model is oversampled along each axis of the
        data (an accuracy setting); 1 by default.
    dtype : str, optional
        The floating type of the model: 'float32' (by default) or
        'float64'.
    fft_dtype : str, optional
        The floating type of the FFT convolutions of the model by the PSF
        and the LSF: 'float64' (by default) or 'float32' (faster, for a
        float32 model only). The rounding of an FFT in float32 adds about
        1e-7 to 1e-6 of the brightest values of the cube to every value,
        which spoils the faint ones (e.g. their moments, and the
        finite-difference derivatives of fits).
    name : str, optional
        Its name, which prefixes its extra outputs instead of its position
        (see ObservationGroup).

    Raises
    ------
    ConfigError
        If a name is invalid, there are no galaxies or they repeat, a band
        is given to an observable that is not an image, it has a
        calibration but no name, the scale is not one integer of at least
        1
        for each axis, there is a likelihood without data, the data are
        not those of the observable, its model can lack values but its
        likelihood has no residual for them (see LikelihoodGaussian), its
        likelihood has free amplitudes and the observable has no spectra
        or it has a calibration, its likelihood fits a noise but it has no
        name, or fits the noise of items that the observable does not
        have, or both the scale and the jitter of the noise of an item
        whose errors are all equal (see _check_noise), the observable
        cannot be seen
        through the instrument (see Observable.check_instrument), the
        dtype or the fft_dtype is not one of the native modules (see
        Driver.dtypes), or the fft_dtype is less precise than the dtype.
    """

    @classmethod
    def load(
            cls,
            info: dict[str, Any],
            drivers: Mapping[str | None, Driver] | None = None,
            catalogue: LineCatalogue | None = None
    ) -> 'Observation':
        """
        Load an observation from its configuration.

        Its option driver is not a driver but the name of one of the
        drivers of the configuration, which declares them at its top
        level. It can be left out when there is one driver, and the
        observation has the host when there are none. Its band is the name
        of a band of the catalogue, or a band (its name and wavelength).

        Parameters
        ----------
        info : dict
            The configuration.
        drivers : Mapping of str to Driver, optional
            The drivers of the configuration, by name; the one unnamed
            driver of a configuration has the name None.
        catalogue : LineCatalogue, optional
            The catalogue of the names of bands; that of GBKFIT by
            default.

        Returns
        -------
        Observation
            The observation.

        Raises
        ------
        ConfigError
            If the configuration has an error, or the driver is not one
            of the drivers.
        """
        with parseutils.config_path('driver'):
            info['driver'] = driver_by_name(
                info.get('driver'), drivers or {}, 'observation')
        if info.get('band') is not None:
            with parseutils.config_path('band'):
                info['band'] = load_band(
                    info['band'],
                    catalogue if catalogue is not None else LineCatalogue())
        # The data are read in the form of the observable (its dataset
        # class), and give the observable some of its options
        dataset = None
        if info.get('data') is not None:
            observable_type = dict(info.get('observable') or {}).get('type')
            observable_cls = observable_parser.registered_class(
                observable_type)
            with parseutils.config_path('data'):
                dataset = observable_cls.dataset_class.load(
                    dict(info['data']))
            info['data'] = dataset
        parseutils.load_option_and_update_info(
            likelihood_parser, info, 'likelihood')
        parseutils.load_option_and_update_info(
            line_of_sight_parser, info, 'line_of_sight', required=True)
        parseutils.load_option_and_update_info(
            foreground_parser, info, 'foreground')
        parseutils.load_option_and_update_info(
            weights_parser, info, 'weights')
        parseutils.load_option_and_update_info(
            instrument_parser, info, 'instrument')
        parseutils.load_option_and_update_info(
            observable_parser, info, 'observable', dataset=dataset)
        return cls(**parseutils.parse_options_for_callable(info, cls.__init__))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Dump the observation to its configuration, without its driver,
        which configurations declare at their top level (see load).

        Parameters
        ----------
        prefix : str, optional
            The start of the names of the files of its data and of the
            data of its parts (e.g. images; see Dataset.dump).
        dump_path : bool, optional
            Whether the configuration has the paths of the files, or only
            their names.
        overwrite : bool, optional
            Whether to overwrite existing files.

        Returns
        -------
        dict
            The options.
        """
        kwargs = dict(prefix=prefix, dump_path=dump_path, overwrite=overwrite)
        name = dict(name=self._name) if self._name is not None else {}
        galaxies = {} if self._galaxies is None else dict(
            galaxies=list(self._galaxies))
        components = {} if self._components is None else dict(
            components=list(self._components))
        lines = {} if self._lines is None else dict(lines=list(self._lines))
        band = {} if self._band is None else dict(
            band=band_parser.dump(self._band))
        weights = {} if self._weights is None else dict(
            weights=self._weights.dump())
        calibration = dict(calibration=True) if self._calibration else {}
        # The form of the data is that of the observable: no type
        data = {} if self._data is None else dict(
            data={k: v for k, v in self._data.dump(**kwargs).items()
                  if k != 'type'},
            likelihood=likelihood_parser.dump(self._likelihood))
        return name | galaxies | components | lines | band | weights \
            | calibration | data | dict(
            line_of_sight=line_of_sight_parser.dump(self._line_of_sight),
            foreground=foreground_parser.dump(self._foreground, **kwargs),
            instrument=instrument_parser.dump(self._instrument, **kwargs),
            observable=observable_parser.dump(
                self._observable, data=self._data, **kwargs),
            scale=self._scale,
            dtype=self._dtype.name,
            fft_dtype=self._fft_dtype.name)

    def __init__(
            self,
            observable: Observable,
            line_of_sight: LineOfSight,
            driver: Driver | None = None,
            foreground: Foreground | None = None,
            instrument: Instrument | None = None,
            galaxies: Sequence[str] | None = None,
            components: Sequence[str] | None = None,
            lines: Sequence[str] | None = None,
            band: Band | None = None,
            weights: Weights | None = None,
            calibration: bool = False,
            data: Dataset | None = None,
            likelihood: Likelihood | None = None,
            scale: Sequence[int] | None = None,
            dtype: str = 'float32',
            fft_dtype: str = 'float64',
            name: str | None = None
    ):
        parseutils.check_name(name)
        # (the second argument was the driver before the line of sight)
        if not isinstance(line_of_sight, LineOfSight):
            raise TypeError(
                f"line_of_sight must be a LineOfSight (e.g. Flat() or "
                f"Grid()); it is {line_of_sight!r}")
        if galaxies is not None:
            galaxies = tuple(galaxies)
            if not galaxies:
                raise ConfigError("an observation sees at least one galaxy")
            if len(set(galaxies)) != len(galaxies):
                raise ConfigError(f"the galaxies repeat: {list(galaxies)}")
            for galaxy in galaxies:
                parseutils.check_name(galaxy)
        if components is not None:
            components = tuple(components)
            check_component_names(components)
        if lines is not None:
            lines = tuple(lines)
            for line in lines:
                parseutils.check_name(line)
        if calibration and name is None:
            raise ConfigError(
                "an observation with a calibration needs a name, which "
                "names its parameter ('{name}_calibration')")
        if band is not None and observable.emission is not BandEmission:
            raise ConfigError(
                "only images have a band; the lines of spectral cubes are "
                "their option lines")
        if driver is None:
            # (imported here: the host driver needs its native module)
            from gbkfit.driver.drivers.host import DriverHost
            driver = DriverHost()
        if foreground is None:
            foreground = Foreground()
        if name is None and foreground.pdescs():
            raise ConfigError(
                "an observation with a lens of parameters needs a name, "
                "which names them (e.g. '{name}_lens_theta_e')")
        foreground = foreground.with_prefix(
            '' if name is None else f'{name}_')
        if instrument is None:
            instrument = Instrument()
        ndim = len(observable.size())
        scale = tuple(scale) if scale is not None else (1,) * ndim
        if len(scale) != ndim or any(s < 1 for s in scale):
            raise ConfigError(
                f"scale must have {ndim} values, each at least 1; it is "
                f"{scale}")
        if data is None and likelihood is not None:
            raise ConfigError("a likelihood needs data")
        if data is not None:
            observable.require_matching(data)
            if likelihood is None:
                likelihood = LikelihoodGaussian()
            if likelihood.amplitudes() is not None:
                _check_amplitudes(observable, calibration)
            _check_noise(likelihood, observable, data)
            if name is None and likelihood.pdescs():
                raise ConfigError(
                    "an observation with a fitted noise needs a name, which "
                    "names its parameters (e.g. '{name}_noise_scale_"
                    "moment1')")
            likelihood = likelihood.with_prefix(
                '' if name is None else f'{name}_')
            if observable.model_can_lack_values() \
                    and likelihood.missing_residual() is None:
                raise ConfigError(
                    "the model of the observable can lack values where the "
                    "data have some (e.g. the velocity of a spectrum below "
                    "its mask_cutoff): its likelihood needs the residual of "
                    "such a value, the option missing_residual (e.g. 3; see "
                    "LikelihoodGaussian)")
        observable.check_instrument(instrument)
        if data is not None:
            _check_beams(data, instrument)
        self._driver = driver
        self._observable = observable
        self._foreground = foreground
        self._instrument = instrument
        self._line_of_sight = line_of_sight
        self._galaxies = galaxies
        self._components = components
        self._lines = lines
        self._band = band
        self._weights = weights
        self._calibration = calibration
        self._data = data
        self._likelihood = likelihood
        self._scale = scale
        self._dtype = _check_dtype(dtype)
        self._fft_dtype = _check_dtype(fft_dtype, 'fft_dtype')
        if self._fft_dtype.itemsize < self._dtype.itemsize:
            raise ConfigError(
                f"the fft_dtype ({self._fft_dtype.name}) is less precise "
                f"than the dtype ({self._dtype.name})")
        self._name = name

    def name(self) -> str | None:
        """Return its name, if any."""
        return self._name

    def line_of_sight(self) -> LineOfSight:
        """Return how it treats the line of sight."""
        return self._line_of_sight

    def galaxies(self) -> tuple[str, ...] | None:
        """Return the names of the galaxies it sees, if given."""
        return self._galaxies

    def components(self) -> tuple[str, ...] | None:
        """Return the names of the components it sees, if given."""
        return self._components

    def lines(self) -> tuple[str, ...] | None:
        """Return the names of the emission lines it sees, if given."""
        return self._lines

    def band(self) -> Band | None:
        """Return its band (of an image), if given."""
        return self._band

    def weights(self) -> Weights | None:
        """Return the spatial weights of its data, if any."""
        return self._weights

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """
        Return its parameters: that of its calibration, if any (see the
        class), those of its lens, '{name}_lens_{parameter}' (see
        Foreground.pdescs), and those of the fitted noise of its
        likelihood, e.g. '{name}_noise_scale_{item}' (see
        LikelihoodGaussian).
        """
        name = self._calibration_param()
        calibration = {} if name is None else {name: ParamScalarDesc(
            desc=f"the flux scale of observation {self._name}")}
        likelihood = {} if self._likelihood is None \
            else self._likelihood.pdescs()
        return calibration | self._foreground.pdescs() | likelihood

    def _calibration_param(self) -> str | None:
        """Return the name of the parameter of its calibration, if any."""
        return f'{self._name}_calibration' if self._calibration else None

    def driver(self) -> Driver:
        """Return the driver the model is evaluated on."""
        return self._driver

    def with_driver(self, driver: Driver) -> Self:
        """
        Return a copy of it whose model is evaluated on another driver;
        the copy shares the rest, its data too (e.g. copies of an
        objective on several devices; see Objective.with_driver).
        """
        observation = copy.copy(self)
        observation._driver = driver
        return observation

    def observable(self) -> Observable:
        """Return the observable."""
        return self._observable

    def foreground(self) -> Foreground:
        """Return the foreground."""
        return self._foreground

    def instrument(self) -> Instrument:
        """Return the instrument."""
        return self._instrument

    def data(self) -> Dataset | None:
        """Return the data, if any."""
        return self._data

    def likelihood(self) -> Likelihood | None:
        """Return the likelihood, if there are data."""
        return self._likelihood

    def data_unit(self) -> DataUnit | None:
        """
        Return the unit of the light of its data, which its model takes
        (see Observable.data_unit); None without data (its model is then
        in the unit of the components it sees), or without data of light.
        """
        return self._observable.data_unit(self._data)

    def scale(self) -> tuple[int, ...]:
        """Return the oversampling of the model along each axis."""
        return self._scale

    def dtype(self) -> np.dtype:
        """Return the floating type of the model."""
        return self._dtype

    def fft_dtype(self) -> np.dtype:
        """Return the floating type of the FFT convolutions of the model."""
        return self._fft_dtype

    def plan(self, scene: Scene) -> ObservablePlan:
        """
        Plan the evaluation of what it sees (see ObservationGroup).

        Parameters
        ----------
        scene : Scene
            What it sees: galaxies along its line of sight.

        Returns
        -------
        ObservablePlan
            The plan.
        """
        return self._observable.plan(
            self._driver, scene, self._foreground, self._instrument,
            self._scale, self._dtype, self._fft_dtype, self._weights,
            self._calibration_param())


def _check_amplitudes(observable: Observable, calibration: bool) -> None:
    """
    Raise ConfigError unless the free amplitudes of a likelihood (see
    LikelihoodGaussian) can scale the model of an observable, which must
    have spectra, and no calibration (which the amplitudes take).
    """
    if observable.spectra_item() is None:
        desc = parseutils.make_typed_desc(observable.__class__, 'observable')
        raise ConfigError(
            f"free amplitudes scale spectra (e.g. those of pixel_spectra, "
            f"slit_spectra or region_spectra); {desc} has none")
    if calibration:
        raise ConfigError(
            "free amplitudes take the flux scale of the model: an "
            "observation with them has no calibration")


def _check_noise(
        likelihood: Likelihood,
        observable: Observable,
        data: Dataset
) -> None:
    """
    Raise ConfigError unless the data items whose noise a likelihood fits
    (see LikelihoodGaussian) are items of an observable, and those whose
    scale and jitter are both fitted have errors that differ between their
    values (with equal errors, or none, the two are one).
    """
    fitted = likelihood.fitted_noise()
    if unknown := sorted(fitted.keys() - set(observable.keys())):
        raise ConfigError(
            f"the likelihood fits the noise of data items that the "
            f"observable does not have: {unknown}; its items are "
            f"{list(observable.keys())}")
    for key, noise in fitted.items():
        if not (noise.scale and noise.jitter):
            continue
        error = data[key].error()
        used = None if error is None else error[data[key].mask() != 0]
        if used is None or used.size == 0 or np.all(used == used[0]):
            raise ConfigError(
                f"the scale and the jitter of the noise of '{key}' cannot "
                f"both be fitted: its errors are all equal, or it has none, "
                f"so only (scale error)^2 + jitter^2 counts; fit one of them")


def _check_beams(data: Dataset, instrument: Instrument) -> None:
    """
    Warn (see parseutils.warn) if an item of the data has one beam, and
    the PSF of the instrument is another beam: the light of data per beam
    is per their beam, but the PSF smooths the model otherwise.
    """
    psf = instrument.psf()
    if not isinstance(psf, PSFGaussBeam):
        return
    for key in data.keys():
        beam = data[key].beam()
        if beam is None or beam.spectral is not None:
            continue
        # (the position angle of a circular beam is any)
        same = np.allclose(
            (beam.bmaj, beam.bmin), (psf.bmaj(), psf.bmin()), rtol=1e-3,
            atol=0) and (np.isclose(beam.bmaj, beam.bmin, rtol=1e-3) or abs(
                (beam.bpa - psf.bpa() + 90) % 180 - 90) < 0.1)
        if not same:
            parseutils.warn(
                f"the data ({key}) have the beam {beam.bmaj:g} x "
                f"{beam.bmin:g} arcsec at {beam.bpa:g} deg, but the PSF of "
                f"the instrument is the beam {psf.bmaj():g} x {psf.bmin():g} "
                f"arcsec at {psf.bpa():g} deg: the model is smoothed by the "
                f"PSF, and its light per beam is per beam of the data")


observation_parser = parseutils.BasicParser(Observation)
