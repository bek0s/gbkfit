from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from gbkfit.dataset import Dataset
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument, instrument_parser
from gbkfit.model import Band, BandEmission, LineCatalogue, band_parser
from gbkfit.model.components.roles import load_band
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


def _check_dtype(dtype: npt.DTypeLike) -> np.dtype:
    """
    Return the floating type of a model, or raise ConfigError if the
    native modules have no kernels for it.
    """
    supported = Driver.dtypes()
    try:
        result = np.dtype(dtype)
    except TypeError:
        result = None
    if result not in supported:
        raise ConfigError(
            f"dtype must be one of {[dt.name for dt in supported]}; it is "
            f"{dtype!r}")
    return result


def _driver_by_name(
        name: object,
        drivers: Mapping[str | None, Driver]
) -> Driver | None:
    """
    Return the driver of the given name (None: the only driver, or None
    if there are no drivers).

    Raises
    ------
    ConfigError
        If the name is not one of the drivers, or is left out while there
        are several drivers.
    """
    names = [n for n in drivers if n is not None]
    if name is None:
        if len(drivers) > 1:
            raise ConfigError(
                f"the observation must name its driver; the drivers are "
                f"{names}")
        return next(iter(drivers.values()), None)
    if not isinstance(name, str):
        raise ConfigError(
            f"the driver of an observation is the name of a driver that "
            f"the configuration declares at its top level (section "
            f"'driver' for one driver, 'drivers' for several by name); it "
            f"is {name!r}")
    if name not in names:
        raise ConfigError(
            f"unknown driver {name!r}; the named drivers are {names}")
    return drivers[name]


class Observation(parseutils.Serializable):
    """
    Galaxies seen as an observable, along a line of sight, through a
    foreground (e.g. a gravitational lens) and an instrument, and
    optionally its data, which the model of the galaxies is compared with
    under a likelihood.

    In its configuration, the data have the form of the observable (no
    type), and give it the options of options_from_data (see
    Observable.from_data).

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
    data : Dataset, optional
        The data, of the dataset class of the observable.
    likelihood : Likelihood, optional
        How the model is compared with the data; Gaussian by default when
        there are data, and none without data.
    scale : Sequence of int, optional
        How many times the model is oversampled along each axis of the
        data (an accuracy setting); 1 by default.
    dtype : str, optional
        The floating type of the model: 'float32' or 'float64'.
    name : str, optional
        Its name, which prefixes its extra outputs instead of its position
        (see ObservationGroup).

    Raises
    ------
    ConfigError
        If a name is invalid, there are no galaxies or they repeat, a band
        is given to an observable that is not an image, the scale is not
        one integer of at least 1
        for each axis, there is a likelihood without data, the data are
        not those of the observable, the observable cannot be seen
        through the instrument (see Observable.check_instrument), or the
        dtype is not one of the native modules (see Driver.dtypes).
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
            info['driver'] = _driver_by_name(info.get('driver'), drivers or {})
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
        # The form of the data is that of the observable: no type
        data = {} if self._data is None else dict(
            data={k: v for k, v in self._data.dump(**kwargs).items()
                  if k != 'type'},
            likelihood=likelihood_parser.dump(self._likelihood))
        return name | galaxies | components | lines | band | weights \
            | data | dict(
            line_of_sight=line_of_sight_parser.dump(self._line_of_sight),
            foreground=foreground_parser.dump(self._foreground, **kwargs),
            instrument=instrument_parser.dump(self._instrument, **kwargs),
            observable=observable_parser.dump(
                self._observable, data=self._data, **kwargs),
            scale=self._scale,
            dtype=self._dtype.name)

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
            data: Dataset | None = None,
            likelihood: Likelihood | None = None,
            scale: Sequence[int] | None = None,
            dtype: str = 'float32',
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
        observable.check_instrument(instrument)
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
        self._data = data
        self._likelihood = likelihood
        self._scale = scale
        self._dtype = _check_dtype(dtype)
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

    def driver(self) -> Driver:
        """Return the driver the model is evaluated on."""
        return self._driver

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

    def scale(self) -> tuple[int, ...]:
        """Return the oversampling of the model along each axis."""
        return self._scale

    def dtype(self) -> np.dtype:
        """Return the floating type of the model."""
        return self._dtype

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
            self._scale, self._dtype, self._weights)


observation_parser = parseutils.BasicParser(Observation)
