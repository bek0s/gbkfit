from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.dataset import Dataset
from gbkfit.driver import Driver, driver_parser
from gbkfit.instrument import Instrument, instrument_parser
from gbkfit.model.base import Model, Selection
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from .foreground import Foreground, foreground_parser
from .likelihood import Likelihood, LikelihoodGaussian, likelihood_parser
from .observables import Observable, ObservablePlan, observable_parser


__all__ = [
    'Observation',
    'observation_parser'
]


class Observation(parseutils.Serializable):
    """
    A model seen as an observable, through a foreground (e.g. a
    gravitational lens) and an instrument, and optionally its data, which
    the model is compared with under a likelihood.

    In its configuration, the data have the form of the observable (no
    type), and give it the options of options_from_data (see
    Observable.from_data).

    Parameters
    ----------
    observable : Observable
        What the data measure.
    driver : Driver, optional
        Where the model is evaluated; by default, the host.
    foreground : Foreground, optional
        What the light meets before the telescope; nothing by default.
    instrument : Instrument, optional
        The telescope and the instrument; a perfect one by default.
    model : str, optional
        The name of the model it observes; required only when there are
        several (see ObservationGroup).
    components, lines : Sequence of str, optional
        The names of the components of the model and of their emission
        lines that it sees (e.g. the tracer and the line of the data); all
        by default (see Selection).
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
        If a name is invalid, the scale is not one integer of at least 1
        for each axis, there is a likelihood without data, the data are
        not those of the observable, or the observable cannot be seen
        through the instrument (see Observable.check_instrument).
    """

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'Observation':
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
            driver_parser, info, 'driver')
        parseutils.load_option_and_update_info(
            foreground_parser, info, 'foreground')
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
        Dump the observation to its configuration.

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
        model = dict(model=self._model) if self._model is not None else {}
        selection = self._selection
        components = {} if selection.components is None else dict(
            components=list(selection.components))
        lines = {} if selection.lines is None else dict(
            lines=list(selection.lines))
        # The form of the data is that of the observable: no type
        data = {} if self._data is None else dict(
            data={k: v for k, v in self._data.dump(**kwargs).items()
                  if k != 'type'},
            likelihood=likelihood_parser.dump(self._likelihood))
        return name | model | components | lines | data | dict(
            driver=driver_parser.dump(self._driver),
            foreground=foreground_parser.dump(self._foreground, **kwargs),
            instrument=instrument_parser.dump(self._instrument, **kwargs),
            observable=observable_parser.dump(
                self._observable, data=self._data, **kwargs),
            scale=self._scale,
            dtype=self._dtype.name)

    def __init__(
            self,
            observable: Observable,
            driver: Driver | None = None,
            foreground: Foreground | None = None,
            instrument: Instrument | None = None,
            model: str | None = None,
            components: Sequence[str] | None = None,
            lines: Sequence[str] | None = None,
            data: Dataset | None = None,
            likelihood: Likelihood | None = None,
            scale: Sequence[int] | None = None,
            dtype: str = 'float32',
            name: str | None = None
    ):
        parseutils.check_name(name)
        if model is not None:
            parseutils.check_name(model)
        if components is not None:
            components = tuple(components)
            for component in components:
                parseutils.check_name(component)
        if lines is not None:
            lines = tuple(lines)
            for line in lines:
                parseutils.check_name(line)
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
        self._model = model
        self._selection = Selection(components, lines)
        self._data = data
        self._likelihood = likelihood
        self._scale = scale
        self._dtype = np.dtype(dtype)
        self._name = name

    def name(self) -> str | None:
        """Return its name, if any."""
        return self._name

    def model(self) -> str | None:
        """Return the name of the model it observes, if given."""
        return self._model

    def selection(self) -> Selection:
        """Return what of the model it sees."""
        return self._selection

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

    def plan(self, model: Model) -> ObservablePlan:
        """
        Plan the evaluation of a model as this observation.

        Parameters
        ----------
        model : Model
            The model.

        Returns
        -------
        ObservablePlan
            The plan.

        Raises
        ------
        ConfigError
            If the model cannot be observed as the observable.
        """
        self._observable.require_compatible(model)
        return self._observable.plan(
            self._driver, model, self._foreground, self._instrument,
            self._scale, self._dtype, self._selection)


observation_parser = parseutils.BasicParser(Observation)
