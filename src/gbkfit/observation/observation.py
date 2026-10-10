from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.dataset import Dataset
from gbkfit.driver import Driver, driver_parser
from gbkfit.instrument import Instrument, instrument_parser
from gbkfit.model.base import Selection
from gbkfit.utils import parseutils
from .foreground import Foreground, foreground_parser
from .likelihood import Likelihood, LikelihoodGaussian, likelihood_parser
from .observables import Observable, observable_parser


__all__ = [
    'Observation',
    'observation_parser'
]


class Observation(parseutils.Serializable):
    """
    A gmodel seen through a foreground (e.g. a gravitational lens) and an
    instrument as an observable, evaluated on a driver, and optionally its
    data, compared with the model under a likelihood. It names its gmodel
    (required only when there are several) and, optionally, the components
    and the emission lines of the gmodel it sees (e.g. the tracer and the
    line of its data), and can have a name, which then prefixes its extra
    outputs instead of its position (see ObservationGroup).
    """

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'Observation':
        # The data are read in the form of the observable (its dataset
        # class), and give the observable its grid
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
        opts = parseutils.parse_options_for_callable(info, cls.__init__)
        return cls(**opts)

    def dump(self, **dump_kwargs) -> dict[str, Any]:
        name = dict(name=self._name) if self._name is not None else {}
        gmodel = dict(gmodel=self._gmodel) if self._gmodel is not None else {}
        selection = self._selection
        components = {} if selection.components is None else dict(
            components=list(selection.components))
        lines = {} if selection.lines is None else dict(
            lines=list(selection.lines))
        # The form of the data is that of the observable: no type
        data = {} if self._data is None else dict(
            data={k: v for k, v in self._data.dump(**dump_kwargs).items()
                  if k != 'type'},
            likelihood=likelihood_parser.dump(self._likelihood))
        return name | gmodel | components | lines | data | dict(
            driver=driver_parser.dump(self._driver),
            foreground=foreground_parser.dump(
                self._foreground, **dump_kwargs),
            instrument=instrument_parser.dump(
                self._instrument, **dump_kwargs),
            observable=observable_parser.dump(
                self._observable, data=self._data, **dump_kwargs),
            scale=self._scale,
            dtype=self._dtype.name)

    def __init__(
            self,
            driver: Driver,
            observable: Observable,
            foreground: Foreground | None = None,
            instrument: Instrument | None = None,
            gmodel: str | None = None,
            components: Sequence[str] | None = None,
            lines: Sequence[str] | None = None,
            data: Dataset | None = None,
            likelihood: Likelihood | None = None,
            scale: Sequence[int] | None = None,
            dtype: str = 'float32',
            name: str | None = None
    ):
        """
        components and lines are the names of the components of the gmodel
        and of their emission lines that the observation sees (all if None;
        see Selection). scale is the oversampling of the model along each axis of
        the data (an accuracy setting; 1 by default). The likelihood is
        Gaussian by default when there are data, and there is none without
        data.
        """
        parseutils.check_name(name)
        if gmodel is not None:
            parseutils.check_name(gmodel)
        if components is not None:
            components = tuple(components)
            for component in components:
                parseutils.check_name(component)
        if lines is not None:
            lines = tuple(lines)
            for line in lines:
                parseutils.check_name(line)
        ndim = len(observable.size())
        scale = tuple(scale) if scale is not None else (1,) * ndim
        if len(scale) != ndim or any(s < 1 for s in scale):
            raise RuntimeError(
                f"scale must have {ndim} values, each at least 1; it is "
                f"{scale}")
        self._driver = driver
        self._observable = observable
        self._foreground = foreground if foreground is not None \
            else Foreground()
        self._instrument = instrument if instrument is not None \
            else Instrument()
        if data is None and likelihood is not None:
            raise RuntimeError("a likelihood needs data")
        if data is not None:
            observable.require_matching(data)
        if data is not None and likelihood is None:
            likelihood = LikelihoodGaussian()
        self._gmodel = gmodel
        self._selection = Selection(components, lines)
        self._data = data
        self._likelihood = likelihood
        self._scale = scale
        self._dtype = np.dtype(dtype)
        self._name = name

    def name(self) -> str | None:
        return self._name

    def gmodel(self) -> str | None:
        """The name of the gmodel it observes, if given."""
        return self._gmodel

    def selection(self) -> Selection:
        """The components and lines of the gmodel it sees."""
        return self._selection

    def driver(self) -> Driver:
        return self._driver

    def observable(self) -> Observable:
        return self._observable

    def foreground(self) -> Foreground:
        return self._foreground

    def instrument(self) -> Instrument:
        return self._instrument

    def data(self) -> Dataset | None:
        return self._data

    def likelihood(self) -> Likelihood | None:
        return self._likelihood

    def scale(self) -> tuple[int, ...]:
        return self._scale

    def dtype(self) -> np.dtype:
        return self._dtype

    def plan(self, gmodel):
        """The evaluation of the gmodel as this observation."""
        self._observable.require_compatible(gmodel)
        return self._observable.plan(
            self._driver, gmodel, self._foreground, self._instrument,
            self._scale, self._dtype, self._selection)


observation_parser = parseutils.BasicParser(Observation)
