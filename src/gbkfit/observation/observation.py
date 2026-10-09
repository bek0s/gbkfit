from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.driver import Driver, driver_parser
from gbkfit.utils import parseutils
from .instrument import Instrument, instrument_parser
from .observables import Observable, observable_parser


__all__ = [
    'Observation',
    'observation_parser'
]


class Observation(parseutils.BasicSerializable):
    """
    A gmodel seen through an instrument as an observable, evaluated on a
    driver. It names its gmodel (required only when there are several),
    and can have a name, which then prefixes its extra outputs instead of
    its position (see ObservationGroup).
    """

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'Observation':
        dataset = kwargs.get('dataset')
        desc = parseutils.make_basic_desc(cls, 'observation')
        parseutils.load_option_and_update_info(
            driver_parser, info, 'driver')
        parseutils.load_option_and_update_info(
            instrument_parser, info, 'instrument')
        parseutils.load_option_and_update_info(
            observable_parser, info, 'observable', dataset=dataset)
        if dataset is not None and 'dtype' not in info:
            info['dtype'] = np.dtype(dataset.dtype()).name
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self) -> dict[str, Any]:
        name = dict(name=self._name) if self._name is not None else {}
        gmodel = dict(gmodel=self._gmodel) if self._gmodel is not None else {}
        return name | gmodel | dict(
            driver=driver_parser.dump(self._driver),
            instrument=instrument_parser.dump(self._instrument),
            observable=observable_parser.dump(self._observable),
            scale=self._scale,
            dtype=self._dtype.name)

    def __init__(
            self,
            driver: Driver,
            observable: Observable,
            instrument: Instrument | None = None,
            gmodel: str | None = None,
            scale: Sequence[int] | None = None,
            dtype: str = 'float32',
            name: str | None = None
    ):
        """
        scale is the oversampling of the model along each axis of the data
        (an accuracy setting; 1 by default).
        """
        parseutils.check_name(name)
        if gmodel is not None:
            parseutils.check_name(gmodel)
        ndim = len(observable.size())
        scale = tuple(scale) if scale is not None else (1,) * ndim
        if len(scale) != ndim or any(s < 1 for s in scale):
            raise RuntimeError(
                f"scale must have {ndim} values, each at least 1; it is "
                f"{scale}")
        self._driver = driver
        self._observable = observable
        self._instrument = instrument if instrument is not None \
            else Instrument()
        self._gmodel = gmodel
        self._scale = scale
        self._dtype = np.dtype(dtype)
        self._name = name

    def name(self) -> str | None:
        return self._name

    def gmodel(self) -> str | None:
        """The name of the gmodel it observes, if given."""
        return self._gmodel

    def driver(self) -> Driver:
        return self._driver

    def observable(self) -> Observable:
        return self._observable

    def instrument(self) -> Instrument:
        return self._instrument

    def scale(self) -> tuple[int, ...]:
        return self._scale

    def dtype(self) -> np.dtype:
        return self._dtype

    def plan(self, gmodel):
        """The evaluation of the gmodel as this observation."""
        self._observable.require_compatible(gmodel)
        return self._observable.plan(
            self._driver, gmodel, self._instrument, self._scale, self._dtype)


observation_parser = parseutils.BasicParser(Observation)
