"""
Mock data of a group of observations (see draw_mocks), and their
options: the section 'mock' of configurations, which the task eval draws
(see MockOptions).
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
from annotated_types import Ge

from gbkfit.config import Serializable, check_options, config_path
from gbkfit.dataset import Data, Dataset
from gbkfit.model import GalaxyState
from gbkfit.params import ParamValues
from .noise import INSTRUMENT, Noise, noise_parser
from .observables import HostModelData, ObservablePlan
from .observation import Observation
from .sky import ObservationSky

if TYPE_CHECKING:
    from .group_plan import GroupPlan
    from .group import ObservationGroup


__all__ = [
    'MockOptions',
    'draw_mocks'
]


class MockOptions(Serializable):
    """
    The options of mock data of the observations of a group (see
    draw_mocks). Fits of the mock data compare them with the models under
    the likelihoods of the setup (see Likelihoods), which a setup of mock
    data can give for them.

    Parameters
    ----------
    noise : Noise or Mapping of str to Noise
        The noise of the data of every observation, or of each, by
        observation name (see ObservationGroup.per_observation, and the
        examples).
    nmocks : int, optional
        The number of mock datasets of each observation.
    nsamples : int, optional
        The number of draws of the errors of measured data (e.g.
        moments; see draw_mocks).
    seed : int, optional
        The seed of the random numbers; the same seed gives the same mock
        data. By default, the mock data differ from run to run.

    Raises
    ------
    ValueError
        If nmocks is below 1, or nsamples below 2.

    Examples
    --------
    One noise for every observation:

    .. code-block:: yaml

        mock:
          noise:
            type: gauss
            rms: 0.01
          nmocks: 10

    A noise for each observation, by name:

    .. code-block:: yaml

        mock:
          noise:
            halpha:
              type: gauss
              rms: 0.01
            hi:
              type: gauss
              rms: 0.002
    """

    def __init__(
            self,
            noise: Noise | Mapping[str, Noise],
            nmocks: Annotated[int, Ge(1)] = 1,
            nsamples: Annotated[int, Ge(2)] = 100,
            seed: int | None = None
    ):
        check_options(MockOptions.__init__, locals())
        self._noise = dict(noise) if isinstance(noise, Mapping) else noise
        self._nmocks = nmocks
        self._nsamples = nsamples
        self._seed = seed

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Dump the options to their configuration.

        Parameters
        ----------
        prefix : str, optional
            The start of the names of the files of the data of the noises
            (e.g. an error map; see Noise.dump), followed by the name of
            the observation and '_' for those given by name.
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
        if isinstance(self._noise, dict):
            noise = {
                name: noise_parser.dump(
                    item, prefix=f'{prefix}{name}_', dump_path=dump_path,
                    overwrite=overwrite)
                for name, item in self._noise.items()}
        else:
            noise = noise_parser.dump(
                self._noise, prefix=prefix, dump_path=dump_path,
                overwrite=overwrite)
        info: dict[str, Any] = dict(
            noise=noise, nmocks=self._nmocks, nsamples=self._nsamples)
        if self._seed is not None:
            info['seed'] = self._seed
        return info

    def nmocks(self) -> int:
        """Return the number of mock datasets of each observation."""
        return self._nmocks

    def nsamples(self) -> int:
        """Return the number of draws of the errors of measured data."""
        return self._nsamples

    def seed(self) -> int | None:
        """Return the seed of the random numbers, if any."""
        return self._seed

    def noises(self, group: 'ObservationGroup') -> list[Noise]:
        """
        Return the noise of each observation of a group.

        Raises
        ------
        ValueError
            If the noises are given by name, and do not name each
            observation (see ObservationGroup.per_observation).
        """
        noises = group.per_observation(self._noise, 'noise')
        return [noises[i] for i in range(group.nobservations())]

    def check(self, group: 'ObservationGroup') -> None:
        """
        Raise ValueError unless the options are of the observations of a
        group: a noise for all of them, or for each (see noises).
        """
        self.noises(group)


def draw_mocks(
        plan: 'GroupPlan', params: ParamValues, options: MockOptions
) -> list[list[Dataset]]:
    """
    Draw mock data of the observations of a group plan, by the options:
    their model data measured from what their instruments record of the
    model plus noise (see ObservablePlan.mock), with their errors.

    The data that are what the instrument records plus the noise (e.g.
    spectra) have its dispersion as their errors, and declare its
    correlation; those measured from them (e.g. moments) the scatter
    of their values over nsamples other draws (the standard deviation,
    of the values that are finite in at least two draws; the others
    are missing). The mock data have the unit and the beam of the
    model of their observation (see Observation.model_unit): those of
    its data, if it has data, or else those its options give it (e.g.
    the unit of ObservationSky). Those of the observations
    that a recording of orbits models are of the orbits as they are, as
    without data (see OrbitMomentsPlan): their model, of the orbits in
    the series of their data (see GroupPlan.model_d), is not of what
    their instruments record.

    The draws write over the models of the plan, which evaluates them
    again after (see GroupPlan.borrow_plans).

    Parameters
    ----------
    plan : GroupPlan
        The plan of the group of the observations.
    params : dict
        The parameters of the group (see ObservationGroup.pdescs).
    options : MockOptions
        The options: the noise of the data of each observation, the
        number of mock datasets and of draws of the errors of measured
        data, and the seed of the random numbers.

    Returns
    -------
    list of list of Dataset
        The mock data, in the units of the models: those of each mock, a
        dataset for each observation (e.g. the datasets of
        GroupPlan.with_data).

    Raises
    ------
    ValueError
        If the options are not of the observations (see
        MockOptions.check), or a noise does not suit the data of its
        observation (see Noise.draw) or is correlated on the sky by a PSF
        that varies along the spectral axis, which the errors of data
        cannot take (see Data).
    """
    noises = options.noises(plan.group())
    streams = np.random.SeedSequence(options.seed()).spawn(len(noises))
    with plan.borrow_plans() as plans:
        states = plan.galaxy_states(params)
        datasets = [
            _draw(plan, plans[i], i, params, states, noise, stream, options)
            for i, (noise, stream) in enumerate(zip(noises, streams))]
    return [list(mock) for mock in zip(*datasets)]


def _draw(
        plan: 'GroupPlan',
        observable_plan: ObservablePlan,
        index: int,
        params: ParamValues,
        states: Sequence[GalaxyState],
        noise: Noise,
        stream: np.random.SeedSequence,
        options: MockOptions
) -> list[Dataset]:
    """
    Draw the mock datasets of the observation of the given index of a
    group plan (see draw_mocks), by its plan, in the states of the
    galaxies, with the random numbers of the stream.
    """
    group = plan.group()
    observation = group.observations()[index]
    nmocks = options.nmocks()
    with config_path('observations', index):
        exact = observable_plan.has_exact_errors()
        items = _data_options(observation, noise, exact)
        draws, errors = observable_plan.mock(
            group.moved(index, params),
            [states[g] for g in group.seen_galaxies(index)],
            noise, np.random.default_rng(stream),
            nmocks + (0 if exact else options.nsamples()))
    if errors is None:
        errors = _scatter(draws[nmocks:])
    return [
        observation.make_dataset({
            key: Data(item.data, error=errors[key], **items[key])
            for key, item in draw.items()})
        for draw in draws[:nmocks]]


def _data_options(
        observation: Observation, noise: Noise, exact: bool
) -> dict[str, dict[str, Any]]:
    """
    Return the options of the mock data of each data item of an
    observation (see draw_mocks): the unit and the beam of its model
    (see Observation.model_unit), the footprint of its data, if any, and
    the correlation of the noise of those that are what the instrument
    records plus the noise (exact errors).

    Raises
    ------
    ValueError
        If their noise is correlated by the instrument of an observation
        that has none (whose data are not on the sky, e.g. a profile), or
        the spatial correlation of their noise varies along the spectral
        axis (e.g. the PSF of the instrument), which their errors cannot
        take (see Data).
    """
    spatial, spectral = noise.correlations() if exact else (None, None)
    if spatial == INSTRUMENT:
        # (only observations on the sky have an instrument, and draw noise
        # correlated by it; see Noise.draw)
        if not isinstance(observation, ObservationSky):
            raise ValueError(
                f"the spatial correlation of the noise is '{INSTRUMENT}', "
                f"but {observation.type()} observations have no "
                f"instrument: their data are not on the sky")
        spatial = observation.instrument().psf()
    if spatial is not None and spatial.varies():
        raise ValueError(
            "the noise of the mock data is correlated on the sky by a PSF "
            "that varies along the spectral axis (e.g. that of the "
            "instrument), but the errors of data take one correlation for "
            "all the channels: give the noise a correlation that does not "
            "vary")
    data = observation.data()
    options = {}
    for key in observation.keys():
        unit, beam = observation.model_unit(key)
        options[key] = dict(
            unit=unit, beam=beam,
            footprint=None if data is None else data[key].footprint(),
            spatial_correlation=spatial, spectral_correlation=spectral)
    return options


def _scatter(draws: Sequence[HostModelData]) -> dict[str, np.ndarray]:
    """
    Return the scatter of the values of each data item over draws of
    them: the standard deviation of the values that are finite in at
    least two draws (NaN for the others).
    """
    scatter = {}
    for key in draws[0]:
        values = np.stack([draw[key].data for draw in draws])
        finite = np.isfinite(values)
        count = finite.sum(axis=0)
        values = np.where(finite, values, 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean = values.sum(axis=0) / count
            variance = np.where(
                finite, (values - mean) ** 2, 0).sum(axis=0) / (count - 1)
        scatter[key] = np.where(count >= 2, np.sqrt(variance), np.nan)
    return scatter
