"""
Mock data of a group of observations (see draw_mocks), and their
options: the section 'mock' of configurations, which the task eval draws
(see MockOptions).
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
from annotated_types import Ge

from gbkfit.config import Serializable, check_options, config_path
from gbkfit.dataset import Data, Dataset
from gbkfit.model import GalaxyState
from gbkfit.params import ParamValues
from .noise import INSTRUMENT, Noise, NoiseGauss
from .plan import HostModelData, ObservationPlan
from .observation import Observation
from .sky import ObservationSky

if TYPE_CHECKING:
    from .group_plan import GroupPlan


__all__ = [
    'MockOptions',
    'draw_mocks',
    'noise_of'
]


class MockOptions(Serializable):
    """
    The options of mock data of the observations of a group (see
    draw_mocks): how many, and their random numbers. The noise of the
    mock data of each observation is its own (see Observation: its option
    noise; by default, that of the errors of its data). Fits of the mock
    data compare them with the models under the likelihoods of the setup
    (see Likelihoods), which a setup of mock data can give for them.

    Parameters
    ----------
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
    Ten mock datasets of each observation, the same from run to run (the
    noise of each in the observation):

    .. code-block:: yaml

        observations:
        - type: pixel_spectra
          noise: {type: gauss, rms: 0.01}
          ...
        mock:
          nmocks: 10
          seed: 1
    """

    def __init__(
            self,
            nmocks: Annotated[int, Ge(1)] = 1,
            nsamples: Annotated[int, Ge(2)] = 100,
            seed: int | None = None
    ):
        check_options(MockOptions.__init__, locals())
        self._nmocks = nmocks
        self._nsamples = nsamples
        self._seed = seed

    def dump(self) -> dict[str, Any]:
        """Dump the options to their configuration."""
        info: dict[str, Any] = dict(
            nmocks=self._nmocks, nsamples=self._nsamples)
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


def noise_of(observation: Observation, exact: bool) -> Noise:
    """
    Return the noise of the mock data of an observation (see draw_mocks):
    its own (its option noise); without one, Gaussian noise of the errors
    of its data, with their correlations, if its data are what its
    instrument records plus the noise (exact errors, see
    ObservationPlan.has_exact_errors; e.g. spectra, an image), in one
    item, with errors.

    Raises
    ------
    ValueError
        If it has no noise, and its data cannot give one.
    """
    if (noise := observation.noise()) is not None:
        return noise
    data = observation.dataset()
    why = None
    if data is None:
        why = "it has no data, whose errors would be its noise"
    elif not exact:
        why = (
            "its data are measured from what its instrument records (e.g. "
            "moments of a cube), to which the noise is added, so their "
            "errors are not its noise")
    elif len(observation.keys()) > 1:
        why = (
            f"its data are of several items ({list(observation.keys())}), "
            f"each of its own errors, but it has one noise")
    elif (item := data[observation.keys()[0]]).error() is None:
        why = "its data have no errors, which would be its noise"
    if why is not None:
        raise ValueError(
            f"the observation has no noise for its mock data (its option "
            f"noise): {why}")
    return NoiseGauss(
        error=item.error(), spatial_correlation=item.spatial_correlation(),
        spectral_correlation=item.spectral_correlation())


def draw_mocks(
        plan: 'GroupPlan', params: ParamValues, options: MockOptions
) -> list[list[Dataset]]:
    """
    Draw mock data of the observations of a group plan, by the options:
    their model data measured from what their instruments record of the
    model plus noise (see ObservationPlan.mock), with their errors.

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

    The noise of each observation is its own (see noise_of). The draws
    write over the models of the plan, which evaluates them again after
    (see GroupPlan.borrow_plans).

    Parameters
    ----------
    plan : GroupPlan
        The plan of the group of the observations.
    params : dict
        The parameters of the group (see ObservationGroup.pdescs).
    options : MockOptions
        The options: the number of mock datasets and of draws of the
        errors of measured data, and the seed of the random numbers.

    Returns
    -------
    list of list of Dataset
        The mock data, in the units of the models: those of each mock, a
        dataset for each observation (e.g. the datasets of
        GroupPlan.with_datasets).

    Raises
    ------
    ValueError
        If an observation has no noise and its data cannot give one (see
        noise_of), or a noise does not suit the data of its
        observation (see Noise.draw) or is correlated on the sky by a PSF
        that varies along the spectral axis, which the errors of data
        cannot take (see Data).
    """
    observations = plan.group().observations()
    streams = np.random.SeedSequence(options.seed()).spawn(len(observations))
    with plan.borrow_plans() as plans:
        states = plan.galaxy_states(params)
        datasets = [
            _draw(plan, plans[i], i, params, states, stream, options)
            for i, stream in enumerate(streams)]
    return [list(mock) for mock in zip(*datasets)]


def _draw(
        plan: 'GroupPlan',
        observation_plan: ObservationPlan,
        index: int,
        params: ParamValues,
        states: Sequence[GalaxyState],
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
        exact = observation_plan.has_exact_errors()
        noise = noise_of(observation, exact)
        items = _data_options(observation, noise, exact)
        draws, errors = observation_plan.mock(
            group.moved(index, params), plan.scene_state(index, states),
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
    (see Observation.model_unit), and the correlation of the noise of
    those that are what the instrument records plus the noise (exact
    errors). Where the instrument looked is that of the data of the
    observation, which its datasets keep (see make_dataset).

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
    options = {}
    for key in observation.keys():
        unit, beam = observation.model_unit(key)
        options[key] = dict(
            unit=unit, beam=beam, spatial_correlation=spatial,
            spectral_correlation=spectral)
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
