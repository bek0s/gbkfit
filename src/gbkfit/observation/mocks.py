"""
The options of mock data of a group of observations: the section 'mock'
of configurations, which the task eval draws (see GroupPlan.mock).
"""

from collections.abc import Sequence
from typing import Annotated, Any

from annotated_types import Ge

from gbkfit.config import Serializable, check_options
from gbkfit.utils import iterutils
from .likelihood import Likelihood, likelihood_parser
from .noise import Noise, noise_parser


__all__ = [
    'MockOptions'
]


class MockOptions(Serializable):
    """
    The options of mock data of the observations of a group (see
    GroupPlan.mock).

    Parameters
    ----------
    noise : Noise or Sequence of Noise
        The noise of the data of every observation, or of each.
    likelihood : Likelihood or Sequence of (Likelihood or None), optional
        The likelihood of the mock data of every observation, or of each;
        by default (None), that of the observation (see
        Observation.with_data).
    nmocks : int, optional
        The number of mock datasets of each observation.
    nsamples : int, optional
        The number of draws of the errors of measured data (e.g.
        moments; see GroupPlan.mock).
    seed : int, optional
        The seed of the random numbers; the same seed gives the same mock
        data. By default, the mock data differ from run to run.

    Raises
    ------
    ValueError
        If nmocks is below 1, or nsamples below 2.
    """

    def __init__(
            self,
            noise: Noise | Sequence[Noise],
            likelihood: Likelihood | Sequence[Likelihood | None] | None = None,
            nmocks: Annotated[int, Ge(1)] = 1,
            nsamples: Annotated[int, Ge(2)] = 100,
            seed: int | None = None
    ):
        check_options(MockOptions.__init__, locals())
        self._noise = noise
        self._likelihood = likelihood
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
            (e.g. an error map; see Noise.dump), followed by
            'observation{i}_' for the noise of each observation i, when
            there is one for each.
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
        if iterutils.is_sequence(self._noise):
            count = len(self._noise)
            noise = noise_parser.dump(
                list(self._noise),
                prefix=[f'{prefix}observation{i}_' for i in range(count)],
                dump_path=[dump_path] * count, overwrite=[overwrite] * count)
        else:
            noise = noise_parser.dump(
                self._noise, prefix=prefix, dump_path=dump_path,
                overwrite=overwrite)
        info: dict[str, Any] = dict(noise=noise)
        if self._likelihood is not None:
            info['likelihood'] = likelihood_parser.dump(self._likelihood)
        info.update(nmocks=self._nmocks, nsamples=self._nsamples)
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

    def per_observation(
            self, nobservations: int
    ) -> tuple[list[Noise], list[Likelihood | None]]:
        """
        Return the noise and the likelihood (or None) of each of a number
        of observations.

        Parameters
        ----------
        nobservations : int
            The number of observations.

        Returns
        -------
        tuple of list
            The noises, and the likelihoods.

        Raises
        ------
        ValueError
            If the noise or the likelihood is a list of another length.
        """
        return (
            _per_observation(self._noise, 'noise', nobservations),
            _per_observation(self._likelihood, 'likelihood', nobservations))


def _per_observation(value: Any, option: str, nobservations: int) -> list:
    """
    Return the value of an option for each of a number of observations:
    one for all, or a list of one for each.
    """
    if not iterutils.is_sequence(value):
        return [value] * nobservations
    if len(value) != nobservations:
        raise ValueError(
            f"option '{option}' is one, or a list of one for each of the "
            f"{nobservations} observations; it is a list of {len(value)}")
    return list(value)
