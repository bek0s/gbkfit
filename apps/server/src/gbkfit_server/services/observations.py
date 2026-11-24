"""
The observations of the projects, as their screens show them: the problems
of an observation of the options of a piece alone, without data, as the
library finds them (see gbkfit.setup.describe_observation; those of the
data it is paired with are of the setups that pair them), and those of a
part that observations share by name (a PSF, an LSF or a noise); the
sampling of an observation (its field on the sky and its channels, see
Sampling); and what a dataset gives an observation of a kind (its
sampling and its beam, see DatasetGives), to copy it.
"""

import dataclasses
import os
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

import gbkfit.project
from gbkfit.config import (
    ConfigError, ConfigProblem, collect_problems, error_message,
    paths_relative_to, strict_mode)
from gbkfit.dataset import dataset_parser
from gbkfit.grid import sky_positions
from gbkfit.instrument import lsf_parser, psf_parser
from gbkfit.observation import ObservationSky, noise_parser, observation_parser
from gbkfit.setup import describe_observation

from .errors import Invalid
from .workspace import Workspace


__all__ = [
    'Channels',
    'DatasetGives',
    'Field',
    'Observations',
    'Sampling',
    'channels_of',
    'field_of'
]


@dataclasses.dataclass
class Field:
    """
    The field of an observation on the sky: its centre (x, y; arcsec, as
    the model measures positions: from the reference pixel of its grid,
    see sky_positions; not that pixel itself, which can be off the
    image), its width and height (arcsec), and the rotation of its pixels
    on the sky (degrees; see Coords.rota).
    """
    centre: tuple[float, float]
    size: tuple[float, float]
    rota: float


@dataclasses.dataclass
class Channels:
    """
    The channels of an observation: the velocities of the first and of
    the last (km/s, in the convention of its spectral axis), their width
    (km/s) and their number.
    """
    first: float
    last: float
    step: float
    size: int


@dataclasses.dataclass
class Sampling:
    """
    The sampling of an observation, as far as it is known (None for what
    it has not, or what cannot be known, e.g. of an observation with a
    problem): its field on the sky (of a grid of pixels on the sky) and
    its channels.
    """
    field: Field | None
    channels: Channels | None


@dataclasses.dataclass
class DatasetGives:
    """
    What a dataset gives an observation of a kind, to copy into it: the
    options of its sampling (see Observation.data_options), and its PSF,
    the Gaussian of the beam of the data, if they have one beam (None,
    with why in psf_note, if not).
    """
    sampling: dict[str, Any]
    psf: dict[str, Any] | None
    psf_note: str | None


# The parsers of the parts that observations share, by family
_PARTS = dict(psf=psf_parser, lsf=lsf_parser, noise=noise_parser)


class Observations:
    """The observations of the projects of a workspace."""

    def __init__(self, workspace: Workspace):
        self._workspace = workspace

    def describe(
            self, project: str, observation: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Return the problems of an observation of the options of a piece of
        a project alone, without data (with the pieces it names, of the
        project), each at the path of its option, as the library finds
        them (see describe_observation).
        """
        root = self._workspace.path(project)
        problems, _ = self._described(root, observation)
        return [_problem(problem) for problem in problems]

    def sampling(
            self, project: str, observation: Mapping[str, Any]
    ) -> Sampling:
        """
        Return the sampling of an observation of the options of a piece of
        a project alone (with the pieces it names, of the project): its
        field on the sky and its channels, as far as the library can make
        it (none of either, for an observation it cannot make).
        """
        _, made = self._described(self._workspace.path(project), observation)
        if made is None:
            return Sampling(None, None)
        return Sampling(field_of(made), channels_of(made))

    def of_dataset(
            self, project: str, type_: str, dataset: str | Mapping[str, Any]
    ) -> DatasetGives:
        """
        Return what a dataset (a piece of a project by name, or its
        options) gives an observation of a type: the options of its
        sampling and its beam (see DatasetGives).

        Raises
        ------
        Invalid
            If the type is not one of an observation, or the dataset
            cannot be read, or is of another kind of data.
        """
        root = self._workspace.path(project)
        try:
            resolved = gbkfit.project.resolve(
                dict(datasets=[dataset]), os.path.join(root, 'observe.yaml'),
                gbkfit.project.open(root))
            with paths_relative_to(root):
                data = dataset_parser.load(resolved.config['datasets'][0])
            kind = observation_parser.registered_class(type_)
            sampling = kind.data_options(data)
        except (ConfigError, ValueError, TypeError, OSError) as e:
            raise Invalid(error_message(e)) from e
        beams = [data[key].beam() for key in data.keys()]
        beams = [beam for beam in beams if beam is not None]
        if not beams:
            return DatasetGives(sampling, None, "the data have no beam")
        beam = beams[0]
        if beam.spectral is not None:
            return DatasetGives(
                sampling, None,
                "the data have a beam for each channel, which a table of "
                "the PSF gives (prep psf_casa_beams)")
        psf = dict(
            type='gauss', bmaj=float(beam.bmaj), bmin=float(beam.bmin),
            bpa=float(beam.bpa))
        return DatasetGives(sampling, psf, None)

    @staticmethod
    def _described(
            root: str, observation: Mapping[str, Any]
    ) -> tuple[list[ConfigProblem], Any]:
        """
        Return the problems of an observation of the options of a piece of
        the project of a folder alone, without data (with the pieces it
        names, of the project), and the observation, if the library could
        make it (see describe_observation).
        """
        resolved = None
        # (the pieces it names, in place: as the observation of a setup)
        with collect_problems() as problems:
            resolved = gbkfit.project.resolve(
                dict(observations=[dict(observation)]),
                os.path.join(root, 'observe.yaml'), gbkfit.project.open(root))
        if resolved is None:
            return list(problems), None
        description = describe_observation(
            resolved.config['observations'][0], root)
        return list(description.problems), description.observation

    def describe_part(
            self,
            project: str,
            family: Literal['psf', 'lsf', 'noise'],
            part: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Return the problems of a part that observations share (a PSF, an
        LSF or a noise) of the options of a piece of a project (its files
        relative to the project), each at the path of its option.
        """
        root = self._workspace.path(project)
        with strict_mode(False), collect_problems() as problems, \
                paths_relative_to(root):
            _PARTS[family].load(dict(part))
        return [_problem(problem) for problem in problems]


def field_of(observation: Any) -> Field | None:
    """
    Return the field of an observation on the sky (see Field): of its grid
    of pixels on the sky, if it has one whose first two axes are x and y;
    else None.
    """
    if not isinstance(observation, ObservationSky):
        return None
    grid = observation.grid()
    spatial = [
        axis for axis in range(len(grid.size))
        if axis != grid.spectral_axis]
    if spatial[:2] != [0, 1]:
        return None
    x, y = sky_positions(grid)
    # (the middle of the first and the last pixels: that of the field)
    centre = (float(x[0, 0] + x[-1, -1]) / 2, float(y[0, 0] + y[-1, -1]) / 2)
    nx, ny = grid.size[:2]
    step = grid.coords.step
    return Field(
        centre, (abs(nx * float(step[0])), abs(ny * float(step[1]))),
        float(grid.coords.rota))


def channels_of(observation: Any) -> Channels | None:
    """
    Return the channels of an observation (see Channels), if it has a
    spectral axis of a known number of them; else None.
    """
    spectral = observation.spectral() \
        if isinstance(observation, ObservationSky) else None
    if spectral is None or spectral.size() is None:
        return None
    velocities = spectral.rval() + spectral.step() * (
        np.arange(spectral.size()) - spectral.rpix())
    return Channels(
        float(velocities[0]), float(velocities[-1]), float(spectral.step()),
        int(spectral.size()))


def _problem(problem: ConfigProblem) -> dict[str, Any]:
    """
    Return a problem as the API gives it, at the path of its option in the
    observation (not in the setup it is resolved in, see describe).
    """
    path = list(problem.path)
    if path[:2] == ['observations', 0]:
        path = path[2:]
    return dict(severity=problem.severity, path=path, message=problem.message)
