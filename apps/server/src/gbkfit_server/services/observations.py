"""
The observations of the projects, as their home shows them: the problems
of an observation of the options of a piece alone, without data, as the
library finds them (see gbkfit.setup.describe_observation; those of the
data it is paired with are of the setups that pair them), and those of a
part that observations share by name (a PSF, an LSF or a noise).
"""

import os
from collections.abc import Mapping
from typing import Any, Literal

import gbkfit.project
from gbkfit.config import (
    ConfigProblem, collect_problems, paths_relative_to, strict_mode)
from gbkfit.instrument import lsf_parser, psf_parser
from gbkfit.observation import noise_parser
from gbkfit.setup import describe_observation

from .workspace import Workspace


__all__ = [
    'Observations'
]


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
        resolved = None
        # (the pieces it names, in place: as the observation of a setup)
        with collect_problems() as problems:
            resolved = gbkfit.project.resolve(
                dict(observations=[dict(observation)]),
                os.path.join(root, 'observe.yaml'), gbkfit.project.open(root))
        if resolved is not None:
            problems = describe_observation(
                resolved.config['observations'][0], root).problems
        return [_problem(problem) for problem in problems]

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


def _problem(problem: ConfigProblem) -> dict[str, Any]:
    """
    Return a problem as the API gives it, at the path of its option in the
    observation (not in the setup it is resolved in, see describe).
    """
    path = list(problem.path)
    if path[:2] == ['observations', 0]:
        path = path[2:]
    return dict(severity=problem.severity, path=path, message=problem.message)
