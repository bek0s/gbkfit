"""
The worker of a live session (see sessions): a process that sets up a
configuration once (see gbkfit.load), and then evaluates its models at
the values of its free parameters that it reads, until its input ends:

    python -m gbkfit_server.services.worker CONFIG

It reads frames from stdin ({type: evaluate, values: {element: value}}),
and writes frames to stdout (see frames): first {type: ready, free, data}
with the arrays of the data, if any (or {type: error, message}), then
the result of each evaluation, {type: result, chi2} with the arrays of
the models and residuals, and their images (see _RESULT), or an error.
Each array has its kind, title and ids (e.g. observation and item). Its
log goes to stderr.
"""

import logging
import os
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np

from gbkfit_server.services import frames

if TYPE_CHECKING:
    from gbkfit.params import ParamSpace


# The kinds of the outputs (see gbkfit.Setup.outputs) of the data, sent
# once, and of each evaluation
_DATA = ('data', 'data_image')
_RESULT = ('model', 'model_image', 'residual', 'residual_image')


def main() -> None:
    """Run the worker of the configuration of the command line."""
    # stdout carries the frames: everything else written to it (e.g. by
    # imports, as the library is imported here) goes to stderr
    out = os.fdopen(os.dup(1), 'wb')
    os.dup2(2, 1)
    from gbkfit import load
    from gbkfit.tasks import log_handler
    from gbkfit.params import ParamEvaluationError
    from gbkfit.utils.parseutils import ConfigError

    logger = logging.getLogger('gbkfit')
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler())
    try:
        setup = load(sys.argv[1])
    except ConfigError as e:
        _write(out, frames.encode(dict(type='error', message=str(e))))
        sys.exit(1)
    params, objective = setup.params, setup.objective
    free = params.free_properties()
    try:
        values = _start_values(params)
        outputs = setup.outputs(setup.evaluate(values), data=True)
    except (ConfigError, ParamEvaluationError) as e:
        _write(out, frames.encode(dict(type='error', message=str(e))))
        sys.exit(1)
    _write(out, frames.array_frame(
        dict(type='ready', free=free, data=objective is not None),
        _arrays(outputs, _DATA)))
    while (frame := frames.read(sys.stdin.buffer)) is not None:
        header, _ = frame
        try:
            if unknown := set(header['values']) - set(free):
                raise ValueError(f"no free elements {sorted(unknown)}")
            # (a value it refuses does not stay for the next frames)
            changed = values | header['values']
            evaluation = setup.evaluate(changed)
            values = changed
            outputs = setup.outputs(evaluation)
            chi2 = None if evaluation.objective is None \
                else evaluation.objective.merits.chi2
            _write(out, frames.array_frame(
                dict(type='result', chi2=chi2), _arrays(outputs, _RESULT)))
        except Exception as e:
            _write(out, frames.encode(dict(type='error', message=str(e))))


def _start_values(params: 'ParamSpace') -> dict[str, float]:
    """
    Return the start value of each free element: its value, or else (e.g.
    of a configuration of a sampler) the median of its prior, or of its
    bounds.

    Raises
    ------
    ConfigError
        If an element has no value, no prior and not both bounds.
    """
    return {
        name: element.value() if element.value() is not None
        else float(element.proper_prior().ppf(0.5))
        for name, element in params.free_elements().items()}


def _arrays(
        outputs: Mapping[str, Any],
        kinds: tuple[str, ...]
) -> list[tuple[dict[str, Any], np.ndarray]]:
    """
    Return the arrays of the outputs of the given kinds (see
    gbkfit.Setup.outputs), each with its kind, title and ids (e.g.
    observation and item); not those of other values (e.g. tables).
    """
    arrays = []
    for output in outputs.values():
        data = getattr(output.value, 'data', output.value)
        if output.kind in kinds and isinstance(data, np.ndarray) \
                and data.dtype.kind in 'fiu':
            arrays.append((
                dict(kind=output.kind, title=output.title, **output.ids),
                data))
    return arrays

def _write(out: BinaryIO, frame: bytes) -> None:
    """Write a frame, at once."""
    out.write(frame)
    out.flush()


if __name__ == '__main__':
    main()
