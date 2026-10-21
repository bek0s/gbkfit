"""
The worker of a live session (see sessions): a process that sets up a
configuration once (see gbkfit.tasks.eval.load), and then evaluates its
models at the values of its free parameters that it reads, until its
input ends:

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
from typing import Any, BinaryIO

import numpy as np

from gbkfit_server.services import frames


# The kinds of the outputs (see gbkfit.tasks.eval.Evaluation.outputs) of
# the data, sent once, and of each evaluation
_DATA = ('data', 'data_image')
_RESULT = ('model', 'model_image', 'residual', 'residual_image')


def main() -> None:
    """Run the worker of the configuration of the command line."""
    # stdout carries the frames: everything else written to it (e.g. by
    # imports, as the library is imported here) goes to stderr
    out = os.fdopen(os.dup(1), 'wb')
    os.dup2(2, 1)
    from gbkfit.tasks import log_handler
    from gbkfit.tasks.eval import load
    from gbkfit.utils.parseutils import ConfigError

    logger = logging.getLogger('gbkfit')
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler())
    try:
        evaluation = load(sys.argv[1])
    except ConfigError as e:
        _write(out, frames.encode(dict(type='error', message=str(e))))
        sys.exit(1)
    params, objective = evaluation.params, evaluation.objective
    free = params.free_properties()
    values = {name: info.get('value') for name, info in free.items()}
    outputs = evaluation.outputs(params.evaluate(values), data=True)
    _write(out, frames.array_frame(
        dict(type='ready', free=free, data=objective is not None),
        _arrays(outputs, _DATA)))
    while (frame := frames.read(sys.stdin.buffer)) is not None:
        header, _ = frame
        try:
            if unknown := set(header['values']) - set(free):
                raise ValueError(f"no free elements {sorted(unknown)}")
            values |= header['values']
            param_values = params.evaluate(values)
            outputs = evaluation.outputs(param_values)
            chi2 = None if objective is None \
                else objective.chi2(param_values)
            _write(out, frames.array_frame(
                dict(type='result', chi2=chi2), _arrays(outputs, _RESULT)))
        except Exception as e:
            _write(out, frames.encode(dict(type='error', message=str(e))))


def _arrays(
        outputs: Mapping[str, Any],
        kinds: tuple[str, ...]
) -> list[tuple[dict[str, Any], np.ndarray]]:
    """
    Return the arrays of the outputs of the given kinds (see
    gbkfit.tasks.eval.Evaluation.outputs), each with its kind, title and
    ids (e.g. observation and item); not those of other values (e.g.
    tables).
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
