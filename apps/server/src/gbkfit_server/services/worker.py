"""
The worker of a live session (see sessions): a process that sets up a
configuration once (see gbkfit.tasks.eval.load), and then evaluates its
models at the values of its free parameters that it reads, until its
input ends:

    python -m gbkfit_server.services.worker CONFIG

It reads frames from stdin ({type: evaluate, values: {element: value}}),
and writes frames to stdout (see frames): first {type: ready,
observations, free} (or {type: error, message}), then the result of each
evaluation (the models; see frames.array_frame) or an error. Its log goes
to stderr.
"""

import logging
import os
import sys
from typing import BinaryIO

from gbkfit_server.services import frames


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
        group, params = load(sys.argv[1])
    except ConfigError as e:
        _write(out, frames.encode(dict(type='error', message=str(e))))
        sys.exit(1)
    free = params.free_properties()
    values = {name: info.get('value') for name, info in free.items()}
    observations = [
        dict(label=group.observation_label(i), items=list(items))
        for i, items in enumerate(group.model_h(params.evaluate(values)))]
    _write(out, frames.encode(dict(
        type='ready', observations=observations, free=free)))
    while (frame := frames.read(sys.stdin.buffer)) is not None:
        header, _ = frame
        try:
            if unknown := set(header['values']) - set(free):
                raise ValueError(f"no free elements {sorted(unknown)}")
            values |= header['values']
            models = group.model_h(params.evaluate(values))
            _write(out, frames.array_frame(dict(type='result'), [
                (dict(observation=o['label'], item=item), parts['d'])
                for o, items in zip(observations, models)
                for item, parts in items.items()]))
        except Exception as e:
            _write(out, frames.encode(dict(type='error', message=str(e))))


def _write(out: BinaryIO, frame: bytes) -> None:
    """Write a frame, at once."""
    out.write(frame)
    out.flush()


if __name__ == '__main__':
    main()
