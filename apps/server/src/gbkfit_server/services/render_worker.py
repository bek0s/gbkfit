"""
The worker of the renders (see render_pool): a process that renders the
galaxies of a workspace (see Renders) one request at a time, until its
input ends:

    python -m gbkfit_server.services.render_worker WORKSPACE

It writes frames to stdout (see frames): {type: ready} once the library
is imported; then, for each frame it reads from stdin, {type: render, id,
kind, args} (kind: 'galaxy', 'cube' or 'observed', the method of Renders,
and args its keyword arguments), {type: progress, id, stage} for each
stage that the render reaches (e.g. 'evaluating'), and its result, {type:
result, id} with the pickled result as data, or {type: error, id, error,
message}, error the name of its class of errors (see errors: 'Invalid',
'NotFound', or 'Failed' for any other). Its log goes to stderr.

A render is stopped by ending the process: its native code cannot be
interrupted in place.
"""

import logging
import os
import pickle
import sys
from typing import BinaryIO

from gbkfit_server.services import frames


def main() -> None:
    """Render the requests of stdin, in the workspace of the command line."""
    # stdout carries the frames: everything else written to it (e.g. by
    # imports, as the library is imported here) goes to stderr
    out = os.fdopen(os.dup(1), 'wb')
    os.dup2(2, 1)
    from gbkfit.tasks import log_handler
    from gbkfit_server.services.errors import Invalid, NotFound
    from gbkfit_server.services.renders import Renders
    from gbkfit_server.services.workspace import Workspace

    logger = logging.getLogger('gbkfit')
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler())
    renders = Renders(Workspace(sys.argv[1]))
    _write(out, frames.encode(dict(type='ready')))
    while (frame := frames.read(sys.stdin.buffer)) is not None:
        header, _ = frame
        id_ = header['id']

        def progress(stage: str, id_: str = id_) -> None:
            _write(out, frames.encode(
                dict(type='progress', id=id_, stage=stage)))
        try:
            result = getattr(renders, header['kind'])(
                **header['args'], progress=progress)
        except (Invalid, NotFound) as e:
            _write(out, frames.encode(dict(
                type='error', id=id_, error=type(e).__name__,
                message=str(e))))
            continue
        # (any other error is a failure of the render, perhaps a bug: logged
        # with its traceback, and the worker goes on)
        except Exception as e:
            logger.exception("the render %s failed", header['kind'])
            _write(out, frames.encode(dict(
                type='error', id=id_, error='Failed', message=str(e))))
            continue
        _write(out, frames.encode(
            dict(type='result', id=id_), pickle.dumps(result)))


def _write(out: BinaryIO, frame: bytes) -> None:
    """Write a frame, at once."""
    out.write(frame)
    out.flush()


if __name__ == '__main__':
    main()
