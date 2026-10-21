"""
gbkfit-server: serve the projects of a folder (see gbkfit_server).

    gbkfit-server [FOLDER] [--ip IP] [--port PORT] [--jobs N] [--gpus LIST]
                  [--open]

It prints the link to connect with, which has its token (and opens it
with --open). It listens on
this machine only by default; reach a remote one through an SSH tunnel
(ssh -L 8765:localhost:8765 HOST).
"""

import argparse
import logging
import os
import secrets
import sys
import threading
import webbrowser
from importlib.metadata import version

import uvicorn

from gbkfit_server.api import create_app
from gbkfit_server.services.executors import LocalExecutor
from gbkfit_server.services.server import Server


def _make_parser() -> argparse.ArgumentParser:
    """Return the parser of the command line."""
    parser = argparse.ArgumentParser(
        prog='gbkfit-server',
        description="Serve the projects of a folder of GBKFIT.")
    parser.add_argument(
        '-v', '--version', action='version', version=version('gbkfit-server'))
    parser.add_argument(
        'folder', nargs='?', default='.',
        help="the folder of the projects; default: the current one")
    parser.add_argument(
        '--ip', default='127.0.0.1',
        help="the address to listen on; default: 127.0.0.1 (this machine)")
    parser.add_argument(
        '--port', type=int, default=8765, help="the port; default: 8765")
    parser.add_argument(
        '--jobs', type=int,
        help="the number of jobs that run at once; default: 1, or one per "
             "GPU")
    parser.add_argument(
        '--gpus', type=lambda text: text.split(','), default=[],
        metavar='LIST',
        help="the GPUs of the jobs (e.g. 0,1), one per job")
    parser.add_argument(
        '--token',
        help="the token of the clients; default: $GBKFIT_SERVER_TOKEN, or "
             "a new random one")
    parser.add_argument(
        '--open', action='store_true',
        help="open the link in the browser (e.g. the Studio)")
    return parser


def main() -> None:
    """Run gbkfit-server: serve a folder, until Ctrl-C."""
    parser = _make_parser()
    args = parser.parse_args()
    if args.jobs is not None and args.jobs < 1:
        parser.error("argument --jobs: must be at least 1")
    try:
        executor = LocalExecutor(
            args.jobs or max(1, len(args.gpus)), args.gpus)
        server = Server(args.folder, executor)
    except (ValueError, NotADirectoryError) as e:
        parser.error(str(e))
    token = args.token or os.environ.get('GBKFIT_SERVER_TOKEN') \
        or secrets.token_urlsafe(32)
    logging.basicConfig(level=logging.INFO, format='{message}', style='{')
    link = f"http://{args.ip}:{args.port}/?token={token}"
    print(
        f"gbkfit-server {version('gbkfit-server')}: serving "
        f"{server.workspace.root}\n    {link}", file=sys.stderr, flush=True)
    if args.open:
        # (once the server listens)
        threading.Timer(1.0, webbrowser.open, [link]).start()
    uvicorn.run(
        create_app(server, token), host=args.ip, port=args.port,
        log_level='warning')
