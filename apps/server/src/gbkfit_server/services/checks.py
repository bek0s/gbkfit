"""
The checks of the texts of files of the workspace (e.g. as they are
edited): each in a process of its own (see checker), which sets their
models up as their tasks would, away from the server.
"""

import asyncio
import json
import sys
from typing import Any

from .errors import Failed
from .workspace import Workspace


__all__ = [
    'Checks'
]


# The longest a check takes (seconds)
_TIMEOUT = 120


class Checks:
    """
    The checks of the texts of files of a workspace.

    Parameters
    ----------
    workspace : Workspace
        The workspace, where the checkers run.
    """

    def __init__(self, workspace: Workspace):
        self._workspace = workspace

    async def check(self, path: str, text: str) -> list[dict[str, Any]]:
        """
        Check the text of a file (e.g. not saved yet), and return its
        diagnostics: their lines (from 1), severities and messages (see
        checker).

        Raises
        ------
        Invalid
            If the path is out of the workspace.
        Failed
            If the check fails, or takes longer than _TIMEOUT.
        """
        request = json.dumps(dict(path=self._workspace.path(path), text=text))
        process = await asyncio.create_subprocess_exec(
            sys.executable, '-m', 'gbkfit_server.services.checker',
            cwd=self._workspace.root, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True)
        try:
            out, err = await asyncio.wait_for(
                process.communicate(request.encode()), _TIMEOUT)
        except TimeoutError as e:
            process.kill()
            await process.wait()
            raise Failed(f"the check of {path} took over {_TIMEOUT} s") from e
        if process.returncode != 0:
            raise Failed(
                f"the check of {path} failed:\n{err.decode()[-2000:]}")
        return json.loads(out)['diagnostics']
