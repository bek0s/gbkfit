"""
The server: its services, which the API serves (see gbkfit_server.api).
"""

from .events import Events
from .executors import Executor
from .jobs import Jobs
from .sessions import Sessions
from .workspace import Workspace


__all__ = [
    'Server'
]


class Server:
    """
    The services of a server of a folder.

    Parameters
    ----------
    root : str
        The folder (see Workspace).
    executor : Executor
        What runs the jobs.

    Attributes
    ----------
    workspace : Workspace
    events : Events
    jobs : Jobs
    sessions : Sessions
    """

    def __init__(self, root: str, executor: Executor):
        self.workspace = Workspace(root)
        self.events = Events()
        self.jobs = Jobs(self.workspace, executor, self.events)
        self.sessions = Sessions(self.workspace)

    async def close(self) -> None:
        """Close the sessions, and stop the jobs."""
        await self.sessions.close_all()
        await self.jobs.close()
