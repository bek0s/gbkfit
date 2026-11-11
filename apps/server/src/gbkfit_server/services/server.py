"""
The server: its services, which the API serves (see gbkfit_server.api).
"""

from .checks import Checks
from .events import Events
from .galaxies import Galaxies
from .executors import Executor
from .jobs import Jobs
from .preferences import Preferences
from .render_pool import RenderPool
from .responses import Responses
from .sessions import Sessions
from .views import Views
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
    preferences : Preferences, optional
        The preferences of the user; by default those in their folder
        (see Preferences).

    Attributes
    ----------
    workspace : Workspace
    events : Events
    jobs : Jobs
    sessions : Sessions
    checks : Checks
    views : Views
    responses : Responses
    galaxies : Galaxies
    renders : RenderPool
    preferences : Preferences
    """

    def __init__(
            self, root: str, executor: Executor,
            preferences: Preferences | None = None):
        self.workspace = Workspace(root)
        self.events = Events()
        self.jobs = Jobs(self.workspace, executor, self.events)
        self.sessions = Sessions(self.workspace)
        self.checks = Checks(self.workspace)
        self.views = Views(self.workspace)
        self.responses = Responses(self.workspace)
        self.galaxies = Galaxies(self.workspace)
        self.renders = RenderPool(self.workspace, self.events)
        self.preferences = preferences if preferences is not None \
            else Preferences()

    async def close(self) -> None:
        """Close the sessions, stop the jobs, and end the renders."""
        await self.sessions.close_all()
        await self.jobs.close()
        await self.renders.close()
