"""
The app of the API: its routes under /api/v1 (see routes), behind a
token, and the responses to the errors of the services.
"""

import contextlib
import secrets
from collections.abc import AsyncIterator, Callable
from importlib.metadata import version

from fastapi import (
    Depends, FastAPI, HTTPException, Request, WebSocketException, status)
from fastapi.responses import JSONResponse
from starlette.requests import HTTPConnection

from gbkfit_server.services.errors import Failed, Invalid, NotFound
from gbkfit_server.services.server import Server
from .models import ServerInfo
from .routes import router


__all__ = [
    'API',
    'create_app',
    'openapi'
]


# The path of the API, of its version (that of its contract)
API = '/api/v1'

# The responses to the errors of the services
_ERRORS = {
    NotFound: status.HTTP_404_NOT_FOUND,
    Invalid: status.HTTP_400_BAD_REQUEST,
    Failed: status.HTTP_500_INTERNAL_SERVER_ERROR}


def create_app(server: Server | None, token: str) -> FastAPI:
    """
    Return the app of a server, whose API needs a token: in the header
    Authorization ('Bearer <token>' or 'token <token>'), or in the query
    (?token=..., e.g. of a link, of events or of a WebSocket).

    Parameters
    ----------
    server : Server or None
        The server (None for an app that only gives its schema; see
        openapi).
    token : str
        The token.

    Returns
    -------
    FastAPI
        The app; it closes the server as it shuts down.
    """
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        if server is not None:
            await server.close()

    app = FastAPI(title="GBKFIT server", version='1', lifespan=lifespan)
    app.state.server = server
    app.include_router(
        router, prefix=API, dependencies=[Depends(_authorize(token))])

    @app.get('/')
    def info() -> ServerInfo:
        """Return the name and version of the server, and its API."""
        return ServerInfo(
            name='gbkfit-server', version=version('gbkfit-server'), api=API)

    for error, code in _ERRORS.items():
        app.add_exception_handler(error, _responder(code))
    return app


def openapi() -> dict:
    """Return the schema of the API (its contract, openapi.json)."""
    return create_app(None, '').openapi()


def _authorize(token: str) -> Callable[[HTTPConnection], None]:
    """Return the dependency that checks the token of a request."""
    def authorize(connection: HTTPConnection) -> None:
        given = connection.query_params.get('token')
        scheme, _, value = connection.headers.get('authorization', '') \
            .partition(' ')
        if scheme.lower() in ('bearer', 'token'):
            given = value
        if given is None \
                or not secrets.compare_digest(given.encode(), token.encode()):
            if connection.scope['type'] == 'websocket':
                raise WebSocketException(status.WS_1008_POLICY_VIOLATION)
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED, "a valid token is needed",
                headers={'WWW-Authenticate': 'Bearer'})
    return authorize


def _responder(code: int) -> Callable[[Request, Exception], JSONResponse]:
    """Return the handler of an error of the services, by its code."""
    def respond(_: Request, error: Exception) -> JSONResponse:
        return JSONResponse(dict(detail=str(error)), status_code=code)
    return respond
