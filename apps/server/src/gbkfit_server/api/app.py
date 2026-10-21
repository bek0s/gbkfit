"""
The app of the server: its API, under /api/v1 (see routes), behind a
token, with the responses to the errors of the services; and its pages
(see pages).
"""

import contextlib
import secrets
from collections.abc import AsyncIterator, Callable, Mapping
from importlib.metadata import version

from fastapi import (
    Depends, FastAPI, HTTPException, Request, WebSocketException, status)
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import HTTPConnection

from gbkfit_server.services.errors import Failed, Invalid, NotFound
from gbkfit_server.services.server import Server
from .models import ServerInfo
from .pages import installed_pages
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


def create_app(
        server: Server | None,
        token: str,
        pages: Mapping[str, str] | None = None
) -> FastAPI:
    """
    Return the app of a server, whose API needs a token: in the header
    Authorization ('Bearer <token>' or 'token <token>'), or in the query
    (?token=..., e.g. of a link, of events or of a WebSocket).

    It serves the pages at /<name>/, and / leads to the first of them
    (with the query, e.g. the token), or else to the information of the
    server (/api).

    Parameters
    ----------
    server : Server or None
        The server (None for an app that only gives its schema; see
        openapi).
    token : str
        The token.
    pages : Mapping of str to str, optional
        The folders of the built pages, by name; by default those
        installed (see installed_pages).

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

    pages = installed_pages() if pages is None else dict(pages)

    @app.get('/api')
    def info() -> ServerInfo:
        """
        Return the name and version of the server, its API, and the pages
        it serves.
        """
        return ServerInfo(
            name='gbkfit-server', version=version('gbkfit-server'), api=API,
            pages=list(pages))

    @app.get('/', include_in_schema=False)
    def home(request: Request) -> RedirectResponse:
        target = f'/{next(iter(pages))}/' if pages else '/api'
        query = request.url.query
        return RedirectResponse(f'{target}?{query}' if query else target)

    for name, folder in pages.items():
        app.mount(f'/{name}', StaticFiles(directory=folder, html=True), name)

    for error, code in _ERRORS.items():
        app.add_exception_handler(error, _responder(code))
    return app


def openapi() -> dict:
    """Return the schema of the API (its contract, openapi.json)."""
    return create_app(None, '', {}).openapi()


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
