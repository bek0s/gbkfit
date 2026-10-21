"""
The routes of the API (version 1): thin, over the services of the server.
Paths are relative to its workspace. Arrays and the results of sessions
are frames (see gbkfit_server.services.frames). The routes of the files
run in threads (they block); those of the jobs, events and sessions in
the event loop of their services.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import (
    APIRouter, Body, Depends, Response, WebSocket, WebSocketException,
    status)
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from starlette.requests import HTTPConnection

from gbkfit_server.services import frames
from gbkfit_server.services.errors import Failed, NotFound
from gbkfit_server.services.server import Server
from gbkfit_server.services.sessions import Latest
from .models import (
    CheckRequest, CheckResult, JobInfo, JobRequest, ProjectInfo, RunInfo,
    SessionInfo, SessionRequest)


__all__ = [
    'router'
]


router = APIRouter()


def _server(connection: HTTPConnection) -> Server:
    """Return the server of the app (see create_app)."""
    return connection.app.state.server


ServerDep = Annotated[Server, Depends(_server)]


@router.get('/projects')
def projects(server: ServerDep) -> list[str]:
    """List the paths of the projects of the workspace."""
    return server.workspace.projects()


@router.get('/projects/{path:path}')
def project(server: ServerDep, path: str) -> ProjectInfo:
    """Return a project: its pieces, setups and runs."""
    return ProjectInfo.model_validate(
        server.workspace.project(path), from_attributes=True)


@router.get('/files/{path:path}', response_class=PlainTextResponse)
def read_file(server: ServerDep, path: str) -> str:
    """Return the text of a file (e.g. a piece or a setup)."""
    return server.workspace.read_text(path)


@router.put('/files/{path:path}', status_code=status.HTTP_204_NO_CONTENT)
def write_file(
        server: ServerDep,
        path: str,
        text: Annotated[str, Body(media_type='text/plain')]
) -> None:
    """Write the text of a file, and its folders."""
    server.workspace.write_text(path, text)


@router.get(
    '/raw/{path:path}', response_class=FileResponse,
    responses={200: {'content': {'application/octet-stream': {}}}})
def raw(server: ServerDep, path: str) -> FileResponse:
    """
    Return a file as it is, of the type of its name (e.g. a figure, to
    show in a page with the token in the query).
    """
    return FileResponse(server.workspace.file(path))


@router.post('/check')
async def check(server: ServerDep, request: CheckRequest) -> CheckResult:
    """
    Check the text of a file (e.g. as it is edited) as its task would:
    a configuration itself, a piece of a project in the setups that name
    it; and return its problems, by line.
    """
    return CheckResult.model_validate(dict(
        diagnostics=await server.checks.check(request.path, request.text)))


@router.get('/runs/{path:path}')
def run(server: ServerDep, path: str) -> RunInfo:
    """Return a run: its manifest."""
    return RunInfo.model_validate(server.workspace.run(path))


@router.get(
    '/arrays/{path:path}', response_class=Response,
    responses={200: {'content': {'application/octet-stream': {}}}})
def array(server: ServerDep, path: str, plane: int | None = None) -> Response:
    """
    Return the data of a FITS file, or one plane of them (along their
    first axis), as a frame of one array (float32).
    """
    data = server.workspace.array(path, plane)
    return Response(
        frames.array_frame(dict(path=path, plane=plane), [({}, data)]),
        media_type='application/octet-stream')


@router.get('/jobs')
async def jobs(server: ServerDep) -> list[JobInfo]:
    """List the jobs, oldest first."""
    return [JobInfo.model_validate(job.info()) for job in server.jobs.list()]


@router.post('/jobs', status_code=status.HTTP_201_CREATED)
async def submit_job(server: ServerDep, request: JobRequest) -> JobInfo:
    """Queue a job, which starts when there is room."""
    job = server.jobs.submit(request.task, request.options)
    return JobInfo.model_validate(job.info())


@router.get('/jobs/{id_}')
async def job(server: ServerDep, id_: str) -> JobInfo:
    """Return a job."""
    return JobInfo.model_validate(server.jobs.get(id_).info())


@router.post('/jobs/{id_}/stop')
async def stop_job(server: ServerDep, id_: str) -> JobInfo:
    """
    Stop a job: cancel it if queued, or interrupt it if running, as
    Ctrl-C does (e.g. a fit ends with its best so far).
    """
    return JobInfo.model_validate(server.jobs.stop(id_).info())


@router.get('/events', response_class=EventSourceResponse)
async def events(server: ServerDep) -> AsyncIterator[ServerSentEvent]:
    """
    Follow the events of the server: 'job' (a job whose state changed),
    'run' (the run of a job, when its manifest changed) and 'progress'
    (a report of the progress of a fit).
    """
    async for event in server.events.subscribe():
        yield ServerSentEvent(event=event.kind, data=event.data)


@router.post('/sessions', status_code=status.HTTP_201_CREATED)
async def open_session(
        server: ServerDep, request: SessionRequest) -> SessionInfo:
    """
    Open a live session of a configuration: set it up once, to evaluate
    it at the values of its free parameters (see evaluate).
    """
    session = await server.sessions.open(request.config)
    return SessionInfo.model_validate(session.info())


@router.get('/sessions/{id_}')
async def session(server: ServerDep, id_: str) -> SessionInfo:
    """Return a live session."""
    return SessionInfo.model_validate(server.sessions.get(id_).info())


@router.delete('/sessions/{id_}', status_code=status.HTTP_204_NO_CONTENT)
async def close_session(server: ServerDep, id_: str) -> None:
    """Close a live session."""
    await server.sessions.close(id_)


@router.websocket('/sessions/{id_}/evaluate')
async def evaluate(websocket: WebSocket, server: ServerDep, id_: str) -> None:
    """
    Evaluate a live session: the client gets the frame of the data first
    (see Session.data), then sends values of free parameters ({values:
    {element: value}}, e.g. as it moves a slider), and gets the frame of
    each result (see Session.evaluate). Values sent while others are
    evaluated replace them (see Latest).
    """
    try:
        live = server.sessions.get(id_)
    except NotFound as e:
        raise WebSocketException(status.WS_1008_POLICY_VIOLATION, str(e))
    await websocket.accept()
    await websocket.send_bytes(live.data())
    latest = Latest()

    async def receive() -> None:
        async for message in websocket.iter_json():
            if not isinstance(message, dict) \
                    or not isinstance(message.get('values'), dict):
                await websocket.close(
                    status.WS_1003_UNSUPPORTED_DATA,
                    "a message is {values: {element: value}}")
                return
            latest.put(message['values'])

    receiving = asyncio.create_task(receive())
    try:
        while True:
            getting = asyncio.create_task(latest.get())
            await asyncio.wait(
                (receiving, getting), return_when=asyncio.FIRST_COMPLETED)
            if not getting.done():
                getting.cancel()
                break
            await websocket.send_bytes(await live.evaluate(getting.result()))
    except Failed as e:
        await websocket.close(status.WS_1011_INTERNAL_ERROR, str(e)[-120:])
    finally:
        receiving.cancel()
