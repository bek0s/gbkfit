"""
The routes of the API (version 1): thin, over the services of the server.
Paths are relative to its workspace. Arrays and the results of sessions
are frames (see gbkfit_server.services.frames). The routes of the files
run in threads (they block); those of the jobs, events and sessions in
the event loop of their services.
"""

import asyncio
import functools
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import (
    APIRouter, Body, Depends, Request, Response, WebSocket,
    WebSocketException, status)
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from starlette.requests import HTTPConnection

import gbkfit
from gbkfit.grid import SPECTRAL_TYPES, air_refractive_index

from gbkfit_server.services import frames
from gbkfit_server.services.errors import Failed, Invalid, NotFound
from gbkfit_server.services.server import Server
from gbkfit_server.services.sessions import Latest
from gbkfit_server.services.units import rest_text
from gbkfit_server.services.views import Cube
from .models import (
    ArrayInfo, CatalogueLine, CheckRequest, CheckResult, ComponentTraits,
    CubeRenderInfo,
    FolderInfo,
    GalaxyInfo, HeaderInfo,
    GalaxyRenderRequest, GalaxyRequest, ImportRequest,
    JobInfo, JobRequest, LSFRequest, LSFViewInfo, ObservedArrayInfo,
    ObservedInfo, ObservedRequest, PSFRequest, PSFViewInfo,
    PieceReference, PieceRename, PieceRequest, PreferencesInfo,
    PreparationInfo,
    ProjectInfo, ProjectRename, ProjectRequest, RunInfo,
    SessionInfo, SessionRequest, SpectrumView)


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


@router.post('/projects', status_code=status.HTTP_201_CREATED)
def create_project(server: ServerDep, request: ProjectRequest) -> ProjectInfo:
    """Create a project, with the folders of its pieces."""
    return ProjectInfo.model_validate(
        server.workspace.create_project(request.path), from_attributes=True)


@router.post('/pieces', status_code=status.HTTP_201_CREATED)
def create_piece(server: ServerDep, request: PieceRequest) -> str:
    """
    Create a piece (or a setup) of a project, with its content or from
    its template; return its path.
    """
    return server.workspace.create_piece(
        request.project, request.folder, request.name, request.content)


@router.get('/pieces/{path:path}')
def piece(server: ServerDep, path: str) -> dict[str, Any]:
    """Return a piece (or a setup) of a project: its options."""
    return server.workspace.piece(path)


@router.get('/projects/{path:path}')
def project(server: ServerDep, path: str) -> ProjectInfo:
    """Return a project: its pieces, setups and runs."""
    return ProjectInfo.model_validate(
        server.workspace.project(path), from_attributes=True)


@router.patch('/projects/{path:path}')
def rename_project(
        server: ServerDep, path: str, request: ProjectRename) -> ProjectInfo:
    """
    Move a project to another folder of the workspace; not while jobs of
    it are queued or running.
    """
    _idle(server, path)
    return ProjectInfo.model_validate(
        server.workspace.rename_project(path, request.path),
        from_attributes=True)


@router.delete('/projects/{path:path}', status_code=status.HTTP_204_NO_CONTENT)
def delete_project(server: ServerDep, path: str) -> None:
    """
    Delete a project and all its files (its data and runs too); not while
    jobs of it are queued or running.
    """
    _idle(server, path)
    server.workspace.delete_project(path)


def _idle(server: Server, path: str) -> None:
    """Raise Invalid if jobs of a folder are queued or running."""
    if busy := server.jobs.busy(path):
        raise Invalid(
            f"'{path}' has {len(busy)} job(s) queued or running; stop them "
            f"first")


@router.patch('/pieces/{path:path}')
def rename_piece(server: ServerDep, path: str, request: PieceRename) -> str:
    """
    Rename a piece (or a setup) of a project, and the pieces and setups
    that name it with it; return its new path.
    """
    return server.workspace.rename_piece(path, request.name)


@router.put('/pieces/{path:path}', status_code=status.HTTP_204_NO_CONTENT)
def update_piece(
        server: ServerDep,
        path: str,
        content: Annotated[dict[str, Any], Body()]
) -> None:
    """
    Write the content of a piece (or a setup) of a project: its options
    that change (its comments stay).
    """
    server.workspace.update_piece(path, content)


@router.delete('/pieces/{path:path}', status_code=status.HTTP_204_NO_CONTENT)
def delete_piece(server: ServerDep, path: str) -> None:
    """
    Delete a piece (or a setup) of a project; the pieces and setups that
    name it still do (see references), and its files of data stay.
    """
    server.workspace.delete_piece(path)


@router.get('/references/{path:path}')
def references(server: ServerDep, path: str) -> list[PieceReference]:
    """Return the pieces and setups of a project that name one of its pieces."""
    return [
        PieceReference.model_validate(reference)
        for reference in server.workspace.references(path)]


@router.get('/folders/{path:path}')
def folder(server: ServerDep, path: str) -> FolderInfo:
    """Return a folder: its folders and its files ('.': the workspace)."""
    return FolderInfo.model_validate(
        server.workspace.folder(path), from_attributes=True)


@router.get('/describe/{path:path}')
def describe(
        server: ServerDep,
        path: str,
        velocity: Literal['radio', 'optical'] | None = None,
        rest: str | None = None,
        spectral_type: Literal[SPECTRAL_TYPES] | None = None
) -> list[ArrayInfo]:
    """
    Describe a file of data from its headers, without reading its data:
    each of its arrays, and what a dataset of it refuses or needs given;
    velocity is the convention of velocities that its headers leave
    ambiguous, rest the rest of its spectral axis (e.g. '6562.8
    Angstrom'), which makes wavelengths or frequencies velocities, as a
    dataset with that rest reads them, and spectral_type what a spectral
    axis whose header gives a type FITS does not define is (e.g. 'AWAV';
    see ArrayInfo).
    """
    return [
        ArrayInfo.model_validate(array)
        for array in server.workspace.describe(
            path, velocity, rest, spectral_type)]


@router.get('/header/{path:path}')
def header(server: ServerDep, path: str, hdu: int = 0) -> HeaderInfo:
    """
    Return the header of an array (an HDU) of a file of data, as it is in
    the file (one header per HDU: the data and the errors of one file have
    each their own).
    """
    return HeaderInfo(cards=server.workspace.header(path, hdu))


# (the responses of frames of arrays)
_FRAME = {200: {'content': {'application/octet-stream': {}}}}


@router.get(
    '/views/channel/{path:path}', response_class=Response, responses=_FRAME)
def view_channel(
        server: ServerDep, path: str, hdu: int, channel: int) -> Response:
    """
    Return a channel of a cube (an array of a file of data, in the
    orientation of the model) as a frame of one array (ny, nx), with the
    number of channels of the cube, the coordinate of the channel along
    the spectral axis and its unit (null if unknown; see SpectrumView),
    the range of the values of the cube for colours, and the least and
    the greatest of its values (extent; null if it has none).
    """
    cube = server.views.cube(path, hdu)
    plane = server.views.channel(path, hdu, channel)
    spectral = None if cube.spectral is None \
        else float(cube.spectral[channel])
    header = dict(
        path=path, hdu=hdu, channel=channel,
        nchannels=cube.values.shape[0], spectral=spectral, unit=cube.unit,
        range=list(cube.range),
        extent=None if cube.extent is None else list(cube.extent))
    return Response(
        frames.array_frame(header, [({}, plane)]),
        media_type='application/octet-stream')


@router.get('/views/spectrum/{path:path}')
def view_spectrum(
        server: ServerDep, path: str, hdu: int, x: int | None = None,
        y: int | None = None, x0: int | None = None, x1: int | None = None,
        y0: int | None = None, y1: int | None = None
) -> SpectrumView:
    """
    Return the spectrum of a pixel (x, y) of a cube, or its total spectrum
    (the sum of its pixels in each channel): of all of them, or of a region
    (x0 to x1, y0 to y1, inclusive).
    """
    region = None if None in (x0, x1, y0, y1) else (x0, x1, y0, y1)
    values = server.views.spectrum(path, hdu, x, y, region)
    cube = server.views.cube(path, hdu)
    return SpectrumView(
        values=[float(v) if np.isfinite(v) else None for v in values],
        spectral=None if cube.spectral is None else cube.spectral.tolist(),
        unit=cube.unit)


@router.get(
    '/views/moments/{path:path}', response_class=Response, responses=_FRAME)
def view_moments(
        server: ServerDep, path: str, hdu: int, snr: float = 5,
        method: Literal['moments', 'gauss_hermite'] = 'moments'
) -> Response:
    """
    Return the quick moment maps of a cube (NaN where they are not
    defined) as a frame of arrays (ny, nx), named so, of the voxels of
    the mask of its signal grown from seeds of signal-to-noise ratio snr
    (0: of all of them; see Views.moments): measured as moments, its
    flux, velocity, dispersion, skewness and kurtosis; or of a
    Gauss-Hermite series fitted to each spectrum, its h3 and h4.
    """
    maps = server.views.moments(path, hdu, snr, method)
    arrays = [(dict(name=name), data) for name, data in maps.items()]
    return Response(
        frames.array_frame(dict(path=path, hdu=hdu), arrays),
        media_type='application/octet-stream')


@router.post('/views/psf')
def view_psf(server: ServerDep, request: PSFRequest) -> PSFViewInfo:
    """
    Draw a PSF of the options of a piece of a project, on pixels of a
    size or of its own scale, with the solid angle of its beam and its
    width.
    """
    view = server.responses.psf(request.project, request.options, request.step)
    return PSFViewInfo(
        values=view.values.tolist(), step=view.step,
        beam_area=view.beam_area, fwhm=view.fwhm, axes=view.axes)


@router.post('/views/lsf')
def view_lsf(server: ServerDep, request: LSFRequest) -> LSFViewInfo:
    """
    Draw an LSF of the options of a piece of a project, on channels of a
    width or of its own scale, of velocities (at the rest of a line, if
    given) or of a unit given, with its width.
    """
    view = server.responses.lsf(
        request.project, request.options, request.step, request.rest,
        request.unit, request.channel)
    return LSFViewInfo(
        values=view.values.tolist(), step=view.step, fwhm=view.fwhm,
        unit=view.unit)


@router.post('/parameters')
def galaxy_parameters(server: ServerDep, request: GalaxyRequest) -> GalaxyInfo:
    """
    Describe a galaxy of the options of a piece of a project, as the
    library does without devices: the parameters it has, and its problems.
    """
    return GalaxyInfo.model_validate(
        server.galaxies.describe(request.project, request.galaxy),
        from_attributes=True)


@router.post('/views/galaxy', response_class=Response, responses=_FRAME)
def view_galaxy(server: ServerDep, request: GalaxyRenderRequest) -> Response:
    """
    Render a galaxy of the options of a piece of a project at values of
    its parameters, through an ideal observation (see Renders.galaxy): a
    frame of its maps (ny, nx; named 'light', 'velocity', 'vx', 'vy',
    'dispersion' and '{component}/{map}'; NaN where there is no light),
    whose header has the size of the pixels (step, arcsec), the pixels
    across (size), its curves (by name, with 'radius'; null where not
    defined) and the problems of those that could not be drawn.
    """
    render = server.renders.galaxy(
        request.project, request.galaxy, request.properties, request.view,
        request.size)
    header = dict(
        step=render.step, size=render.size, curves=render.curves,
        problems=render.problems)
    arrays = [(dict(name=name), data) for name, data in render.maps.items()]
    return Response(
        frames.array_frame(header, arrays),
        media_type='application/octet-stream')


@router.post('/views/galaxy-cube')
def view_galaxy_cube(
        server: ServerDep, request: GalaxyRenderRequest) -> CubeRenderInfo:
    """
    Render the cube of a galaxy (see Renders.cube), of its components that
    emit lines, and keep it for the views of cubes (channel, spectrum and
    moments) at the path returned.
    """
    cube = server.renders.cube(
        request.project, request.galaxy, request.properties, request.view,
        request.size)
    path = server.views.put(cube)
    nchannels, ny, nx = cube.values.shape
    return CubeRenderInfo(
        path=path, shape=[nx, ny, nchannels], unit=cube.unit or '')


@router.post('/views/observed')
def view_observed(server: ServerDep, request: ObservedRequest) -> ObservedInfo:
    """
    Evaluate a galaxy through an observation of its project (see
    Renders.observed), and keep its arrays (the model of each item, and
    the data and errors of the observation, if it has them) for the views
    of cubes (channel, spectrum and moments), each at the path returned.
    """
    observed = server.renders.observed(
        request.project, request.galaxy, request.properties,
        request.observation)
    arrays = []
    for array in observed.arrays:
        values = array.values if array.values.ndim == 3 \
            else array.values[np.newaxis]
        spectral = observed.velocities if array.values.ndim == 3 else None
        cube = Cube.of(
            values, spectral, 'km/s' if spectral is not None else None)
        nchannels, ny, nx = values.shape
        arrays.append(ObservedArrayInfo(
            kind=array.kind, item=array.item, path=server.views.put(cube),
            shape=[nx, ny, nchannels], unit=array.unit,
            step=list(observed.step), rpix=list(observed.rpix)))
    return ObservedInfo(
        observable=observed.observable, arrays=arrays, notes=observed.notes)


@router.get('/schema')
def config_schema() -> dict[str, Any]:
    """
    Return the JSON Schema of the configurations of GBKFIT, from the
    options of the library (see gbkfit.schema): the forms of the pieces
    are made of it.
    """
    return _schema()


@router.get('/lines')
def catalogue_lines() -> list[CatalogueLine]:
    """
    Return the emission lines of the catalogue of GBKFIT, which emissions
    name and a spectral axis of wavelengths or frequencies takes the rest
    of (in vacuum or in air, as its type says).
    """
    catalogue = gbkfit.model.LineCatalogue()
    return [_catalogue_line(catalogue.line(name)) for name in catalogue.names()]


def _catalogue_line(line: Any) -> CatalogueLine:
    """Return a line of the catalogue (see CatalogueLine)."""
    rest = line.rest()
    wavelength = air = frequency = None
    if rest.unit.physical_type == 'length':
        wavelength = rest.to_value('m')
        if wavelength >= 2e-7:
            air = wavelength / air_refractive_index(wavelength)
    else:
        frequency = rest.to_value('Hz')
    return CatalogueLine(
        name=line.name(), rest=rest_text(rest), wavelength=wavelength,
        air=air, frequency=frequency)


@router.get('/components')
def component_traits() -> dict[str, ComponentTraits]:
    """
    Return the traits that each type of component takes, by its type (see
    gbkfit.model.components.TRAIT_TABLES): the types of its traits of each
    kind, and the directions of its velocity and dispersion traits. (The
    schema has every type of each kind, as the library reads them; a
    component takes some of them.)
    """
    return {
        component: ComponentTraits(
            traits={
                kind: [cls.type() for cls in classes]
                for kind, classes in table.classes.items()},
            directions=list(table.directions),
            dispersion_directions=list(table.dispersion_directions))
        for component, table in
        gbkfit.model.components.TRAIT_TABLES.items()}


@functools.cache
def _schema() -> dict[str, Any]:
    """Return the JSON Schema of the configurations (once)."""
    return gbkfit.schema()


@router.post('/imports', status_code=status.HTTP_201_CREATED)
def import_data(server: ServerDep, request: ImportRequest) -> str:
    """
    Copy a file of data of the workspace into a project; return the path
    of the copy (that of the same file if it has one).
    """
    return server.workspace.import_data(request.project, request.path)


@router.post(
    '/uploads', status_code=status.HTTP_201_CREATED,
    openapi_extra={'requestBody': {
        'required': True,
        'content': {'application/octet-stream': {}}}})
async def upload_data(
        server: ServerDep, request: Request, project: str, name: str
) -> str:
    """
    Write a file of data sent in the body (e.g. from the computer of the
    page) into a project, by its name; return its path (that of the same
    file if it has one).
    """
    return await server.workspace.receive_data(
        project, name, request.stream())


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
    job = server.jobs.submit(request.task, request.options, request.preview)
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


@router.post('/jobs/{id_}/keep')
async def keep_job(server: ServerDep, id_: str) -> JobInfo:
    """
    Keep the files of a preview of prep into the folder of its output_dir
    (see JobRequest); the job then gives the files kept.
    """
    return JobInfo.model_validate(server.jobs.keep(id_).info())


@router.get('/preparations/{path:path}')
def preparation(server: ServerDep, path: str) -> PreparationInfo | None:
    """
    Return how a file of data was prepared by a job of prep (see
    PreparationInfo); null if it was not (e.g. a file imported as it is).
    """
    info = server.workspace.preparation(path)
    return None if info is None else PreparationInfo.model_validate(info)


@router.delete('/data/{path:path}', status_code=status.HTTP_204_NO_CONTENT)
def delete_data(server: ServerDep, path: str) -> None:
    """
    Delete a file of data of a project (of its folder data), unless the
    project names it (a piece, a setup or a run) or a job uses it.
    """
    if server.jobs.busy(path):
        raise Invalid(f"'{path}' is in use by a job")
    server.workspace.delete_data(path)


@router.get('/preferences')
def preferences(server: ServerDep) -> PreferencesInfo:
    """
    Return the preferences of the user (of the pages, e.g. the theme of
    the Studio), and their file, in the folder of the user (~/.gbkfit),
    the same for every folder served.
    """
    return PreferencesInfo(
        file=str(server.preferences.file), values=server.preferences.read())


@router.patch('/preferences')
async def update_preferences(
        server: ServerDep, changes: Annotated[dict[str, Any], Body()]
) -> PreferencesInfo:
    """
    Set preferences of the user (a value of null removes one), keeping
    the others; the clients are told (the event 'preferences').
    """
    # (in the event loop, so that the changes and their events are in the
    # order they came; the file is small)
    values = server.preferences.update(changes)
    server.events.publish('preferences', dict(values=values))
    return PreferencesInfo(file=str(server.preferences.file), values=values)


@router.get('/events', response_class=EventSourceResponse)
async def events(server: ServerDep) -> AsyncIterator[ServerSentEvent]:
    """
    Follow the events of the server: 'job' (a job whose state changed),
    'run' (the run of a job, when its manifest changed), 'progress' (a
    report of the progress of a fit) and 'preferences' (the preferences
    of the user, all of them, as they changed).
    """
    # (a comment first, which clients pass over: the response starts with
    # it, so that they know they are connected before any event)
    yield ServerSentEvent(comment='connected')
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
