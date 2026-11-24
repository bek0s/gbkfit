"""
The models of the requests and responses of the API (their schemas are
in its contract, openapi.json).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from gbkfit_server.services.jobs import TASKS


__all__ = [
    'ArrayInfo',
    'ArrayProblem',
    'CheckRequest',
    'CheckResult',
    'ComponentPartInfo',
    'ComponentType',
    'Diagnostic',
    'FileEntry',
    'FolderInfo',
    'GalaxyExtraRequest',
    'GalaxyInfo',
    'GalaxyRenderRequest',
    'GalaxyRequest',
    'GalaxyVolumeRequest',
    'HeaderInfo',
    'ImportRequest',
    'JobInfo',
    'JobRequest',
    'LSFRequest',
    'LSFViewInfo',
    'ObservedArrayInfo',
    'ObservedInfo',
    'ObservedRequest',
    'PSFRequest',
    'PSFViewInfo',
    'PieceLink',
    'PieceReference',
    'PieceRename',
    'PieceRequest',
    'PreferencesInfo',
    'ProjectInfo',
    'ProjectRename',
    'ProjectRequest',
    'RenderInfo',
    'RunInfo',
    'ServerInfo',
    'SessionInfo',
    'SessionRequest',
    'SetupInfo',
    'SetupObservedRequest',
    'SetupRequest',
    'SpectrumView'
]


class ServerInfo(BaseModel):
    """
    The server: its name and version, the path of its API, and the names
    of the pages it serves (at /<name>/).
    """
    name: str
    version: str
    api: str
    pages: list[str]


class PieceReference(BaseModel):
    """A piece (or a setup) of a project: its folder and name."""
    folder: str
    name: str


class PieceLink(BaseModel):
    """
    A piece (or a setup) that names a piece (source), the piece it names
    (target), and whether that piece is missing (named, but not in the
    project: e.g. deleted).
    """
    source: PieceReference
    target: PieceReference
    missing: bool


class ProjectInfo(BaseModel):
    """
    A project: its path, the names of its pieces and setups by folder
    (e.g. 'galaxies'), the paths of its runs, and its links: each piece or
    setup that names a piece, with the piece it names (see PieceLink).
    Also whether it has notes (its README.md), the type of each piece that
    has one (its option 'type'), by folder and name, when the file of each
    piece last changed (ISO 8601, UTC), by folder and name, and the status
    of the latest run of each setup that has run ('running', 'complete' or
    'failed'), by name.
    """
    path: str
    pieces: dict[str, list[str]]
    runs: list[str]
    links: list[PieceLink]
    notes: bool = False
    types: dict[str, dict[str, str]] = {}
    modified: dict[str, dict[str, str]] = {}
    latest: dict[str, str] = {}


class ProjectRename(BaseModel):
    """The new folder of a project, in the workspace."""
    path: str


class PieceRename(BaseModel):
    """The new name of a piece (a path in its folder, without .yaml)."""
    name: str


class GalaxyRequest(BaseModel):
    """
    A galaxy to describe: the project of its piece, and its options (its
    files relative to the project).
    """
    project: str
    galaxy: dict[str, Any]


class ComponentPartInfo(BaseModel):
    """
    A part of a type of component (see gbkfit's ComponentPart): its option,
    and the option of the height traits paired with its traits, if any.
    """
    option: str
    heights: str | None


class ComponentType(BaseModel):
    """
    A type of component: its parts, in the order people read them (see
    gbkfit's Component.parts; its other options are its settings), and the
    traits it takes (see TraitTable), if it takes traits: the types of
    its traits of each kind (e.g. 'vpt': ['uniform', 'arctan', ...]), and
    the directions of its velocity and dispersion traits.
    """
    parts: list[ComponentPartInfo]
    traits: dict[str, list[str]]
    directions: list[str]
    dispersion_directions: list[str]


class CatalogueLine(BaseModel):
    """
    An emission line of the catalogue of GBKFIT (see its LineCatalogue):
    its name, its rest as text (e.g. '6564.61 Angstrom', in vacuum), and
    as a number: its rest wavelength in vacuum (m) and in air (m; above
    2000 Angstrom, where air wavelengths are used), or its rest frequency
    (Hz), as the catalogue gives it.
    """
    name: str
    rest: str
    wavelength: float | None
    air: float | None
    frequency: float | None


class GalaxyInfo(BaseModel):
    """
    A galaxy as the library describes it (see Galaxies.describe): its
    parameters ({name, title, desc, size, default, unit, range, period,
    owner}, in its order), its problems ({severity, path, message};
    paths in the galaxy), and whether it was made as a whole (else its
    parameters are those of its parts made on their own).
    """
    parameters: list[dict[str, Any]]
    problems: list[dict[str, Any]]
    whole: bool


class LensRequest(BaseModel):
    """
    A lens to describe: the project of its piece, and its options (its
    files relative to the project).
    """
    project: str
    lens: dict[str, Any]


class LensInfo(BaseModel):
    """
    A lens as the library describes it (see Lenses.describe): its
    parameters, by the names of the lens (as those of GalaxyInfo; their
    owner, their piece: ['pieces', i]), and its problems.
    """
    parameters: list[dict[str, Any]]
    problems: list[dict[str, Any]]


class ObservationRequest(BaseModel):
    """
    An observation to describe, alone (without data): the project of its
    piece, and its options (the pieces it names, of the project; its files
    relative to it).
    """
    project: str
    observation: dict[str, Any]


class PartRequest(BaseModel):
    """
    A part that observations share to describe: the project of its piece,
    its family (a PSF, an LSF or a noise) and its options (its files
    relative to the project).
    """
    project: str
    family: Literal['psf', 'lsf', 'noise']
    part: dict[str, Any]


class SetupRequest(BaseModel):
    """
    A setup to describe: its path (of a project, relative to the
    workspace) and its options (e.g. as it is edited, not saved yet).
    """
    path: str
    setup: dict[str, Any]


class SetupInfo(BaseModel):
    """
    A setup as the library describes it (see Setups.describe): its
    parameters ({name, title, desc, size, default, unit, item, properties,
    source}, in the order of the library; item the name of the item they
    are of, '' for one galaxy without a name; properties those its layers
    give them, by key; source the file of the layer that gives them,
    relative to the project), and its problems ({severity, path,
    message}; paths in the setup).
    """
    parameters: list[dict[str, Any]]
    problems: list[dict[str, Any]]


class SetupObservedRequest(BaseModel):
    """
    A setup to observe at the starts of its parameters (see
    Renders.setup_observed): as SetupRequest, with the index of its
    observation that sees it.
    """
    path: str
    setup: dict[str, Any]
    observation: int = 0


class LensViewRequest(BaseModel):
    """
    A lens to view (see Lenses.view): as LensRequest, with the properties
    of its parameters (by the names of the lens), the number of points
    across the field, its half-width and its centre (x and y; arcsec; by
    default, fitted to the lens), and the pieces hidden in it (by index).
    """
    project: str
    lens: dict[str, Any]
    properties: dict[str, Any]
    size: int = Field(96, ge=16, le=256)
    half: float | None = Field(None, gt=0)
    centre: tuple[float, float] | None = None
    hidden: list[int] = []


class LensViewInfo(BaseModel):
    """
    What a lens makes of a source behind it (see Lenses.view and LensView):
    the half-width and the centre of the field (arcsec), the redshift of
    the source it is of (None: at D_ls / D_s = 1), the points across it,
    the deflection at each (arcsec, row-major, y up), the critical curves
    and the caustics (lists of [x, y]), and the pieces left out of it.
    """
    half: float
    centre: tuple[float, float]
    source_redshift: float | None
    size: int
    alpha_x: list[float]
    alpha_y: list[float]
    critical: list[list[list[float]]]
    caustics: list[list[list[float]]]
    left_out: list[str]


class GalaxyVolumeRequest(BaseModel):
    """
    A galaxy to render in 3d (see Renders.volume): as GalaxyRenderRequest,
    but always face-on.
    """
    project: str
    galaxy: dict[str, Any]
    properties: dict[str, Any] = {}
    size: int = 96
    label: str | None = None


class GalaxyExtraRequest(BaseModel):
    """
    An extra output of a galaxy to render (see Renders.extra): as
    GalaxyRenderRequest, with the name and the ids of the output (see the
    list of a render of the galaxy).
    """
    project: str
    galaxy: dict[str, Any]
    properties: dict[str, Any] = {}
    name: str
    ids: dict[str, str] = {}
    view: Literal['sky', 'face-on', 'edge-on'] = 'sky'
    size: int = 160
    label: str | None = None


class GalaxyRenderRequest(BaseModel):
    """
    A galaxy to render (see Renders.galaxy): the project of its piece, its
    options (its files relative to the project), the values of its
    parameters (numbers, or lists of them for vectors; the defaults for
    those not given), how it is seen ('sky', 'face-on' or 'edge-on'), and
    the pixels across; and what it is, for the pages that show the
    renders that run long (e.g. the name of its piece).
    """
    project: str
    galaxy: dict[str, Any]
    properties: dict[str, Any] = {}
    view: Literal['sky', 'face-on', 'edge-on'] = 'sky'
    size: int = 160
    label: str | None = None


class ObservedRequest(BaseModel):
    """
    A galaxy to observe (see Renders.observed): the project of its piece,
    its options and the values of its parameters (as a galaxy to render;
    the defaults for those not given, also of the observation), the
    observation of the project that sees it, by name, and the dataset it
    is paired with, if any (a piece of the project by name, or its
    options); and what the galaxy is, for the pages that show the renders
    that run long.
    """
    project: str
    galaxy: dict[str, Any]
    properties: dict[str, Any] = {}
    observation: str
    dataset: str | dict[str, Any] | None = None
    label: str | None = None


class RenderInfo(BaseModel):
    """
    A render that runs long (see RenderPool): its id, what it renders
    (e.g. "galaxy ngc2403"), its kind ('galaxy', 'cube' or 'observed'),
    the stage it has reached (e.g. 'evaluating') and the seconds since it
    started.
    """
    id: str
    what: str
    kind: str
    stage: str
    elapsed: float


class ObservedArrayInfo(BaseModel):
    """
    An array of a galaxy observed: its kind ('model', 'data' or 'error'),
    its item (e.g. 'spectra', 'brightness', 'moment1'), the path the views of
    cubes take it at (its array 0; a map, a cube of one channel; kept for a
    while, see Views.put), its shape in the order of the model (nx, ny,
    nchannels: 1 for a map), the unit of its values ('' if unknown), and
    its pixels on the sky: their size (x, y; arcsec) and the pixel (x, y;
    0-based) that the model measures positions from.
    """
    kind: str
    item: str
    path: str
    shape: list[int]
    unit: str
    step: list[float]
    rpix: list[float]


class ObservedInfo(BaseModel):
    """
    A galaxy as an observation of its project sees it: the type of the
    observation (e.g. 'pixel_spectra'), its arrays (the model of each item,
    and the data and errors of the data, if it has them), and the warnings
    of the library as it set them up.
    """
    type: str
    arrays: list[ObservedArrayInfo]
    notes: list[str]


class CubeRenderInfo(BaseModel):
    """
    The cube of a galaxy rendered (see Renders.cube): the path the views
    of cubes take it at (its array 0; kept for a while, see Views.put), its
    shape in the order of the model (nx, ny, nchannels), the unit of the
    coordinates of its channels (km/s), the size of its pixels (arcsec),
    and how the velocities of the galaxy land on its channels (see
    CubeAxis): the convention of their velocities ('optical', 'radio' or
    'relativistic') and the 1 + z of its lines (shift).
    """
    path: str
    shape: list[int]
    unit: str
    step: float
    convention: Literal['optical', 'radio', 'relativistic']
    shift: float


class PSFRequest(BaseModel):
    """
    A PSF to draw: the project of its piece, its options (its files
    relative to the project), and the size of the pixels (x, y; arcsec;
    none: of its own scale).
    """
    project: str
    options: dict[str, Any]
    step: tuple[float, float] | None = None


class LSFRequest(BaseModel):
    """
    An LSF to draw: the project of its piece, its options (its files
    relative to the project), the width of the channels (none: of its own
    scale), the rest of the line it is drawn at (e.g. '6562.8 Angstrom';
    widths in a unit of wavelength or frequency need it), the unit of the
    channels: km/s (velocities) by default, or another, which draws it as
    its widths give it (e.g. 'Angstrom', or 'pix': channels of the data;
    without a rest), and the width of the channels of the data, which its
    widths in pixels and its Hanning smoothing are of (none: those it is
    drawn on).
    """
    project: str
    options: dict[str, Any]
    step: float | None = None
    rest: str | None = None
    unit: str | None = None
    channel: float | None = None


class PSFViewInfo(BaseModel):
    """
    A PSF drawn on pixels (see Responses.psf): its values (rows of y,
    from the bottom; they sum to 1), the size of the pixels (x, y;
    arcsec), the solid angle of its beam (arcsec^2) and the full width at
    half maximum of the circular Gaussian of that beam (arcsec; none for
    a point), and its full widths at half maximum along its major and
    minor axes (arcsec) and the position angle of its major axis
    (degrees; none if round), as measured (none for a point).
    """
    values: list[list[float]]
    step: tuple[float, float]
    beam_area: float | None
    fwhm: float | None
    axes: tuple[float, float, float | None] | None


class LSFViewInfo(BaseModel):
    """
    An LSF drawn on channels (see Responses.lsf): its values (they sum
    to 1), the width of the channels, its full width at half maximum (none
    for a point), and the unit of both (e.g. 'km/s').
    """
    values: list[float]
    step: float
    fwhm: float | None
    unit: str


class FileEntry(BaseModel):
    """A file of a folder: its name, and its size (bytes)."""
    name: str
    size: int


class FolderInfo(BaseModel):
    """
    A folder: its path, and its folders and files (but the hidden ones),
    by name.
    """
    path: str
    folders: list[str]
    files: list[FileEntry]


class Problem(BaseModel):
    """
    A problem of the options of a piece (see gbkfit's ConfigProblem): an
    error or a warning, the path of the option that fixes it ([] if none)
    and its message.
    """
    severity: Literal['error', 'warning']
    path: list[str | int]
    message: str


class ArrayProblem(Problem):
    """
    A problem of an array of a file (see gbkfit's describe_fits): an error
    (a dataset of it refuses it) or a warning (it is read as FITS says,
    but may not be meant), the option of a dataset of it that fixes it
    (e.g. ['spectral', 'unit']; [] if none) and its message (e.g. "CUNIT3
    'KMS' invalid").
    """


class DatasetRequest(BaseModel):
    """
    A dataset to check: the project of its piece, and its options (its
    files relative to the project).
    """
    project: str
    dataset: dict[str, Any]


class ArrayInfo(BaseModel):
    """
    An array of a file of data (e.g. an HDU of a FITS file), from its
    header: its index and name in the file, its shape and the types and
    units of its axes as written (CTYPE and CUNIT; '' none), in the
    order of the model (x, y, then the spectral axis),
    the unit of its values, the size of its pixels in that order (arcsec,
    and km/s along a velocity axis; the spatial ones alone while its
    spectral axis cannot be read, e.g. velocities of no convention or
    wavelengths without a rest; none if its header has no coordinates on
    the sky), the pixel (x, y; 0-based) that the model measures positions
    on the sky from (its reference pixel; none without coordinates), its
    beam (bmaj and bmin in arcsec, bpa in degrees; lists
    for a beam of each channel), the noise of its values that its header
    states (keyword RMS or NOISE), whether its header leaves it open
    whether its velocities are radio or optical, its problems, each at
    the option of a dataset of it that fixes it (see ArrayProblem), what
    its values are as errors, if its header states it ('sigma',
    'variance' or 'inverse_variance'; see gbkfit's
    HDUDescription.error_kind), and the rest of its spectral axis (e.g.
    '6562.8 Angstrom': its header's, RESTWAV or RESTFRQ, or that given;
    none if none, or while the axis cannot be read).
    """
    index: int
    name: str
    shape: list[int]
    axes: list[str]
    axis_units: list[str]
    unit: str | None
    step: list[float] | None
    rpix: list[float] | None = None
    beam: dict[str, float | list[float]] | None
    rms: float | None
    ambiguous_velocity: bool
    problems: list[ArrayProblem]
    holds: Literal['sigma', 'variance', 'inverse_variance'] | None = None
    rest: str | None = None


class SpectrumView(BaseModel):
    """
    The spectrum of a pixel of a cube: its values (none where it has
    none), and the coordinates of its channels along its spectral axis
    and their unit (km/s for velocities, GHz for frequencies, micron for
    wavelengths; none if its header does not give them).
    """
    values: list[float | None]
    spectral: list[float] | None
    unit: str | None


class RunInfo(BaseModel):
    """A run: its manifest (see gbkfit.outputs)."""
    path: str
    task: str
    options: dict[str, Any]
    status: str
    gbkfit_version: str
    created: str
    updated: str
    products: list[dict[str, Any]]


class JobRequest(BaseModel):
    """
    A job to run: its task, and the options of the task (see
    gbkfit.tasks), with their paths relative to the workspace. eval and
    fit need a configuration ('config'), and write their run in the runs
    of its setup, unless they give its folder ('output_dir'); resume
    needs the folder of the run of the fit ('output_dir'); prep needs the
    folder of the prepared files ('output_dir'), where they are kept as
    imported files are (an identical file is reused; a name taken is
    numbered), and how they were prepared is recorded (see
    PreparationInfo). A job of prep can be a preview: its files stay
    apart, to be looked at, until kept (POST /jobs/{id}/keep); previews
    not kept are removed in time.
    """
    task: Literal[TASKS]
    options: dict[str, Any] = {}
    preview: bool = False


class JobInfo(BaseModel):
    """
    A job: its id, task and options, whether it is a preview of prep not
    kept, its state, its run, the files that a job of prep prepared (by
    the options of their files, e.g. file_d; of moment maps, a list, one
    for each map, None for a file not prepared; none for a preview
    removed), when it was created, started and ended, the exit status
    of its process, and the end of its log.
    """
    id: str
    task: str
    options: dict[str, Any]
    created: str
    preview: bool
    state: Literal[
        'queued', 'running', 'complete', 'failed', 'stopped', 'cancelled']
    run: str | None
    files: dict[str, str | list[str | None]] | None
    started: str | None
    ended: str | None
    status: int | None
    output: str


class PreparationInfo(BaseModel):
    """
    How a file of data was prepared by a job of prep: the task, its
    options (their files: a path, or {file, hdu}; of moment maps, lists),
    the files it prepared (by the options of their files, as those of a
    job), with their paths relative to the workspace, and when (ISO 8601,
    UTC).
    """
    task: str
    options: dict[str, Any]
    files: dict[str, str | list[str | None]]
    time: str


class PreferencesInfo(BaseModel):
    """
    The preferences of the user (of the pages, e.g. the theme of the
    Studio): their values by name, and the file they are kept in, in
    the folder of the user.
    """
    file: str
    values: dict[str, Any]


class HeaderInfo(BaseModel):
    """
    The header of an array (an HDU) of a file of data, as it is in the
    file: its cards, as lines of text.
    """
    cards: list[str]


class SessionRequest(BaseModel):
    """A live session to open: its configuration (its path)."""
    config: str


class SessionInfo(BaseModel):
    """
    A live session: its id, its configuration, its free parameters (by
    element name), with their properties (e.g. their values, min and
    max), and whether it has data (and so residuals and a chi-square).
    """
    id: str
    config: str
    free: dict[str, dict[str, Any]]
    data: bool


class CheckRequest(BaseModel):
    """The text of a file to check (e.g. not saved yet), and its path."""
    path: str
    text: str


class Diagnostic(BaseModel):
    """A problem of a text: its line (from 1), severity and message."""
    line: int
    severity: Literal['error', 'warning']
    message: str


class CheckResult(BaseModel):
    """The problems of a text (see the checker of the server)."""
    diagnostics: list[Diagnostic]


class ProjectRequest(BaseModel):
    """A project to create: its folder (its path)."""
    path: str


class ImportRequest(BaseModel):
    """
    A file of data of the workspace to copy into a project: the project,
    and the path of the file.
    """
    project: str
    path: str


class PieceContent(BaseModel):
    """The content of a piece (its options; e.g. those of a dataset)."""
    content: dict[str, Any]


class PieceRequest(BaseModel):
    """
    A piece (or a setup) to create: its project, its folder (e.g.
    galaxies, psfs or setups), its name (its path in the folder, without
    .yaml), and its content (its options); by default, that of the
    template of its folder.
    """
    project: str
    folder: str
    name: str
    content: dict[str, Any] | None = None
