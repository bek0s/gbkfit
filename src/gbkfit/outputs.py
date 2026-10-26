"""
The outputs of GBKFIT: each run of a task writes a directory of standard
files (FITS, ECSV, JSON and YAML) and a manifest, gbkfit.json, which lists
every product with its kind (e.g. 'model', 'residual', 'samples') and the
ids of what it is of (e.g. the observation and the data item). Readers
look products up by kind and ids, never by file name (see open); tasks
write them with a RunWriter (see create and reopen). The outputs of an
evaluation (the models and residuals of the observations, and their
data; see evaluation_outputs and data_outputs) are written as products
by write_outputs.

The manifest is rewritten (atomically, as every file) after each product,
so a reader always sees a consistent run, also while it is running (its
status is then 'running').
"""

import copy
import dataclasses
import datetime
import json
import math
import os
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import astropy.io.fits as fits
import numpy as np
import ruamel.yaml
from astropy.table import Table

import gbkfit
from gbkfit.utils import fitsutils, gridutils, miscutils, numutils

if TYPE_CHECKING:
    import astropy.units
    from gbkfit.driver import Driver
    from gbkfit.model import Extra
    from gbkfit.objective import Objective
    from gbkfit.observation import GroupPlan, Observation
    from gbkfit.params import ParamValues


__all__ = [
    'FORMAT',
    'FORMAT_VERSION',
    'MANIFEST',
    'Output',
    'Product',
    'Run',
    'RunWriter',
    'create',
    'data_outputs',
    'evaluation_outputs',
    'make_directory',
    'open',
    'reopen',
    'write_outputs'
]


# The name and the version of the format of the manifest: additive changes
# (new kinds, ids or fields) keep the version, so that readers of older
# versions still read newer runs
FORMAT = 'gbkfit-run'
FORMAT_VERSION = 1

# The file name of the manifest, in the directory of a run
MANIFEST = 'gbkfit.json'

Status = Literal['running', 'complete', 'failed']

# The formats of the products, by file extension
_FORMATS = {
    '.fits': 'fits', '.ecsv': 'ecsv', '.json': 'json', '.yaml': 'yaml',
    '.png': 'png', '.pdf': 'pdf', '.svg': 'svg'}

# The formats of figures, written as the bytes of their files
_FIGURES = ('png', 'pdf', 'svg')

# The values that JSON and YAML products hold (with lists and dicts of
# them)
_PLAIN = (type(None), bool, int, float, str, list, tuple, dict, np.generic)


@dataclasses.dataclass(frozen=True)
class Product:
    """
    A product of a run, as its manifest lists it.

    Attributes
    ----------
    path : str
        Its file, relative to the directory of the run ('/'-separated).
    kind : str
        What it is (e.g. 'model', 'residual', 'samples', 'figure').
    ids : dict of str to str
        What it is of (e.g. observation, galaxy, component, item).
    format : str
        The format of its file ('fits', 'ecsv', 'json', 'yaml', or that of
        a file written by other code, e.g. 'log').
    unit : str or None
        The unit of its values, if it has one (the columns of a table
        carry their own).
    shape : tuple of int or None
        The shape of its array (numpy order), or (nrows,) of a table.
    title : str or None
        A short title (e.g. for the caption of a figure of it).
    description : str or None
        What it is, in words.
    sources : tuple of str or None
        The products it is made from, by their paths (e.g. those of the
        run that a figure shows).
    """
    path: str
    kind: str
    ids: dict[str, str]
    format: str
    unit: str | None = None
    shape: tuple[int, ...] | None = None
    title: str | None = None
    description: str | None = None
    sources: tuple[str, ...] | None = None

    def matches(self, kind: str | None, ids: Mapping[str, str]) -> bool:
        """
        Return whether it is of the given kind (any, if None) and has the
        given ids (it may have others).
        """
        return (kind is None or self.kind == kind) and all(
            self.ids.get(name) == value for name, value in ids.items())

    def dump(self) -> dict[str, Any]:
        """Return it as an entry of a manifest (JSON values)."""
        info = dict(
            path=self.path, kind=self.kind, ids=self.ids, format=self.format,
            unit=self.unit,
            shape=None if self.shape is None else list(self.shape),
            title=self.title, description=self.description,
            sources=None if self.sources is None else list(self.sources))
        return {key: x for key, x in info.items() if x is not None}

    @classmethod
    def load(cls, info: Mapping[str, Any]) -> 'Product':
        """Return the product of an entry of a manifest."""
        shape, sources = info.get('shape'), info.get('sources')
        return cls(
            info['path'], info['kind'], dict(info.get('ids', {})),
            info['format'], info.get('unit'),
            None if shape is None else tuple(shape), info.get('title'),
            info.get('description'),
            None if sources is None else tuple(sources))


class Run:
    """
    A run of a task, as its manifest describes it (see open).

    Attributes
    ----------
    path : str
        Its directory.
    task : str
        The task that made it (e.g. 'eval', 'fit', 'plot').
    options : dict
        The options of the task that tell its products apart (e.g. the
        mode of eval).
    status : {'running', 'complete', 'failed'}
        Whether the task is still running, finished, or stopped by an
        error.
    gbkfit_version : str
        The version of GBKFIT that made it.
    created, updated : str
        When it was created, and when its manifest was last written (ISO
        8601, UTC).
    """

    def __init__(self, path: str, manifest: Mapping[str, Any]) -> None:
        self.path = path
        self.task: str = manifest['task']
        self.options: dict[str, Any] = dict(manifest.get('options', {}))
        self.status: Status = manifest['status']
        self.gbkfit_version: str = manifest['gbkfit_version']
        self.created: str = manifest['created']
        self.updated: str = manifest['updated']
        self._products = [Product.load(p) for p in manifest['products']]

    def products(
            self,
            kind: str | None = None,
            **ids: str
    ) -> list[Product]:
        """
        Return its products of a kind (any, by default) and with the given
        ids, in the order they were written.

        Examples
        --------
        >>> run.products('model', observation='observation0')
        """
        return [p for p in self._products if p.matches(kind, ids)]

    def product(self, kind: str, **ids: str) -> Product:
        """
        Return its one product of a kind and with the given ids.

        Raises
        ------
        LookupError
            If it has none, or several.
        """
        found = self.products(kind, **ids)
        if len(found) != 1:
            raise LookupError(
                f"run {self.path} has {len(found)} products of kind "
                f"{kind!r} with the ids {ids}; one was asked for")
        return found[0]

    def filename(self, product: Product) -> str:
        """Return the absolute path of the file of a product."""
        return os.path.join(self.path, *PurePosixPath(product.path).parts)

    def read(self, product: Product) -> Any:
        """
        Return a product, as astropy and Python objects: a FITS file as
        its primary HDU (with its data in memory; its header has the world
        coordinates and the unit), an ECSV file as an astropy Table (with
        the units of its columns), and a JSON or YAML file as its values.

        Raises
        ------
        ValueError
            If it is in another format (read its file, see filename).
        """
        filename = self.filename(product)
        match product.format:
            case 'fits':
                data, header = fits.getdata(filename, header=True)
                return fits.PrimaryHDU(data, header)
            case 'ecsv':
                return Table.read(filename, format='ascii.ecsv')
            case 'json':
                return json.loads(Path(filename).read_text())
            case 'yaml':
                return ruamel.yaml.YAML(typ='safe').load(Path(filename))
        raise ValueError(
            f"products in the format {product.format!r} have no reader; "
            f"read the file {filename}")


def open(path: str) -> Run:
    """
    Open a run: read its manifest.

    A run that is still running gives its products so far; open it again
    to see newer ones.

    Parameters
    ----------
    path : str
        The directory of the run.

    Returns
    -------
    Run
        The run.

    Raises
    ------
    ValueError
        If the directory has no manifest of a run, or one of a newer
        version of the format.
    """
    path = os.path.abspath(path)
    filename = os.path.join(path, MANIFEST)
    try:
        manifest = json.loads(Path(filename).read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"{path} is not a run of GBKFIT: {e}") from e
    if not isinstance(manifest, dict) or manifest.get('format') != FORMAT \
            or not isinstance(manifest.get('version'), int):
        raise ValueError(
            f"{filename} is not the manifest of a run of GBKFIT")
    if manifest['version'] > FORMAT_VERSION:
        raise ValueError(
            f"run {path} is of version {manifest['version']} of the format, "
            f"newer than this GBKFIT reads ({FORMAT_VERSION}); upgrade GBKFIT")
    return Run(path, manifest)


class RunWriter:
    """
    Write the products of a run of a task, and its manifest (see create
    and reopen). Threads can write products at once.

    Use it as a context manager: on leaving, the status of the run is
    'complete', or 'failed' if an exception left.
    """

    def __init__(
            self,
            path: str,
            task: str,
            options: Mapping[str, Any],
            created: str,
            products: Iterable[Product]
    ) -> None:
        self.path = path
        self._task = task
        self._options = dict(options)
        self._created = created
        self._status: Status = 'running'
        self._products = {product.path: product for product in products}
        self._lock = threading.Lock()
        self._write_manifest()

    def __enter__(self) -> 'RunWriter':
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None
    ) -> None:
        self.close('complete' if exc_type is None else 'failed')

    def write(
            self,
            path: str,
            kind: str,
            value: Any,
            *,
            unit: str | None = None,
            title: str | None = None,
            description: str | None = None,
            sources: Iterable[str] | None = None,
            **ids: str
    ) -> Product:
        """
        Write a product, and add it to the manifest (a product of the
        same path is replaced, e.g. a best fit so far).

        The extension of the path gives the format, which must suit the
        value:

        - .fits: data on a grid (GridData) or spectra (SpectraData), with
          their world coordinates and unit, or another array (with the
          given unit); the header has provenance cards (ORIGIN, DATE,
          GBKFVER, GBKFTASK and GBKFKIND);
        - .ecsv: an astropy Table, with the units of its columns and the
          provenance in its metadata;
        - .json and .yaml: numbers, strings, lists and dicts (strict JSON:
          NaN and infinities are null);
        - .png, .pdf and .svg: the bytes of a figure (e.g. from
          matplotlib's savefig).

        Parameters
        ----------
        path : str
            The file, relative to the directory of the run
            ('/'-separated, without '..').
        kind : str
            What it is (e.g. 'model', 'residual', 'samples').
        value : Any
            What to write.
        unit : str, optional
            The unit of an array (data on a grid and spectra carry their
            own).
        title : str, optional
            A short title (e.g. for the caption of a figure of it).
        description : str, optional
            What it is, in words.
        sources : Iterable of str, optional
            The products it is made from, by their paths.
        **ids : str
            What it is of (e.g. observation='ngc1313', item='vlos').

        Returns
        -------
        Product
            Its entry in the manifest.

        Raises
        ------
        ValueError
            If the path is invalid, or its extension is not of a format.
        TypeError
            If the value does not suit the format, or a unit is given for
            a value that carries its own.
        """
        path = _check_path(path)
        format_ = _FORMATS.get(PurePosixPath(path).suffix)
        if format_ is None:
            raise ValueError(
                f"the extension of {path!r} must be one of {list(_FORMATS)}")
        write, unit, shape = self._writer(path, format_, kind, value, unit)
        _write_atomic(self.filename(path), write)
        return self._add(Product(
            path, kind, ids, format_, unit, shape, title, description,
            None if sources is None else tuple(sources)))

    def record(
            self,
            path: str,
            kind: str,
            format: str,
            *,
            title: str | None = None,
            description: str | None = None,
            **ids: str
    ) -> Product:
        """
        Add a product that other code writes (e.g. a log, written as it
        goes) to the manifest; write it to its filename (see filename).

        Returns
        -------
        Product
            Its entry in the manifest.
        """
        path = _check_path(path)
        return self._add(Product(
            path, kind, ids, format, None, None, title, description))

    def filename(self, path: str) -> str:
        """
        Return the absolute path of a file of the run (by its path in the
        run), and create its directory.
        """
        filename = os.path.join(self.path, *PurePosixPath(path).parts)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        return filename

    def close(self, status: Status = 'complete') -> None:
        """Write the final status of the run to its manifest."""
        with self._lock:
            self._status = status
            self._write_manifest()

    def _writer(
            self,
            path: str,
            format_: str,
            kind: str,
            value: Any,
            unit: str | None
    ) -> tuple[Callable[[str], None], str | None, tuple[int, ...] | None]:
        """
        Return how to write a value to a file of a format, its unit and
        its shape.
        """
        carries_unit = isinstance(
            value, (gridutils.GridData, gridutils.SpectraData))
        if unit is not None and not isinstance(value, np.ndarray):
            raise TypeError(
                f"{path}: a unit is given only for an array; a "
                f"{type(value).__name__} carries its own, or has none")
        cards = self._cards(kind)
        if format_ == 'fits' and isinstance(value, gridutils.GridData):
            return (lambda f: fitsutils.write_data(
                f, value.data, value.coords, value.spectral_axis,
                overwrite=True, unit=value.unit, cards=cards),
                value.unit, value.data.shape)
        if format_ == 'fits' and isinstance(value, gridutils.SpectraData):
            return (lambda f: fitsutils.write_spectra(
                f, value.data, value.coords, overwrite=True, unit=value.unit,
                cards=cards), value.unit, value.data.shape)
        if format_ == 'fits' and isinstance(value, np.ndarray):
            header = fits.Header(cards)
            if unit is not None:
                header['BUNIT'] = unit
            return (lambda f: fits.writeto(
                f, value, header, output_verify='exception', overwrite=True,
                checksum=True), unit, value.shape)
        if format_ == 'ecsv' and isinstance(value, Table):
            table = value.copy(copy_data=False)
            table.meta.update({key.lower(): v for key, v in cards.items()})
            return (lambda f: table.write(
                f, format='ascii.ecsv', overwrite=True), None, (len(table),))
        # (data on a grid and spectra are tuples, but not plain values)
        if format_ in ('json', 'yaml') and isinstance(value, _PLAIN) \
                and not carries_unit:
            values = _strict(numutils.nativify(value))
            if format_ == 'json':
                text = json.dumps(values, indent=2, allow_nan=False) + '\n'
                return lambda f: Path(f).write_text(text), None, None
            return (lambda f: ruamel.yaml.YAML().dump(values, Path(f)),
                    None, None)
        if format_ in _FIGURES and isinstance(value, bytes):
            return lambda f: Path(f).write_bytes(value), None, None
        raise TypeError(
            f"{path}: a {type(value).__name__} cannot be written as "
            f"{format_}")

    def _cards(self, kind: str) -> dict[str, str]:
        """Return the provenance cards of a product of a kind."""
        # (the FITS DATE is in UTC, without a time zone)
        date = datetime.datetime.now(datetime.timezone.utc).strftime(
            '%Y-%m-%dT%H:%M:%S')
        return dict(
            ORIGIN='GBKFIT', DATE=date, GBKFVER=gbkfit.__version__,
            GBKFTASK=self._task, GBKFKIND=kind)

    def _add(self, product: Product) -> Product:
        """Add a product to the manifest (replacing one of its path)."""
        if bad := {k: v for k, v in product.ids.items()
                   if not isinstance(v, str)}:
            raise TypeError(
                f"{product.path}: ids are strings; these are not: {bad}")
        with self._lock:
            self._products.pop(product.path, None)
            self._products[product.path] = product
            self._write_manifest()
        return product

    def _write_manifest(self) -> None:
        """Write the manifest, atomically."""
        manifest = dict(
            format=FORMAT, version=FORMAT_VERSION, task=self._task,
            options=self._options, status=self._status,
            gbkfit_version=gbkfit.__version__, created=self._created,
            updated=_now(),
            products=[p.dump() for p in self._products.values()])
        text = json.dumps(_strict(manifest), indent=2, allow_nan=False) + '\n'
        _write_atomic(
            os.path.join(self.path, MANIFEST),
            lambda f: Path(f).write_text(text))


def create(
        path: str,
        task: str,
        exists: Literal['terminate', 'overwrite', 'unique'] = 'terminate',
        options: Mapping[str, Any] | None = None
) -> RunWriter:
    """
    Create the directory of a run of a task, and its manifest (status
    'running').

    Parameters
    ----------
    path : str
        The directory.
    task : str
        The task (e.g. 'eval', 'fit', 'plot').
    exists : {'terminate', 'overwrite', 'unique'}
        What to do if the directory exists: raise an error, use it (its
        manifest is replaced; files of the earlier run that this one does
        not rewrite stay, but are not in the manifest), or create the
        first free path with a numeric suffix (e.g. 'out_2').
    options : Mapping, optional
        The options of the task that tell its products apart (JSON
        values; e.g. the mode of eval).

    Returns
    -------
    RunWriter
        The writer of the run.

    Raises
    ------
    FileExistsError
        If the directory exists and exists is 'terminate', or the path is
        a file.
    """
    return RunWriter(
        make_directory(path, exists), task, options or {}, _now(), ())


def reopen(path: str) -> RunWriter:
    """
    Reopen a run to write more products (status 'running' again; e.g. to
    resume a fit, or to add a bootstrap to it), keeping its products.

    Parameters
    ----------
    path : str
        The directory of the run.

    Returns
    -------
    RunWriter
        The writer of the run.

    Raises
    ------
    ValueError
        If the directory is not a run that this GBKFIT reads (see open).
    """
    run = open(path)
    return RunWriter(
        run.path, run.task, run.options, run.created, run.products())


def make_directory(
        path: str,
        exists: Literal['terminate', 'overwrite', 'unique']
) -> str:
    """
    Create the directory of a run, e.g. to know it before the run starts
    (create then writes into it with exists 'overwrite').

    Creating a directory is atomic, so two runs cannot both create the
    same one.

    Parameters
    ----------
    path : str
        The directory.
    exists : {'terminate', 'overwrite', 'unique'}
        What to do if it exists: fail, use it, or create the first free
        path with a numeric suffix (e.g. output_2).

    Returns
    -------
    str
        The directory created (or used), absolute.

    Raises
    ------
    FileExistsError
        If the directory exists and exists is 'terminate', or the path is
        a file.
    """
    path_obj = Path(path).absolute()
    try:
        path_obj.mkdir(parents=True)
        return str(path_obj)
    except FileExistsError:
        pass
    if not path_obj.is_dir():
        raise FileExistsError(f"path '{path_obj}' exists as a file")
    if exists == 'terminate':
        raise FileExistsError(f"path '{path_obj}' already exists")
    if exists == 'overwrite':
        return str(path_obj)
    while True:
        candidate = miscutils.make_unique_path(path_obj)
        try:
            candidate.mkdir(parents=True)
            return str(candidate)
        except FileExistsError:
            # Another run created it first: try the next one
            continue


class Output(NamedTuple):
    """
    An output to write as a product of a run (see write_outputs): its path
    in the run, kind, value and ids, and its title and description, if
    any.
    """
    path: str
    kind: str
    value: Any
    ids: dict[str, str]
    title: str | None = None
    description: str | None = None


# The folders of the extra outputs by their ids, in this order (see
# _extra_output)
_EXTRA_FOLDERS = dict(
    observation='observations', galaxy='galaxies', group='groups',
    component='components')


def data_outputs(group_plan: 'GroupPlan') -> dict[str, Output]:
    """
    Return the data of the observations of a plan of a group as outputs
    (the images on the drivers of the observations), by path, in the form
    the observations give them (see Observation.output): their values
    (kind 'data', NaN where masked) and errors ('error'), in their units,
    of each data item, and the images of their values and of their PSF
    that the observations make ('data_image'; e.g. the dirty image and
    beam of visibilities, see Observation.images and psf_images), in
    observations/<observation>/<kind>/<item or image>.
    """
    group = group_plan.group()
    outputs = []
    for i, (observation, driver) in enumerate(zip(
            group.observations(), group_plan.observation_drivers())):
        if (dataset := observation.data()) is None:
            continue
        name = group.observation_label(i)
        for key in observation.keys():
            item = dataset[key]
            unit = item.unit()
            outputs.append(_observation_output(
                name, 'data', key, observation.output(item.data(), unit)))
            if item.error() is not None:
                outputs.append(_observation_output(
                    name, 'error', key,
                    observation.output(item.error(), unit)))
        weights = _weights(dataset)
        images = observation.images(
            driver, {key: dataset[key].data() for key in observation.keys()},
            weights, _units(dataset)) | observation.psf_images(driver, weights)
        outputs += [
            _observation_output(name, 'data_image', image, x, 'image')
            for image, x in images.items()]
    return {output.path: output for output in outputs}


def _units(dataset: Any) -> dict[str, Any] | None:
    """Return the units of the data items of a dataset; None without data."""
    return None if dataset is None \
        else {key: x.unit() for key, x in dataset.items()}


def _weights(dataset: Any) -> dict[str, np.ndarray] | None:
    """
    Return the weights of the values of the data items of a dataset, 1 /
    error^2 (0 where they have none); None without data or errors.
    """
    if dataset is None or any(x.error() is None for x in dataset.values()):
        return None
    with np.errstate(divide='ignore'):
        return {
            key: np.where(np.isfinite(x.error()), 1 / x.error() ** 2, 0)
            for key, x in dataset.items()}


def _unit(
        observation: 'Observation', key: str
) -> 'astropy.units.UnitBase | None':
    """
    Return the unit of a data item of an observation (which its model
    takes too), if it has data with one; None otherwise.
    """
    data = observation.data()
    return None if data is None else data[key].unit()


def evaluation_outputs(
        group_plan: 'GroupPlan',
        objective: 'Objective | None',
        values: 'ParamValues',
        model: bool,
        residuals: bool,
        extra: bool = True
) -> dict[str, Output]:
    """
    Return the outputs of an evaluation of a plan of a group, by path, in
    the form the observations give them (their images made on their
    drivers), in observations/<observation>/<kind>/<item>
    of each data item: the models (kind 'model'; 'model_mask' and
    'model_weight'), and the residuals ((model - data) / error) without
    and with the weights of the data items ('residual' and 'wresidual';
    these need the objective; the values of the parameters must have
    those of the solved elements that it solves, see Objective.solved).
    With the models, a flux scale of several values of an observation
    (see ObservationSky) is 'flux_scale'
    (observations/<observation>/flux_scale), of the shape of its spectra
    but their axis. With the models, the images that the observations
    make of them ('model_image', in place of the item: the name of the
    image; see Observation.images), and of the data minus them
    ('residual_image'), with the weights of the data. Unless left out
    (extra), the extra outputs of the models are 'extra', with their
    titles and descriptions (see _extra_output).
    """
    # (imported here: reading runs does not need the models)
    from gbkfit.model import Extras
    group = group_plan.group()
    outputs = []
    extras = Extras() if extra else None
    if model:
        # (the plan reuses its arrays of the models, so copy them: the next
        # evaluation would overwrite the outputs)
        models = copy.deepcopy(group_plan.model_h(values, extras))
        for i, data_i in enumerate(models):
            observation = group.observations()[i]
            name = group.observation_label(i)
            for key, value in data_i.items():
                for kind, part in (
                        ('model', 'd'), ('model_mask', 'm'),
                        ('model_weight', 'w')):
                    if value.get(part) is None:
                        continue
                    output = observation.output(
                        value[part],
                        _unit(observation, key) if part == 'd' else None)
                    outputs.append(
                        _observation_output(name, kind, key, output))
            outputs += _images(
                observation, group_plan.observation_drivers()[i], name,
                {k: v['d'] for k, v in data_i.items()})
            if (scale := _flux_scale_map(observation, values, data_i)) \
                    is not None:
                outputs.append(Output(
                    f'observations/{name}/flux_scale.fits', 'flux_scale',
                    scale, dict(observation=name)))
    if residuals:
        if objective is None:
            raise ValueError("the residuals need an objective")
        for kind, weighted in (('residual', False), ('wresidual', True)):
            # (the objective reuses its residual arrays, so copy them;
            # without the models, the extra outputs are those of the
            # first residual evaluation, which evaluates the models)
            data = copy.deepcopy(objective.residual_nddata_h(
                values, weighted, None if model or weighted else extras))
            for i, data_i in enumerate(data):
                name = group.observation_label(i)
                observation = group.observations()[i]
                outputs += [
                    _observation_output(
                        name, kind, key, observation.output(value))
                    for key, value in data_i.items()]
    outputs += [_extra_output(x) for x in extras or ()]
    return {output.path: output for output in outputs}


def _flux_scale_map(
        observation: 'Observation',
        values: 'ParamValues',
        model: Mapping[str, Mapping[str, np.ndarray | None]]
) -> np.ndarray | None:
    """
    Return the values of the flux scale of an observation of several
    values (see ObservationSky), of the shape of its model but the axis of
    the scale (e.g. an image of the scale of each spectrum of a cube);
    None for an observation without one.
    """
    if (flux_scale := observation.flux_scale()) is None \
            or flux_scale[1].axis is None:
        return None
    param, scale = flux_scale
    shape = list(model[scale.items[0]]['d'].shape)
    del shape[scale.axis]
    return np.reshape(values[param], shape)


def _images(
        observation: 'Observation',
        driver: 'Driver',
        name: str,
        model: Mapping[str, np.ndarray]
) -> list[Output]:
    """
    Return the images that an observation (of the given name in the
    outputs) makes of its model ('model_image') and of its data minus the
    model ('residual_image'; see Observation.images), on its driver.
    """
    dataset = observation.data()
    weights = _weights(dataset)
    units = _units(dataset)
    images = [
        _observation_output(name, 'model_image', image, x, 'image')
        for image, x in observation.images(
            driver, model, weights, units).items()]
    # (an observation that makes no images of its model makes none of the
    # residual either)
    if images and dataset is not None:
        residual = {key: dataset[key].data() - model[key] for key in model}
        images += [
            _observation_output(name, 'residual_image', image, x, 'image')
            for image, x in observation.images(
                driver, residual, weights, units).items()]
    return images


def _extra_output(extra: 'Extra') -> Output:
    """
    Return an extra output of a model (kind 'extra', with its ids and its
    name as the id 'name'), in the folders of its ids (see
    _EXTRA_FOLDERS; e.g. observations/<observation>/galaxies/<galaxy>/
    components/<component>/<name>), or in observations/<observation>/
    extra/<name> if it is of the observation alone (e.g. the model cube;
    its folder holds the other outputs of the observation by kind).
    """
    folders = []
    for key, folder in _EXTRA_FOLDERS.items():
        if key in extra.ids:
            folders += [folder, extra.ids[key]]
    if set(extra.ids) == {'observation'}:
        folders.append('extra')
    return Output(
        '/'.join(folders + [f'{extra.name}{_extension(extra.value)}']),
        'extra', extra.value, extra.ids | dict(name=extra.name),
        extra.title, extra.description)


def _observation_output(
        observation: str,
        kind: str,
        name: str,
        value: Any,
        id_name: str = 'item'
) -> Output:
    """
    Return an output of a kind of an observation, of a data item (or of
    another id: e.g. 'image'), in observations/<observation>/<kind>/
    <name>.
    """
    return Output(
        f'observations/{observation}/{kind}/{name}{_extension(value)}', kind,
        value, {'observation': observation, id_name: name})


def _extension(value: Any) -> str:
    """
    Return the extension of the file of an output by its type: FITS for
    arrays (data on a grid and spectra with world coordinates), ECSV for
    tables, and JSON for other values (numbers, strings, lists and dicts).

    Raises
    ------
    TypeError
        If the type is not of these.
    """
    if isinstance(
            value, (gridutils.GridData, gridutils.SpectraData, np.ndarray)):
        return '.fits'
    if isinstance(value, Table):
        return '.ecsv'
    if isinstance(value, (bool, int, float, str, list, tuple, dict,
                          np.generic)):
        return '.json'
    raise TypeError(f"outputs of type {type(value).__name__} are unsupported")


def write_outputs(
        run: RunWriter,
        outputs: Mapping[str, Output]
) -> None:
    """Write outputs as products of a run (see RunWriter.write)."""
    for output in outputs.values():
        run.write(
            output.path, output.kind, output.value, title=output.title,
            description=output.description, **output.ids)


def _check_path(path: str) -> str:
    """
    Return the path of a product in a run, normalised ('/'-separated).

    Raises
    ------
    ValueError
        If it is absolute, goes up ('..'), or is the manifest.
    """
    parts = PurePosixPath(path.replace(os.sep, '/')).parts
    if not parts or parts[0] == '/' or '..' in parts:
        raise ValueError(
            f"the path of a product must be relative to the run, without "
            f"'..'; it is {path!r}")
    path = '/'.join(p for p in parts if p != '.')
    if path == MANIFEST:
        raise ValueError(f"{MANIFEST} is the manifest of the run")
    return path


def _write_atomic(filename: str, write: Callable[[str], None]) -> None:
    """
    Write a file atomically: write it to a temporary file next to it, then
    rename that, so that readers see the old file or the new one, never a
    part.
    """
    directory, name = os.path.split(filename)
    os.makedirs(directory, exist_ok=True)
    # (a unique name, rather than a tempfile, whose files only their owner
    # can read)
    temporary = os.path.join(directory, f'.{name}.{uuid.uuid4().hex}.tmp')
    try:
        write(temporary)
        os.replace(temporary, filename)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _strict(x: Any) -> Any:
    """Return JSON values with NaN and infinities as None."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, list):
        return [_strict(item) for item in x]
    if isinstance(x, dict):
        return {key: _strict(value) for key, value in x.items()}
    return x


def _now() -> str:
    """Return the time now (ISO 8601, UTC, in seconds)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec='seconds')
