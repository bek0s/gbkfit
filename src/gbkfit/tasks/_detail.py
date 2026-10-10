
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import astropy.io.fits as fits
import numpy as np

import gbkfit.model
import gbkfit.observation
from gbkfit.objective import Objective
from gbkfit.observation import ObservationGroup
from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import (
    fitsutils, gridutils, iterutils, miscutils, numutils, parseutils)


_log = logging.getLogger(__name__)


def prepare_config(
        config: dict[str, Any] | None,
        req_sections: tuple = (),
        opt_sections: tuple = ()
) -> dict[str, Any]:
    """
    Prepare and validate a configuration dictionary.

    This function ensures required sections exist, removes unknown
    sections, verifies types, and structures the configuration in a
    consistent format.

    Currently, this validation applies primarily to the root sections
    of the  configuration. However, future enhancements may extend
    these checks to nested sections as needed.
    """

    # If the configuration file was empty, config will be None
    # Convert it to an empty dict to keep the validation rolling
    if config is None:
        config = {}

    # Get rid of unrecognised sections
    known_sections = []
    unknown_sections = []
    for s in config:
        if s in req_sections + opt_sections:
            known_sections.append(s)
        else:
            unknown_sections.append(s)
    if unknown_sections:
        parseutils.report_unknown(
            "unknown configuration sections", unknown_sections,
            req_sections + opt_sections)
    config = {s: config[s] for s in known_sections}

    # Ensure that the required sections are present
    missing_sections = [s for s in req_sections if s not in config]
    if missing_sections:
        raise RuntimeError(
            f"the following sections must be defined: "
            f"{missing_sections}")

    # Convert sections to empty dicts if they are None/null.
    # This will later on allow the loader functions to parse the empty
    # dicts and provide better error messages to the user.
    for s in config:
        if config[s] is None:
            config[s] = {}

    # Ensure the sections have the right type (if they are present)
    wrong_type_dict = []
    wrong_type_dict_or_seq = []
    for s in ['pdescs', 'params', 'fitter']:
        if s in config and not iterutils.is_mapping(config[s]):
            wrong_type_dict.append(s)
    for s in ['models', 'observations']:
        if s in config and not iterutils.is_sequence_or_mapping(config[s]):
            wrong_type_dict_or_seq.append(s)
    if wrong_type_dict:
        raise RuntimeError(
            f"the following sections must be dictionaries: "
            f"{wrong_type_dict}")
    if wrong_type_dict_or_seq:
        raise RuntimeError(
            f"the following sections must be dictionaries or sequences: "
            f"{wrong_type_dict_or_seq}")

    # Listify some sections to make parsing more streamlined (the
    # observations refer to the models by name, so their numbers differ)
    for s in ['models', 'observations']:
        if s in config:
            config[s] = iterutils.listify(config[s])

    # Make sure the return value is pure json
    return json.loads(json.dumps(config))


def make_output_dir(
        path: str,
        mode: Literal['terminate', 'overwrite', 'unique']
) -> str:
    """
    Create the output directory at the given path and return its absolute
    path. If it exists: an error (terminate), use it (overwrite), or create
    the first free path with a numeric suffix (unique, e.g. 'out_2').
    Creating a directory is atomic, so two runs cannot both create the
    same one.
    """
    path_obj = Path(path).absolute()
    try:
        path_obj.mkdir(parents=True)
        return str(path_obj)
    except FileExistsError:
        pass
    if not path_obj.is_dir():
        raise RuntimeError(f"path '{str(path_obj)}' exists as a file")
    if mode == 'terminate':
        raise RuntimeError(f"path '{str(path_obj)}' already exists")
    if mode == 'overwrite':
        return str(path_obj)
    while True:
        candidate = miscutils.make_unique_path(path_obj)
        try:
            candidate.mkdir(parents=True)
            return str(candidate)
        except FileExistsError:
            # Another run created it first: try the next one
            continue


def dump_dict(json_, yaml_, info: Any, filename: str) -> None:
    """Dump a dictionary to JSON and YAML files."""
    filename = Path(filename)
    try:
        with filename.with_suffix(".json").open("w") as f:
            json_.dump(info, f, indent=2)
        with filename.with_suffix(".yaml").open("w") as f:
            yaml_.dump(info, f)
    except Exception as e:
        raise RuntimeError(f"failed to write to {filename}: {e}")


def merge_pdescs(
        pdescs1: dict[str, ParamDesc] | None,
        pdescs2: dict[str, ParamDesc] | None
) -> dict[str, ParamDesc]:
    """
    Merge two parameter descriptor dictionaries, ensuring no conflicts.

    While this operation is quite simple, this function is supposed to
    be used by particular parts of the task code and provide an
    informative message to the user.
    """
    pdescs1 = pdescs1 or {}
    pdescs2 = pdescs2 or {}
    if conflicting := set(pdescs1) & set(pdescs2):
        raise RuntimeError(
            f"the names of the following user-defined pdescs "
            f"conflict with the names of the model parameters: {conflicting}")
    return pdescs1 | pdescs2


def load_observation_group(cfg):
    """
    The models and observations (with their data, if any) of a
    configuration, as an ObservationGroup.
    """
    with parseutils.config_path('models'):
        models = gbkfit.model.model_parser.load(cfg['models'])
    with parseutils.config_path('observations'):
        observations = gbkfit.observation.observation_parser.load(
            cfg['observations'])
    return gbkfit.observation.ObservationGroup(models, observations)


def evaluation_outputs(
        group: ObservationGroup,
        objective: Objective | None,
        values: ParamValues,
        model: bool,
        residuals: bool
) -> dict[str, Any]:
    """
    Return the outputs of an evaluation, by name, in the form their
    observables give them: the models (model_<i>_<item>_<d|m|w>), the
    unweighted and weighted residuals (residual_<item>_d and
    wresidual_<item>_d, with _<i> after the prefix if there are several
    observations; these need the objective), and the extra outputs
    (<prefix>_extra_<name>; those of the models once, under the first
    prefix: each residual evaluation evaluates the models again).
    """
    outputs = {}
    model_extra: dict[str, Any] = {}
    if model:
        for i, data_i in enumerate(group.model_h(values, model_extra)):
            observable = group.observations()[i].observable()
            for key, value in data_i.items():
                for kind in ('d', 'm', 'w'):
                    if value.get(kind) is not None:
                        outputs[f'model_{i}_{key}_{kind}'] = \
                            observable.output(value[kind])
    extras = [('model', model_extra)]
    if residuals:
        if objective is None:
            raise ValueError("the residuals need an objective")
        for prefix, weighted in (('residual', False), ('wresidual', True)):
            extra: dict[str, Any] = {}
            # (the objective reuses its residual arrays, so copy them)
            data = copy.deepcopy(objective.residual_nddata_h(
                values, weighted, None if model or weighted else extra))
            for i, data_i in enumerate(data):
                prefix_i = prefix + f'_{i}' * (objective.nobservations() > 1)
                observable = group.observations()[i].observable()
                for key, value in data_i.items():
                    outputs[f'{prefix_i}_{key}_d'] = observable.output(value)
            extras.append((prefix, extra))
    for prefix, extra in extras:
        for key, value in extra.items():
            outputs[f'{prefix}_extra_{key}'] = value
    return outputs


def write_outputs(
        json_: Any, yaml_: Any, output_dir: str, outputs: dict[str, Any],
        extra_filename: str
) -> None:
    """
    Write each output by its type: data on a grid (GridData) and spectra
    (SpectraData) to a FITS file with world coordinates, other arrays to
    a FITS file without, and the other values (numbers, strings, lists
    and dicts) together to <extra_filename>.json and .yaml.
    """
    values = {}
    for name, value in outputs.items():
        filename = os.path.join(output_dir, f'{name}.fits')
        if isinstance(value, gridutils.GridData):
            fitsutils.write_data(
                filename, value.data, value.coords, value.spectral_axis,
                overwrite=True)
        elif isinstance(value, gridutils.SpectraData):
            fitsutils.write_spectra(
                filename, value.data, value.coords, overwrite=True)
        elif isinstance(value, np.ndarray):
            fits.writeto(filename, value, overwrite=True)
        elif isinstance(value, (bool, int, float, str, list, tuple, dict,
                                np.generic)):
            values[name] = numutils.nativify(value)
        else:
            raise TypeError(
                f"output {name} has an unsupported type: "
                f"{type(value).__name__}")
    if values:
        dump_dict(
            json_, yaml_, values, os.path.join(output_dir, extra_filename))
