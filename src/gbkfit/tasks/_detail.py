
import json
import logging
from pathlib import Path
from typing import Any, Literal

from gbkfit.params import ParamDesc
from gbkfit.utils import iterutils, miscutils


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
        _log.info(
            f"the following sections are not recognised and will be ignored: "
            f"{unknown_sections}")
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
    for s in ['pdescs', 'params', 'objective', 'fitter']:
        if s in config and not iterutils.is_mapping(config[s]):
            wrong_type_dict.append(s)
    for s in ['datasets', 'models']:
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

    # Listify some sections to make parsing more streamlined
    # and ensure they have the same length
    lengths = {}
    for s in ['datasets', 'models']:
        if s in config:
            config[s] = iterutils.listify(config[s])
            lengths[s] = len(config[s])
    if len(set(lengths.values())) > 1:
        raise RuntimeError(
            f"the following sections must have the same length: "
            f"{lengths}")

    # Make sure the return value is pure json
    return json.loads(json.dumps(config))


def make_output_dir(
        path: str,
        mode: Literal['terminate', 'overwrite', 'unique']
) -> str:
    """
    Handle directory path based on specified mode and create the
    directory.
    """
    path_obj = Path(path).absolute()
    result_path = str(path_obj)

    # Handle current directory case
    if path_obj.samefile(Path.cwd()):
        return result_path

    # Handle non-existent path case
    if not path_obj.exists():
        Path(result_path).mkdir(parents=True, exist_ok=True)
        return result_path

    # At this point it is guaranteed that the path exists

    # Handle file exists case
    if path_obj.is_file():
        raise RuntimeError(f"path '{path}' exists as a file")

    # Handle directory exists case
    if mode == 'terminate':
        raise RuntimeError(f"path '{path}' already exists")
    if mode == 'unique':
        result_path = miscutils.make_unique_path(str(path_obj))
        Path(result_path).mkdir(parents=True, exist_ok=True)

    # Handle directory exists but mode == 'overwrite' case
    return result_path


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
