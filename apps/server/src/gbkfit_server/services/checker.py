"""
The checker of the text of a file of the workspace (see checks): a
process that checks it as its task would (see gbkfit.tasks.check), and
places its problems on its lines:

    python -m gbkfit_server.services.checker

It reads {path, text} (JSON) from stdin, and writes {diagnostics: [{line,
severity, message}]} (JSON; lines from 1) to stdout; its log goes to
stderr. A configuration (e.g. a setup) is checked itself; a piece of a
project is checked in each setup that names it, with its text in place,
for the problems in it; a piece that no setup names is only read.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import ruamel.yaml
from ruamel.yaml.comments import CommentedMap, CommentedSeq


# The sections of the configurations that name the pieces of a folder
_SECTIONS = dict(
    galaxies='galaxies', observations='observations', params='params',
    fitters='fitter')

_yaml = ruamel.yaml.YAML()
_safe_yaml = ruamel.yaml.YAML(typ='safe')


def main() -> None:
    """Check the file of the request on stdin."""
    # stdout carries the result: everything else written to it (e.g. by
    # imports, as the library is imported here) goes to stderr
    out = os.fdopen(os.dup(1), 'w')
    os.dup2(2, 1)
    request = json.load(sys.stdin)
    json.dump(dict(diagnostics=diagnose(request['path'], request['text'])),
              out)
    out.flush()


def diagnose(path: str, text: str) -> list[dict[str, Any]]:
    """Return the diagnostics of the text of a file."""
    import gbkfit.project
    from gbkfit.tasks.check import check
    try:
        document = _yaml.load(text)
    except ruamel.yaml.MarkedYAMLError as e:
        mark = e.problem_mark or e.context_mark
        return [_diagnostic(
            0 if mark is None else mark.line, 'error', str(e.problem))]
    try:
        project = gbkfit.project.find(path)
    except ValueError as e:
        return [_diagnostic(0, 'error', str(e))]
    folder = None if project is None else _piece_folder(project.path, path)
    if folder is None:
        return [
            _diagnostic(_line(document, p.path), p.severity, _text(p))
            for p in check(path, {path: text}).problems]
    name = Path(path).relative_to(
        Path(project.path, folder)).with_suffix('').as_posix()
    diagnostics = {}
    for setup in sorted(Path(project.path, 'setups').rglob('*.yaml')):
        prefixes = _uses(setup, _SECTIONS[folder], name)
        if not prefixes:
            continue
        via = setup.relative_to(project.path).as_posix()
        for problem in check(str(setup), {path: text}).problems:
            for prefix in prefixes:
                if problem.path[:len(prefix)] == prefix:
                    diagnostic = _diagnostic(
                        _line(document, problem.path[len(prefix):]),
                        problem.severity, f"{_text(problem)} (in {via})")
                    diagnostics[diagnostic['line'], _text(problem)] = \
                        diagnostic
    return list(diagnostics.values())


def _piece_folder(project: str, path: str) -> str | None:
    """Return the folder of the pieces that a file is in, if any."""
    parts = Path(path).relative_to(project).parts
    return parts[0] if len(parts) > 1 and parts[0] in _SECTIONS else None


def _uses(setup: Path, section: str, name: str) -> list[list[str | int]]:
    """
    Return the paths in a setup of the items of a section that name a
    piece (e.g. [['galaxies', 1]]); none if it cannot be read.
    """
    try:
        value = _safe_yaml.load(setup).get(section)
    except (ruamel.yaml.YAMLError, AttributeError):
        return []
    if section in ('fitter', 'params') and (
            value == name or isinstance(value, list) and name in value):
        # (the layers of params are merged: their paths are those of all)
        return [[section]]
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [[section, i] for i, item in enumerate(value) if item == name]


def _line(document: Any, path: list[str | int]) -> int:
    """
    Return the line (from 0) of the part of a document at a path: of its
    deepest part in the document.
    """
    node, line = document, 0
    for segment in path:
        if isinstance(node, CommentedMap) and segment in node:
            line = node.lc.key(segment)[0]
        elif isinstance(node, CommentedSeq) and isinstance(segment, int) \
                and 0 <= segment < len(node):
            line = node.lc.item(segment)[0]
        else:
            break
        node = node[segment]
    return line


def _text(problem: Any) -> str:
    """Return the message of a problem with its path."""
    path = ''.join(
        f'[{s}]' if isinstance(s, int) else f'.{s}' for s in problem.path)
    path = path.lstrip('.')
    return f"{path}: {problem.message}" if path else problem.message


def _diagnostic(line: int, severity: str, message: str) -> dict[str, Any]:
    """Return a diagnostic, its line from 1."""
    return dict(line=line + 1, severity=severity, message=message)


if __name__ == '__main__':
    main()
