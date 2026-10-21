"""
The library never imports the applications of GBKFIT (gbkfit_cli,
gbkfit_server, ...): they depend on it, not the other way.
"""

import ast
import pathlib

import gbkfit


SOURCE = pathlib.Path(gbkfit.__file__).parent


def imported_modules(path):
    """Yield each module a file imports (absolute imports only)."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module


def test_the_library_does_not_import_the_applications():
    for path in SOURCE.rglob('*.py'):
        for module in imported_modules(path):
            assert not module.startswith('gbkfit_'), (
                f"{path.relative_to(SOURCE)} imports {module}")
