"""
The rule of the applications of GBKFIT: they use the public API of the
library only (no private modules or names), and never import each other
(they reach each other through the API of the server).
"""

import ast
import pathlib

import gbkfit_cli


SOURCE = pathlib.Path(gbkfit_cli.__file__).parent


def is_private(name):
    """Return whether a name is private (_name, not __name__)."""
    return name.startswith('_') and not name.endswith('__')


def imports(path):
    """Yield each module a file imports, with the names it imports."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, []
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module, [alias.name for alias in node.names]


def test_only_the_public_api_of_the_library_is_imported():
    for path in SOURCE.rglob('*.py'):
        for module, names in imports(path):
            parts = module.split('.')
            if parts[0] == 'gbkfit':
                assert not any(map(is_private, parts + names)), (
                    f"{path.name} imports {module} {names}")
            elif parts[0].startswith('gbkfit_'):
                assert parts[0] == 'gbkfit_cli', (
                    f"{path.name} imports the application {parts[0]}")
