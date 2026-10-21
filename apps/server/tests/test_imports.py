"""
The rules of the imports of the server: as every application of GBKFIT,
it uses the public API of the library only, and no other application;
and its services are plain Python, without the web framework of its API.
"""

import ast
import pathlib

import gbkfit_server


SOURCE = pathlib.Path(gbkfit_server.__file__).parent

# The web framework, which only the API uses
WEB = ('fastapi', 'starlette', 'uvicorn')


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
                assert parts[0] == 'gbkfit_server', (
                    f"{path.name} imports the application {parts[0]}")


def test_the_services_do_not_import_the_web_framework():
    for path in (SOURCE / 'services').rglob('*.py'):
        for module, _ in imports(path):
            assert module.split('.')[0] not in WEB, (
                f"services/{path.name} imports {module}")
