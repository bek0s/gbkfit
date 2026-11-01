"""
The library never imports the applications of GBKFIT (gbkfit_cli,
gbkfit_server, ...): they depend on it, not the other way. Its packages
import the names that the others export, not their private modules.
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


def imported_names(path):
    """
    Yield each module a file imports, and each name it imports from one
    (e.g. 'gbkfit.params._cache.StageCache'), relative imports resolved.
    """
    package = path.relative_to(SOURCE.parent).with_suffix('').parts[:-1]
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            parts = node.module.split('.') if node.module else []
            if node.level:
                parts = [*package[:len(package) - node.level + 1], *parts]
            module = '.'.join(parts)
            yield module
            yield from (f'{module}.{alias.name}' for alias in node.names)


def test_packages_do_not_import_the_private_parts_of_others():
    # (audit 2026-10-31 #54; the modules at the root of the library are
    # one package, e.g. the schema reads the sections of the setup)
    for path in SOURCE.rglob('*.py'):
        own = path.relative_to(SOURCE).parts[0].removesuffix('.py')
        for name in imported_names(path):
            parts = name.split('.')
            if parts[0] != 'gbkfit' or len(parts) < 3 or parts[1] == own:
                continue
            if path.parent == SOURCE and (SOURCE / f'{parts[1]}.py').exists():
                continue
            assert not any(part.startswith('_') for part in parts[2:]), (
                f"{path.relative_to(SOURCE)} imports {name}")
