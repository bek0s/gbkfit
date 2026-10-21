"""
The rule of the imports of the Studio: as every application of GBKFIT, it
uses the public API of the library only, and no other application (it
starts the server as a program, and the server finds its page by its
entry point).
"""

import ast
import pathlib

import gbkfit_studio


SOURCE = pathlib.Path(gbkfit_studio.__file__).parent


def test_no_other_application_and_no_private_library_is_imported():
    for path in SOURCE.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module]
            else:
                continue
            for module in modules:
                top = module.split('.')[0]
                assert top in ('gbkfit', 'gbkfit_studio') \
                    or not top.startswith('gbkfit'), (
                        f"{path.name} imports the application {top}")
                assert not any(
                    part.startswith('_') for part in module.split('.')), (
                        f"{path.name} imports {module}")
