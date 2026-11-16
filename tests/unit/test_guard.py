"""
A guard of the design: code outside the package of the components, or of
the mass, asks them what it needs (e.g. Component.needs,
Emission.seen_in) and never tests their concrete types, which would make
it edited for each new type (see interface-audit, 2026-11-16).
"""

import ast
from pathlib import Path

import gbkfit
from gbkfit.model.galaxy import component_parser
from gbkfit.model.components.roles import emission_parser
from gbkfit.model.mass import mass_component_parser


_ROOTS = (
    Path(gbkfit.__file__).parent,
    Path(__file__).parents[2] / 'apps' / 'server' / 'src')

# The package that owns the types of each family: the only code that may
# test them
_OWNERS = {
    'components': Path(gbkfit.__file__).parent / 'model' / 'components',
    'mass': Path(gbkfit.__file__).parent / 'model' / 'mass'}


def _concrete_types() -> dict[str, Path]:
    """Return the owner of each concrete type, by its class name."""
    owners = {}
    for parser, owner in (
            (component_parser, 'components'), (emission_parser, 'components'),
            (mass_component_parser, 'mass')):
        for cls in parser.registered_classes().values():
            for klass in cls.__mro__:
                if klass.__module__.startswith('gbkfit.model.'):
                    owners.setdefault(klass.__name__, _OWNERS[owner])
    # (the abstract bases that every member of a family is, e.g.
    # Component, are no concrete types)
    for name in ('Component', 'Emission', 'MassComponent',
                 'TypedSerializable', 'Serializable'):
        owners.pop(name, None)
    return owners


def _tested_types(node: ast.Call) -> list[str]:
    """Return the names of the types that a call of isinstance tests."""
    if not (isinstance(node.func, ast.Name) and node.func.id == 'isinstance'
            and len(node.args) == 2):
        return []
    types = node.args[1]
    items = types.elts if isinstance(types, ast.Tuple) else [types]
    names = []
    for item in items:
        if isinstance(item, ast.Name):
            names.append(item.id)
        elif isinstance(item, ast.Attribute):
            names.append(item.attr)
    return names


def test_no_code_tests_concrete_types_outside_their_package():
    owners = _concrete_types()
    found = []
    for root in _ROOTS:
        for path in root.rglob('*.py'):
            if 'build' in path.parts:
                continue
            tree = ast.parse(path.read_text(), str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for name in _tested_types(node):
                    owner = owners.get(name)
                    if owner is not None and not path.is_relative_to(owner):
                        found.append(f"{path}:{node.lineno}: {name}")
    assert not found, (
        "concrete types tested outside their package (ask them instead):\n"
        + "\n".join(found))
