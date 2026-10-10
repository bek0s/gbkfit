"""
Helpers shared by the components.
"""

from .lines import line_parser


def dump_lines(lines_):
    """The lines option of a spectral component: none if not given."""
    if lines_.lines() is None:
        return {}
    return dict(lines=line_parser.dump(list(lines_.lines())))


def dump_name(component):
    """The name option of a component: none if it has no name."""
    name = component.name()
    return dict(name=name) if name is not None else {}
