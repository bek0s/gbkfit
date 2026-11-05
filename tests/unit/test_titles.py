"""
Tests that what configurations choose and what models fit is named for
people: the type of each member of a family has a title (see
TypedSerializable.type_title), every choice of an option has one (see
Choices), and every parameter that the library declares has a title and
a description (see ParamDesc).
"""

import ast
import pathlib
import typing

import pytest

import gbkfit
import gbkfit.fitting
import gbkfit.grid
import gbkfit.instrument
import gbkfit.model
import gbkfit.observation
from gbkfit import catalogue
from gbkfit.config import Choices, option_types


FAMILIES = catalogue()

# The classes whose options are declared, besides the members of the
# families (see test_declared_options)
CLASSES = [
    gbkfit.fitting.BootstrapOptions,
    gbkfit.fitting.UncertaintyOptions,
    gbkfit.grid.SpectralAxis,
    gbkfit.instrument.Instrument,
    gbkfit.model.Galaxy,
    gbkfit.model.components.disks.Clouds,
    gbkfit.observation.Depth]

# The sources of the library, whose declarations of parameters are read
SOURCES = pathlib.Path(gbkfit.__file__).parent


def _title_problem(title):
    """What is wrong with a title (see TypedSerializable.type_title)."""
    if not isinstance(title, str) or not title or title.strip() != title:
        return "not a word"
    if not (title[0].isupper() or title[0].isdigit()):
        return "not in sentence case"
    if title.endswith('.') or len(title) > 40:
        return "not a short name"
    return None


@pytest.mark.parametrize('name', FAMILIES)
def test_every_type_has_a_title_unique_in_its_family(name):
    titles = {m.type: m.cls.type_title() for m in FAMILIES[name].members}
    problems = {
        type_: problem for type_, title in titles.items()
        if (problem := _title_problem(title))}
    assert problems == {}, titles
    lowered = [title.lower() for title in titles.values()]
    assert len(set(lowered)) == len(lowered), titles


def _choices(type_, where):
    """
    Yield the Literal annotations within an annotation, each with its
    Choices (or None) and where it is.
    """
    origin, args = typing.get_origin(type_), typing.get_args(type_)
    if origin is typing.Annotated:
        inner, *metadata = args
        choices = next((x for x in metadata if isinstance(x, Choices)), None)
        if typing.get_origin(inner) is typing.Literal:
            yield inner, choices, where
            return
        yield from _choices(inner, where)
        return
    if origin is typing.Literal:
        yield type_, None, where
        return
    for arg in args:
        if not isinstance(arg, (str, int, float)) and arg is not Ellipsis:
            yield from _choices(arg, where)


def _options_with_choices():
    """
    Yield the Literal options of the constructors that configurations
    call, each once (an option can be in several constructors of a class,
    e.g. its __init__ and from_file), with its Choices and where it is.
    """
    classes = [*CLASSES, *(
        member.cls for family in FAMILIES.values()
        for member in family.members)]
    seen = set()
    for cls in classes:
        for name in cls.config_constructors:
            func = cls.__init__ if name == '__init__' else getattr(cls, name)
            if func is object.__init__:
                continue
            for option, type_ in option_types(func).items():
                for literal, choices, where in _choices(
                        type_, f'{cls.__qualname__}.{option}'):
                    if (where, literal) not in seen:
                        seen.add((where, literal))
                        yield literal, choices, where


@pytest.mark.parametrize(
    'literal, choices, where',
    [pytest.param(*x, id=x[2]) for x in _options_with_choices()])
def test_every_choice_of_an_option_has_a_title(literal, choices, where):
    assert choices is not None, f"{where}: no titles of its choices"
    assert {key for key, _ in choices.titles} == set(
        typing.get_args(literal))
    for _, title in choices.titles:
        assert _title_problem(title) is None, (where, title)


def _declarations():
    """
    Yield the calls of the library that declare parameters (of
    ParamScalarDesc or ParamVectorDesc), each with where it is.
    """
    for path in sorted(SOURCES.rglob('*.py')):
        if path.name == 'pdescs.py':
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(
                    node.func, 'id', getattr(node.func, 'attr', None)) in (
                    'ParamScalarDesc', 'ParamVectorDesc'):
                where = f'{path.relative_to(SOURCES)}:{node.lineno}'
                yield node, where


@pytest.mark.parametrize(
    'node, where', [pytest.param(*x, id=x[1]) for x in _declarations()])
def test_every_parameter_has_a_title_and_a_description(node, where):
    keywords = {k.arg: k.value for k in node.keywords}
    # (desc is the first positional of a scalar, after the size of a
    # vector)
    position = 0 if getattr(
        node.func, 'id', getattr(node.func, 'attr', None)) \
        == 'ParamScalarDesc' else 1
    desc = keywords.get('desc') or (
        node.args[position] if len(node.args) > position else None)
    title = keywords.get('title')
    assert title is not None, f"{where}: no title"
    assert desc is not None, f"{where}: no description"
    if isinstance(title, ast.Constant):
        assert _title_problem(title.value) is None, (where, title.value)
