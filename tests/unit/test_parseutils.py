"""
Tests for the parsing of configurations.
"""

import typing
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pytest

from gbkfit.utils import parseutils, typeutils


DIMENSIONAL = dict(size=int, step=float)


@pytest.mark.parametrize('info, sanitized', [
    (dict(size=4, step=0.5), dict(size=[4, 4, 4], step=[0.5, 0.5, 0.5])),
    (dict(size=[1, 2, 3]), dict(size=[1, 2, 3])),
    (dict(size=[1, 2, 3, 4]), dict(size=[1, 2, 3])),
    (dict(size=None), dict(size=None)),
    (dict(), dict())])
def test_sanitize_dimensional_options(info, sanitized):
    parseutils.sanitize_dimensional_options(info, DIMENSIONAL, 3)
    assert info == sanitized


@pytest.mark.parametrize('info', [
    dict(size=[1, 2]),
    dict(size=1.5),
    dict(step='a'),
    # A bool is an int in Python, but not a number in a configuration
    dict(size=True),
    dict(size=[True, 2, 3])])
def test_sanitize_dimensional_options_errors(info):
    with pytest.raises(
            parseutils.ConfigError, match=f"option '{next(iter(info))}'"):
        parseutils.sanitize_dimensional_options(info, DIMENSIONAL, 3)


def test_config_error_paths():
    # Each level adds its part of the path; only the innermost type of
    # the part with the error describes it
    with pytest.raises(parseutils.ConfigError) as error:
        with parseutils.config_path('models'):
            with parseutils.config_path(0, context='outer'):
                with parseutils.config_path('traits'):
                    with parseutils.config_path(1, context='inner'):
                        raise ValueError("a bad value")
    assert str(error.value) == "models[0].traits[1] [inner]: a bad value"


def test_config_error_context_is_for_the_part_itself():
    # An error in an option of a part is not described by the part's type
    with pytest.raises(parseutils.ConfigError) as error:
        with parseutils.config_path('component', context='smdisk'):
            with parseutils.config_path('traits'):
                raise parseutils.ConfigError("unknown type 'x'")
    assert str(error.value) == "component.traits: unknown type 'x'"


def test_describe_unknown():
    assert parseutils.describe_unknown(
        ['rnmx', 'zzz'], ['rnmin', 'rnmax']) == (
        "'rnmx' (did you mean 'rnmax'?), 'zzz'")


def test_unknown_options_are_warnings_or_errors_in_strict_mode(caplog):
    def parse():
        return parseutils.parse_options(
            dict(rnmax=1, rnmx=2), optional={'rnmin', 'rnmax'})
    assert parse() == dict(rnmax=1)
    assert "'rnmx' (did you mean 'rnmax'?)" in caplog.text
    with parseutils.strict_mode():
        with pytest.raises(parseutils.ConfigError, match="'rnmx'"):
            parse()
    assert parse() == dict(rnmax=1)


class Shape(parseutils.TypedSerializable):

    def dump(self):
        return {}


class Circle(Shape):

    @staticmethod
    def type():
        return 'circle'

    def __init__(self, radius: float):
        self.radius = radius


shape_parser = parseutils.TypedParser(Shape, Circle)


def test_unknown_types_suggest_similar_ones():
    with pytest.raises(
            parseutils.ConfigError,
            match=r"unknown type 'circel' \(did you mean 'circle'\?\)"):
        shape_parser.load(dict(type='circel'))


def test_options_must_match_the_annotations():
    with pytest.raises(parseutils.ConfigError) as error:
        shape_parser.load(dict(type='circle', radius='a'))
    assert str(error.value) == (
        "circle: option 'radius' must be of type float; it is 'a'")


def ellipse(centre, axes: Sequence[float], angle: float = 0, colour=None):
    pass


def test_options_for_callable():
    # Parameters ignored (centre), renamed (axes) or not, and options
    # added (required, typed or not, and optional)
    def parse(**info):
        return parseutils.parse_options_for_callable(
            info, ellipse, ignore_params=['centre'],
            rename_params=dict(axes='ab'), add_required=dict(shape=str),
            add_optional=dict(label=None))
    assert parse(ab=[2, 1], shape='e', label=3) == dict(
        axes=[2, 1], shape='e', label=3)
    with pytest.raises(parseutils.ConfigError, match="'shape'"):
        parse(ab=[2, 1])
    with pytest.raises(parseutils.ConfigError, match="'shape' must be of"):
        parse(ab=[2, 1], shape=5)
    with pytest.raises(parseutils.ConfigError, match="'ab' must be of"):
        parse(ab=2, shape='e')


@pytest.mark.parametrize('options, message', [
    (dict(ignore_params=['radius']), "has no parameters 'radius'"),
    (dict(rename_params=dict(radius='r')), "has no parameters 'radius'"),
    (dict(ignore_params=['axes'], rename_params=dict(axes='ab')),
     "both ignored and renamed"),
    (dict(rename_params=dict(axes='a', angle='a')), "to the same name"),
    (dict(rename_params=dict(axes='angle')), "with the new names 'angle'"),
    (dict(add_required=dict(x=int), add_optional=dict(x=int)),
     "both required and optional"),
    (dict(add_optional=dict(angle=float)), "names of the parameters"),
    (dict(rename_params=dict(axes='ab'), add_optional=dict(ab=None)),
     "names of the parameters")])
def test_options_for_callable_must_fit_its_parameters(options, message):
    with pytest.raises(RuntimeError, match=message):
        parseutils.parse_options_for_callable({}, ellipse, **options)


def test_warnings_have_the_path_of_the_configuration(caplog):
    with parseutils.config_path('shapes'):
        shape_parser.load([dict(type='circle', radius=1, colour='red')])
    assert "shapes[0] [circle]: unknown options: 'colour'" in caplog.text


@pytest.mark.parametrize('names, prefix, prefix_first, prefixes', [
    ([None, None, None], 'cmp', False, ['', 'cmp1_', 'cmp2_']),
    ([None, None], 'ocmp', True, ['ocmp_', 'ocmp1_']),
    (['disk', 'bulge'], 'cmp', False, ['disk_', 'bulge_']),
    ([], 'cmp', False, [])])
def test_item_prefixes(names, prefix, prefix_first, prefixes):
    # Names, if all items have one; positions, if none has
    assert parseutils.item_prefixes(
        names, 'items', prefix, prefix_first) == prefixes


@pytest.mark.parametrize('names, message', [
    (['disk', None], "either all or none of the components must have a name"),
    (['disk', 'disk'],
     r"the names of the components must be unique; repeated: \['disk'\]")])
def test_item_prefixes_errors(names, message):
    with pytest.raises(parseutils.ConfigError, match=message):
        parseutils.item_prefixes(names, 'components', 'cmp', False)


@pytest.mark.parametrize('name', ['disk', 'Disk_2', '_x', '2'])
def test_valid_names(name):
    parseutils.check_name(name)


@pytest.mark.parametrize('name', ['', 'my disk', 'disk-1', 'disk.x', 3])
def test_invalid_names(name):
    with pytest.raises(parseutils.ConfigError, match="invalid name"):
        parseutils.check_name(name)


class Thing:
    pass


@pytest.mark.parametrize('value, type_, matches', [
    # Bools are not numbers, ints are floats, and numbers of other types
    # (e.g. numpy) are numbers
    (True, bool, True), (True, int, False), (True, float, False),
    (True, int | None, False), (2, float, True), (2.5, int, False),
    (np.float32(1.5), float, True), (np.int64(3), float, True),
    (np.float64(1.5), int | None, False), (np.bool_(True), int, False),
    # Strings are not sequences of strings
    (['disk'], Sequence[str], True), ('disk', Sequence[str], False),
    ('disk', str | Sequence[str], True),
    # Sequences, tuples of any length or of given types, sets, mappings
    ([1, 2.5], Sequence[float], True), ([1, 'a'], Sequence[float], False),
    ((1, 2, 3), tuple[int, ...], True), ((1, 'a'), tuple[int, ...], False),
    ((1, 'a'), tuple[int, str], True), ((1, 2, 3), tuple[int, int], False),
    ([1, 2], tuple[int, int], False), ({1, 2}, set[int], True),
    ([1, 2], set[int], False), ({'a': 1}, Mapping[str, int], True),
    ({'a': 'b'}, Mapping[str, int], False),
    # Nested, Literal and other classes
    ([[1, 2], [3.5]], Sequence[Sequence[float]], True),
    ([[1, 2], 3], Sequence[Sequence[float]], False),
    ({'a': [(1, 2)]}, dict[str, list[tuple[int, int]]], True),
    ({'a': [(1, 'b')]}, dict[str, list[tuple[int, int]]], False),
    ('fast', Literal['fast', 'slow'], True),
    ('quick', Literal['fast', 'slow'], False),
    (Thing(), Thing, True), (None, Thing | None, True), ('x', Thing, False)])
def test_matches_type(value, type_, matches):
    assert typeutils.matches_type(value, type_) == matches


@pytest.mark.parametrize('type_', [5, typing.Self])
def test_matches_type_of_unrecognized_types(type_):
    with pytest.raises(TypeError, match="type is not recognized"):
        typeutils.matches_type(1, type_)


@pytest.mark.parametrize('type_, label', [
    (int, 'int'), (Sequence[int] | None, 'Sequence[int] | None'),
    (Mapping[str, float], 'Mapping[str, float]')])
def test_describe_type(type_, label):
    assert typeutils.describe_type(type_) == label
