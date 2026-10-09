"""
Tests for the parsing of configurations.
"""

import typing
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pytest

from gbkfit.utils import parseutils, typeutils


DIMENSIONAL = dict(size=int, step=int | float)


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
    with pytest.raises(RuntimeError, match=f"option '{next(iter(info))}'"):
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
            dict(rnmax=1, rnmx=2), 'disk', optional={'rnmin', 'rnmax'})
    assert parse() == dict(rnmax=1)
    assert "'rnmx' (did you mean 'rnmax'?)" in caplog.text
    with parseutils.strict_mode():
        with pytest.raises(parseutils.ConfigError, match="'rnmx'"):
            parse()
    assert parse() == dict(rnmax=1)


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
