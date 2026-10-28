"""
Tests for the parsing of configurations.
"""

import re
import typing
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

import astropy.units as u
import numpy as np
import pytest
from annotated_types import Gt, Le

from gbkfit.utils import parseutils, typeutils
from gbkfit.utils.typeutils import FitsFile, Unit


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


def test_unknown_options_are_errors_or_warnings_in_lenient_mode(caplog):
    def parse():
        return parseutils.parse_options(
            dict(rnmax=1, rnmx=2), optional={'rnmin', 'rnmax'})
    with pytest.raises(
            parseutils.UnknownNamesError,
            match=r"'rnmx' \(did you mean 'rnmax'\?\)"):
        parse()
    with parseutils.strict_mode(False):
        assert parse() == dict(rnmax=1)
    assert "'rnmx' (did you mean 'rnmax'?); they are ignored" in caplog.text


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


class Square(Shape):
    # (of a module that is not installed)
    requires = {'missing_package': "pip install missing_package"}

    @staticmethod
    def type():
        return 'square'

    def __init__(self, side: float):
        self.side = side


def test_types_of_missing_modules_say_what_to_install():
    # A type whose class needs a missing module is listed, and its options
    # can be read, but asking for it says what to install
    parser = parseutils.TypedParser(Shape, [Circle, Square])
    assert parser.types() == ['circle', 'square']
    assert parseutils.missing_modules(Square) == {
        'missing_package': "pip install missing_package"}
    assert parseutils.missing_modules(Circle) == {}
    with pytest.raises(
            parseutils.ConfigError,
            match="'square' needs 'missing_package': pip install"):
        parser.load(dict(type='square', side=1))


def test_options_must_match_the_annotations():
    with pytest.raises(parseutils.ConfigError) as error:
        shape_parser.load(dict(type='circle', radius='a'))
    assert str(error.value) == (
        "circle: option 'radius' must be of type float; it is 'a'")


def ellipse(centre, axes: Sequence[float], angle: float = 0, colour=None):
    pass


def test_options_for_callable():
    # The parameters ignored (centre) are not options
    def parse(**info):
        return parseutils.parse_options_for_callable(
            info, ellipse, ignore_params=['centre'])
    assert parse(axes=[2, 1], angle=3) == dict(axes=[2, 1], angle=3)
    with pytest.raises(parseutils.ConfigError, match="'axes'"):
        parse(angle=3)
    with pytest.raises(parseutils.ConfigError, match="'axes' must be of"):
        parse(axes=2)
    with pytest.raises(RuntimeError, match="has no parameters 'radius'"):
        parseutils.parse_options_for_callable(
            {}, ellipse, ignore_params=['radius'])


def test_warnings_have_the_path_of_the_configuration(caplog):
    with parseutils.strict_mode(False), parseutils.config_path('shapes'):
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


class Beam(parseutils.Serializable):
    """A beam given by its axes, or by a file."""

    config_constructors = ('__init__', 'from_file')

    def __init__(
            self,
            bmaj: Annotated[float, Gt(0), Unit('arcsec')],
            ratio: Annotated[float, Gt(0), Le(1)] = 1.0,
            shape: Literal['gauss', 'airy'] = 'gauss'
    ):
        parseutils.check_options(Beam.__init__, locals())
        self.bmaj = bmaj

    @classmethod
    def from_file(
            cls, file: FitsFile, scale: u.Quantity | None = None
    ) -> 'Beam':
        parseutils.check_options(Beam.from_file, locals())
        beam = cls(1.0)
        beam.file, beam.scale = file, scale
        return beam

    def dump(self):
        return {}


@pytest.mark.parametrize('kwargs, message', [
    (dict(bmaj=-1), "option 'bmaj' must be greater than 0; it is -1"),
    (dict(bmaj=1, ratio=1.5),
     "option 'ratio' must be greater than 0 and at most 1; it is 1.5"),
    (dict(bmaj=1, shape='box'),
     "option 'shape' must be one of 'gauss', 'airy'; it is 'box'"),
    (dict(bmaj='1'), "option 'bmaj' must be of type float; it is '1'")])
def test_python_calls_are_checked(kwargs, message):
    with pytest.raises(parseutils.ConfigError, match=re.escape(message)):
        Beam(**kwargs)
    with pytest.raises(parseutils.ConfigError, match=re.escape(message)):
        Beam.load(kwargs)


def test_numpy_values_are_checked_as_python_values():
    assert Beam(np.float64(2)).bmaj == 2


def test_configurations_call_the_constructor_of_their_options(tmp_path):
    assert Beam.load(dict(bmaj=2)).bmaj == 2
    with parseutils.paths_relative_to(str(tmp_path)):
        beam = Beam.load(dict(
            file=dict(file='beam.fits', hdu=1),
            scale=dict(value=2, unit='arcsec')))
    assert beam.file == (str(tmp_path / 'beam.fits'), 1)
    assert beam.scale == 2 * u.arcsec


@pytest.mark.parametrize('info, message', [
    (dict(bmaj=2, file='beam.fits'),
     "it has those of several: 'bmaj', 'file'"),
    (dict(), "it has none of them")])
def test_configurations_have_the_options_of_one_constructor(info, message):
    with pytest.raises(parseutils.ConfigError, match=message):
        Beam.load(info)
