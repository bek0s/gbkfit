"""
Tests for the parsing of configurations.
"""

import copy
import re
import typing
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

import astropy.units as u
import numpy as np
import pytest
from annotated_types import Gt, Le

from gbkfit.config import (
    ConfigError, ConfigProblem, DataFile, Serializable, TypedParser,
    TypedSerializable, Unit, UnknownNamesError, check_name, check_options,
    collect_problems, config_path, describe_type, describe_unknown,
    error_location, error_message, item_prefixes, matches_type,
    missing_modules, parse_options, parse_options_for_callable,
    paths_relative_to, strict_mode, type_problem, warn)


def test_config_error_paths():
    # Each level adds its part of the path; only the innermost type of
    # the part with the error describes it
    with pytest.raises(ConfigError) as error:
        with config_path('models'):
            with config_path(0, context='outer'):
                with config_path('traits'):
                    with config_path(1, context='inner'):
                        raise ConfigError("a bad value")
    assert str(error.value) == "models[0].traits[1] [inner]: a bad value"


def test_other_errors_pass_config_paths_with_the_path():
    # Of their own type: each was a ConfigError of its message (that of a
    # KeyError was "x: 'key'"), so that bugs and failed runs read as
    # mistakes of the configuration
    with pytest.raises(KeyError) as error:
        with config_path('models'):
            with config_path(0, context='disk'):
                raise KeyError('key')
    assert error_location(error.value) == (['models', 0], 'disk')
    assert error.value.__notes__ == [
        "in the configuration at models[0] [disk]"]
    assert error_message(error.value) == "models[0] [disk]: KeyError: 'key'"


def test_config_error_context_is_for_the_part_itself():
    # An error in an option of a part is not described by the part's type
    with pytest.raises(ConfigError) as error:
        with config_path('component', context='smdisk'):
            with config_path('traits'):
                raise ConfigError("unknown type 'x'")
    assert str(error.value) == "component.traits: unknown type 'x'"


def test_describe_unknown():
    assert describe_unknown(
        ['rnmx', 'zzz'], ['rnmin', 'rnmax']) == (
        "'rnmx' (did you mean 'rnmax'?), 'zzz'")


def test_unknown_options_are_errors_or_warnings_in_lenient_mode(caplog):
    def parse():
        return parse_options(
            dict(rnmax=1, rnmx=2), optional={'rnmin', 'rnmax'})
    with pytest.raises(
            UnknownNamesError,
            match=r"'rnmx' \(did you mean 'rnmax'\?\)"):
        parse()
    with strict_mode(False):
        assert parse() == dict(rnmax=1)
    assert "'rnmx' (did you mean 'rnmax'?); they are ignored" in caplog.text


class Shape(TypedSerializable):

    def dump(self):
        return {}


class Circle(Shape):

    @staticmethod
    def type():
        return 'circle'

    @staticmethod
    def type_title():
        return 'Circle'

    def __init__(self, radius: float):
        self.radius = radius


shape_parser = TypedParser(Shape, Circle)


def test_unknown_types_suggest_similar_ones():
    with pytest.raises(
            ConfigError,
            match=r"unknown type 'circel' \(did you mean 'circle'\?\)"):
        shape_parser.load(dict(type='circel'))


class Square(Shape):
    # (of a module that is not installed)
    requires = {'missing_package': "pip install missing_package"}

    @staticmethod
    def type():
        return 'square'

    @staticmethod
    def type_title():
        return 'Square'

    def __init__(self, side: float):
        self.side = side


def test_types_of_missing_modules_say_what_to_install():
    # A type whose class needs a missing module is listed, and its options
    # can be read, but asking for it says what to install
    parser = TypedParser(Shape, [Circle, Square])
    assert parser.types() == ['circle', 'square']
    assert missing_modules(Square) == {
        'missing_package': "pip install missing_package"}
    assert missing_modules(Circle) == {}
    with pytest.raises(
            ConfigError,
            match="'square' needs 'missing_package': pip install"):
        parser.load(dict(type='square', side=1))


def test_options_must_match_the_annotations():
    with pytest.raises(ConfigError) as error:
        shape_parser.load(dict(type='circle', radius='a'))
    assert str(error.value) == (
        "circle: option 'radius' must be of type float; it is 'a'")


def ellipse(centre, axes: Sequence[float], angle: float = 0, colour=None):
    pass


def test_options_for_callable():
    # The parameters ignored (centre) are not options
    def parse(**info):
        return parse_options_for_callable(
            info, ellipse, ignore_params=['centre'])
    assert parse(axes=[2, 1], angle=3) == dict(axes=[2, 1], angle=3)
    with pytest.raises(ConfigError, match="'axes'"):
        parse(angle=3)
    with pytest.raises(ConfigError, match="'axes' must be of"):
        parse(axes=2)
    with pytest.raises(RuntimeError, match="has no parameters 'radius'"):
        parse_options_for_callable(
            {}, ellipse, ignore_params=['radius'])


def test_warnings_have_the_path_of_the_configuration(caplog):
    with strict_mode(False), config_path('shapes'):
        shape_parser.load([dict(type='circle', radius=1, colour='red')])
    assert "shapes[0] [circle]: unknown options: 'colour'" in caplog.text


@pytest.mark.parametrize('names, prefix, prefix_first, prefixes', [
    ([None, None, None], 'cmp', False, ['', 'cmp1_', 'cmp2_']),
    ([None, None], 'ocmp', True, ['ocmp_', 'ocmp1_']),
    (['disk', 'bulge'], 'cmp', False, ['disk_', 'bulge_']),
    ([], 'cmp', False, [])])
def test_item_prefixes(names, prefix, prefix_first, prefixes):
    # Names, if all items have one; positions, if none has
    assert item_prefixes(
        names, 'items', prefix, prefix_first) == prefixes


@pytest.mark.parametrize('names, message', [
    (['disk', None], "either all or none of the components must have a name"),
    (['disk', 'disk'],
     r"the names of the components must be unique; repeated: \['disk'\]")])
def test_item_prefixes_errors(names, message):
    with pytest.raises(ValueError, match=message):
        item_prefixes(names, 'components', 'cmp', False)


@pytest.mark.parametrize('name', ['disk', 'Disk_2', '_x', '2'])
def test_valid_names(name):
    check_name(name)


@pytest.mark.parametrize('name, error', [
    ('', ValueError), ('my disk', ValueError), ('disk-1', ValueError),
    ('disk.x', ValueError), (3, TypeError)])
def test_invalid_names(name, error):
    with pytest.raises(error, match="invalid name"):
        check_name(name)


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
    assert matches_type(value, type_) == matches


@pytest.mark.parametrize('type_', [5, typing.Self])
def test_matches_type_of_unrecognized_types(type_):
    with pytest.raises(TypeError, match="type is not recognized"):
        matches_type(1, type_)


@pytest.mark.parametrize('type_, label', [
    (int, 'int'), (Sequence[int] | None, 'Sequence[int] | None'),
    (Mapping[str, float], 'Mapping[str, float]')])
def test_describe_type(type_, label):
    assert describe_type(type_) == label


def test_describe_type_keeps_modules_whose_names_end_in_types():
    Foo = type('Foo', (), {'__module__': 'mytypes'})
    assert describe_type(list[Foo]) == 'list[mytypes.Foo]'


@pytest.mark.parametrize('value, type_, message', [
    ([1, -2], Sequence[Annotated[float, Gt(0)]],
     "item [1] must be greater than 0; it is -2"),
    ([[1, 2], [3, -1]], Sequence[Sequence[Annotated[float, Gt(0)]]],
     "item [1][1] must be greater than 0; it is -1"),
    ((1, 'x'), tuple[int, ...], "item [1] must be of type int; it is 'x'"),
    (1, type(None), "must be of type None; it is 1")])
def test_problems_say_which_item_is_wrong(value, type_, message):
    assert type_problem(value, type_) == message


class Beam(Serializable):
    """A beam given by its axes, or by a file."""

    config_constructors = ('__init__', 'from_file')

    def __init__(
            self,
            bmaj: Annotated[float, Gt(0), Unit('arcsec')],
            ratio: Annotated[float, Gt(0), Le(1)] = 1.0,
            shape: Literal['gauss', 'airy'] = 'gauss'
    ):
        check_options(Beam.__init__, locals())
        self.bmaj = bmaj

    @classmethod
    def from_file(
            cls, file: DataFile, scale: u.Quantity | None = None
    ) -> 'Beam':
        check_options(Beam.from_file, locals())
        beam = cls(1.0)
        beam.file, beam.scale = file, scale
        return beam

    def dump(self):
        return {}


@pytest.mark.parametrize('kwargs, error, message', [
    (dict(bmaj=-1), ValueError,
     "option 'bmaj' must be greater than 0; it is -1"),
    (dict(bmaj=1, ratio=1.5), ValueError,
     "option 'ratio' must be greater than 0 and at most 1; it is 1.5"),
    (dict(bmaj=1, shape='box'), ValueError,
     "option 'shape' must be one of 'gauss', 'airy'; it is 'box'"),
    (dict(bmaj='1'), TypeError,
     "option 'bmaj' must be of type float; it is '1'")])
def test_python_calls_are_checked(kwargs, error, message):
    # Alike: in Python as errors of the arguments, in configurations as
    # mistakes of the configuration
    with pytest.raises(error, match=re.escape(message)):
        Beam(**kwargs)
    with pytest.raises(ConfigError, match=re.escape(message)):
        Beam.load(kwargs)


def test_numpy_values_are_checked_as_python_values():
    assert Beam(np.float64(2)).bmaj == 2


def test_configurations_call_the_constructor_of_their_options(tmp_path):
    assert Beam.load(dict(bmaj=2)).bmaj == 2
    with paths_relative_to(str(tmp_path)):
        beam = Beam.load(dict(
            file=dict(file='beam.fits', hdu=1),
            scale=dict(value=2, unit='arcsec')))
    assert beam.file == (str(tmp_path / 'beam.fits'), 1)
    assert beam.scale == 2 * u.arcsec


def test_options_of_classes_are_loaded_by_their_annotation():
    # The PSF of an instrument (of a family), and the PSFs of a sum (a
    # sequence of them); the configuration is not changed (audit
    # 2026-10-31 #41)
    from gbkfit.instrument import Instrument, PSFSum
    info = dict(psf=dict(type='sum', weights=[1, 1], psfs=[
        dict(type='gauss', sigma=1), dict(type='point')]))
    original = copy.deepcopy(info)
    psf = Instrument.load(info).psf()
    assert isinstance(psf, PSFSum)
    assert [x['type'] for x in psf.dump()['psfs']] == ['gauss', 'point']
    assert info == original
    with pytest.raises(ConfigError) as error:
        Instrument.load(dict(psf=dict(type='sum', weights=[1], psfs=[
            dict(type='gaus', sigma=1)])))
    assert str(error.value).startswith("psf.psfs[0]: unknown type 'gaus'")
    with pytest.raises(ConfigError) as error:
        Instrument.load(dict(psf=dict(type='sum', weights=[1], psfs=[5])))
    assert str(error.value) == "psf.psfs[0]: must be of type PSF; it is 5"


@pytest.mark.parametrize('info, message', [
    (dict(bmaj=2, file='beam.fits'),
     "it has those of several: 'bmaj', 'file'"),
    (dict(), "it has none of them")])
def test_configurations_have_the_options_of_one_constructor(info, message):
    with pytest.raises(ConfigError, match=message):
        Beam.load(info)


def test_problems_have_the_paths_of_the_blocks_around():
    # As warnings have: the error had only the path within the block of
    # collect_problems
    with config_path('galaxies'), collect_problems() as problems:
        with config_path(0):
            warn("a warning")
            raise ConfigError("an error")
    assert problems == [
        ConfigProblem('warning', ['galaxies', 0], "a warning"),
        ConfigProblem('error', ['galaxies', 0], "an error")]


def test_problems_are_errors_of_any_type(caplog):
    # A bug, or an error of a constructor, with its type and its path
    with collect_problems() as problems, config_path('galaxies', 0):
        raise ValueError("rnmax must be greater than rnmin")
    assert problems == [ConfigProblem(
        'error', ['galaxies', 0],
        "ValueError: rnmax must be greater than rnmin")]
    assert "Traceback" in caplog.text


@pytest.mark.parametrize('info', ['gauss', [dict(sigma=1)], 3])
def test_options_must_be_a_dictionary(info):
    # A string was read as the options of its characters
    with pytest.raises(ConfigError, match="must be a dictionary"):
        parse_options(info, optional={'sigma'})


def test_problems_are_collected_per_thread():
    # (two threads that collect at once: each saw the problems of both,
    # audit 2026-10-31 #32)
    import threading
    barrier = threading.Barrier(2)
    seen = {}

    def collect(name):
        with collect_problems() as problems:
            barrier.wait()
            warn(f"warning of {name}")
            barrier.wait()
        seen[name] = [problem.message for problem in problems]
    threads = [threading.Thread(target=collect, args=(n,)) for n in 'ab']
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert seen == {'a': ["warning of a"], 'b': ["warning of b"]}
