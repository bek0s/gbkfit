"""
Tests for the parsing of configurations.
"""

import pytest

from gbkfit.utils import parseutils


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
    dict(step='a')])
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
