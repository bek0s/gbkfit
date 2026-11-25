"""
Tests of the JSON Schema of configurations, written from the signatures
of their constructors.
"""

import json
import pathlib

import astropy.units
import jsonschema
import pytest
import ruamel.yaml

import gbkfit
from gbkfit.config import (
    Column, DataFile, TypedParser, TypedSerializable, start_config,
    strict_mode)
from gbkfit.config import Family, Member
from gbkfit.config.schema import SchemaWriter
from gbkfit.instrument import PSF, PSFGauss
from gbkfit.model import Galaxy
from gbkfit.objective import FittedNoise


def test_members_of_families_have_their_options():
    writer = SchemaWriter()
    assert writer.option(PSF | None) == {
        'anyOf': [{'$ref': '#/$defs/psf'}, {'type': 'null'}]}
    # (a member is named by the title of its type)
    gauss = next(
        x for x in writer.defs()['psf']['oneOf']
        if x['title'] == PSFGauss.type_title())
    # (a member is described by the first paragraph of the docstring of
    # its class, in a line)
    assert gauss['description'] == (
        'A Gaussian, round or elliptical: by its sigma, its full width at '
        'half maximum, or those of its axes, as of radio beams (also of '
        'the header of a file).')
    # (a member of several forms has one for each of its constructors,
    # those named described by their docstrings: e.g. a beam by a file)
    by_sigma, by_fwhm, by_beam, by_file = gauss['oneOf']
    assert by_sigma['required'] == ['type', 'sigma']
    assert by_sigma['additionalProperties'] is False
    assert by_sigma['properties']['type'] == {'const': 'gauss'}
    assert 'description' not in by_sigma
    assert by_fwhm['required'] == ['type', 'fwhm']
    assert by_beam['required'] == ['type', 'bmaj', 'bmin']
    assert by_file['description'].startswith('Read the beam of a file')
    # (a number, or a table of the values along the spectral axis; see
    # PSF.varying)
    sigma = by_sigma['properties']['sigma']
    number, table = sigma['anyOf']
    assert (number, table) == (
        {'type': 'number', 'exclusiveMinimum': 0},
        {'$ref': '#/$defs/SpectralTable'})
    assert sigma['unit'] == 'arcsec'
    assert sigma['description'].startswith('The standard deviation')
    assert by_sigma['properties']['ratio']['default'] == 1.0


def test_lists_of_a_length_have_their_bounds():
    # (one value for both axes, or one for each: their range, and the
    # length of the list; see gbkfit.grid.Size)
    from gbkfit.grid import Size
    from gbkfit.observation.sky._options import Oversampling3
    writer = SchemaWriter()
    one, each = writer.option(Size)['anyOf']
    assert one == {'type': 'integer', 'minimum': 1}
    assert each == {
        'type': 'array', 'items': {'type': 'integer', 'minimum': 1},
        'minItems': 2, 'maxItems': 2}
    assert writer.option(Oversampling3) == {
        'type': 'array', 'items': {'type': 'integer', 'minimum': 1},
        'minItems': 3, 'maxItems': 3}


def test_options_have_titles_and_say_what_they_name():
    writer = SchemaWriter()
    writer.option(Galaxy)
    traced = next(
        x for x in writer.defs()['mass_component']['oneOf']
        if x['title'] == 'Traced spheroid')
    # (the component that it follows is one of those of its galaxy)
    component = traced['properties']['component']
    assert component['title'] == 'Component'
    assert component['references'] == 'components'
    assert 'references' not in traced['properties']['name']


def test_options_have_the_forms_of_configurations():
    writer = SchemaWriter()
    fits = writer.option(DataFile)
    assert fits['anyOf'][0] == {'type': 'string', 'format': 'path'}
    assert fits['anyOf'][1]['required'] == ['file']
    assert writer.option(astropy.units.Quantity)['required'] == [
        'value', 'unit']
    column = writer.option(Column)
    assert (column['required'], list(column['properties'])) == (
        ['column'], ['column', 'unit'])
    noise = writer.option(dict[str, FittedNoise])
    assert noise['additionalProperties']['properties']['scale_min'][
        'anyOf'][0] == {'type': 'number', 'exclusiveMinimum': 0}


class Shape(TypedSerializable):

    def dump(self):
        return {}


class Square(Shape):

    @staticmethod
    def type():
        return 'square'

    @staticmethod
    def type_title():
        return 'Square'

    def __init__(self, side: float):
        self.side = side


def test_members_whose_modules_are_missing_say_what_to_install():
    family = Family('shape', TypedParser(Shape, Square), (
        Member('square', Square, {'missing': 'pip install missing'}),))
    square, = SchemaWriter()._family(family)['oneOf']
    assert (square['available'], square['requires']) == (
        False, 'pip install missing')
    assert list(square['properties']) == ['type', 'side']


def test_the_schema_has_the_sections_of_configurations():
    schema = gbkfit.schema()
    json.dumps(schema)
    assert schema['required'] == ['galaxies', 'observations', 'params']
    assert schema['properties']['fitter'] == {
        '$ref': '#/$defs/fitter', 'description': 'The fitter.'}
    galaxy = schema['$defs']['Galaxy_with_driver']['properties']
    assert galaxy['driver'] == {'type': 'string'}


GOLDEN_DIR = pathlib.Path(__file__).parents[1] / 'data' / 'golden'


@pytest.fixture(scope='module')
def validator():
    return jsonschema.Draft202012Validator(gbkfit.schema())


@pytest.mark.parametrize(
    'path', sorted(GOLDEN_DIR.glob('*/config.yaml')),
    ids=lambda path: path.parent.name)
def test_golden_configurations_are_valid(validator, path):
    # (the configurations that the golden cases dump, of every kind of
    # option: the schema must describe what the loader reads)
    config = ruamel.yaml.YAML(typ='safe').load(path)
    errors = [
        f"{'/'.join(map(str, e.absolute_path))}: {e.message[:200]}"
        for e in validator.iter_errors(config)]
    assert not errors


def test_the_schema_takes_the_section_root(validator):
    # (which load reads, and editors that validate against the schema
    # flagged)
    path = sorted(GOLDEN_DIR.glob('*/config.yaml'))[0]
    config = ruamel.yaml.YAML(typ='safe').load(path)
    assert not list(validator.iter_errors(config | dict(root='data')))
    assert list(validator.iter_errors(config | dict(root=1)))


def test_classes_have_the_schemas_of_their_configurations():
    # (a family's have the type, as its parser reads them; a member's are
    # its own options, as its load reads them)
    psf = dict(type='gauss', sigma=0.6)
    jsonschema.validate(psf, PSF.schema())
    jsonschema.validate(dict(sigma=0.6), PSFGauss.schema())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(psf, PSFGauss.schema())
    # A galaxy alone, with a warp and ties
    path = GOLDEN_DIR / 'params_features' / 'config.yaml'
    galaxy, = ruamel.yaml.YAML(typ='safe').load(path)['galaxies']
    jsonschema.validate(galaxy, Galaxy.schema())


def _families_of(schema):
    """Return the names of the families that a schema refers to."""
    families = gbkfit.catalogue()
    defs, names, todo = schema['$defs'], set(), [schema]
    while todo:
        node = todo.pop()
        if isinstance(node, list):
            todo.extend(node)
        elif isinstance(node, dict):
            if (ref := node.get('$ref')) and ref.split('/')[-1] not in names:
                names.add(name := ref.split('/')[-1])
                todo.append(defs[name])
            todo.extend(node.values())
    return [name for name in sorted(names) if name in families]


def _is_file(option):
    """Whether an option is a file (or one of the forms of a file)."""
    return any(
        x.get('format') == 'path' for x in [option, *option.get('anyOf', [])])


GALAXY_MEMBERS = [
    (name, member) for name in _families_of(Galaxy.schema())
    for member in gbkfit.catalogue()[name].members]


@pytest.mark.parametrize(
    'name, member', GALAXY_MEMBERS,
    ids=[f'{name}-{member.type}' for name, member in GALAXY_MEMBERS])
def test_a_new_member_of_a_family_of_a_galaxy_starts_valid(name, member):
    start = start_config(member.cls)
    schema = SchemaWriter().constructors(member.cls)
    assert schema['examples'] == [start]
    assert start['type'] == member.type
    # (it has every option it requires, but those that only a
    # configuration can give: its name, the names of other things, and a
    # file with what is in it; with the names, it loads)
    for form in schema.get('oneOf', [schema]):
        lacks = {
            key: form['properties'][key] for key in form['required']
            if key not in start}
        if any(_is_file(option) for option in lacks.values()):
            return
        if all(key == 'name' or 'references' in option
               for key, option in lacks.items()):
            break
    else:
        pytest.fail(f"{start} lacks options it requires")
    with strict_mode():
        gbkfit.catalogue()[name].parser.load(
            start | dict.fromkeys(lacks, 'other'))


@pytest.mark.parametrize('path, keyword, value', [
    # (the grids on the sky share the types of those of observations)
    (('dataset', 'pixel_brightness', 'step'), 'exclusiveMinimum', 0),
    (('dataset', 'pixel_brightness', 'rval'), 'maxItems', 2),
    (('regions', 'bins', 'step'), 'exclusiveMinimum', 0),
    (('primary_beam', 'image', 'rpix'), 'maxItems', 2),
    (('lens_piece', 'deflection_map', 'step'), 'maxItems', 2),
    (('psf', 'image', 'step'), 'exclusiveMinimum', 0),
    (('dataset', 'pixel_brightness', 'error'), 'exclusiveMinimum', 0),
    (('fitter', 'emcee', 'seed'), 'maximum', 2**32 - 1),
    (('fitter', 'emcee', 'burnin'), 'minimum', 0),
    (('psf', 'sum', 'weights'), 'exclusiveMinimum', 0),
    (('joint_prior', 'smooth2d', 'size'), 'maxItems', 2),
    (('cosmology', 'flat_lambda_cdm', 'omega_matter'), 'maximum', 1),
    (('slit_mask', 'shutters', 'size'), 'minItems', 2),
    (('bootstrap_scheme', 'blocks', 'size'), 'minItems', 1),
    (('component', 'thin_disk', 'rstep'), 'exclusiveMinimum', 0),
    (('component', 'thin_disk', 'rnodes'), 'minItems', 2)])
def test_ranges_and_lengths_reach_the_schema(path, keyword, value):
    # (somewhere in the schema of the option: of its numbers, or of its
    # items)
    family, member, option = path
    schema = gbkfit.schema()
    node = next(
        m for m in schema['$defs'][family]['oneOf']
        if m.get('properties', {}).get('type', {}).get('const') == member)
    found = json.dumps(node['properties'][option])
    assert f'"{keyword}": {json.dumps(value)}' in found
