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
from gbkfit.config import Column, DataFile, TypedParser, TypedSerializable
from gbkfit.config.catalogue import Family, Member
from gbkfit.config.schema import SchemaWriter
from gbkfit.instrument import PSF, PSFGauss, PSFGaussBeam
from gbkfit.model import Galaxy
from gbkfit.observation import FittedNoise


def test_members_of_families_have_their_options():
    writer = SchemaWriter()
    assert writer.option(PSF | None) == {
        'anyOf': [{'$ref': '#/$defs/psf'}, {'type': 'null'}]}
    # (a member is named by the title of its type)
    gauss = next(
        x for x in writer.defs()['psf']['oneOf']
        if x['title'] == PSFGauss.type_title())
    assert gauss['required'] == ['type', 'sigma']
    assert gauss['additionalProperties'] is False
    assert gauss['properties']['type'] == {'const': 'gauss'}
    # (a number, or a table of the values along the spectral axis; see
    # PSF.varying)
    sigma = gauss['properties']['sigma']
    number, table = sigma['anyOf']
    assert (number, table) == (
        {'type': 'number', 'exclusiveMinimum': 0},
        {'$ref': '#/$defs/SpectralTable'})
    assert sigma['unit'] == 'arcsec'
    assert sigma['description'].startswith('The standard deviation')
    assert gauss['properties']['ratio']['default'] == 1.0
    # (a member is described by the first paragraph of the docstring of
    # its class, in a line)
    assert gauss['description'] == (
        'A Gaussian, round or elliptical, by its sigma, axis ratio and '
        'angle.')
    # (and each form of a member of several by its named constructor:
    # e.g. a beam by its axes, or by a file, from_file)
    beam = next(
        x for x in writer.defs()['psf']['oneOf']
        if x['title'] == PSFGaussBeam.type_title())
    axes, file = beam['oneOf']
    assert 'description' not in axes
    assert file['description'].startswith('Read a beam from the header')


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
