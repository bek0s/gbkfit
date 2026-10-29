"""
The JSON Schema (2020-12) of the configurations of GBKFIT, written from
the signatures of the constructors that configurations call (see
Serializable.config_constructors): their parameters are the options,
with the types, ranges and choices of their annotations, their units
(Unit; the keyword 'unit'), their defaults, and the descriptions of their
docstrings.

The options of a family (e.g. a PSF) are one of its members, chosen by
their type (see catalogue); a member that needs modules that are not
installed has the keywords 'available' (false) and 'requires' (what to
install), so that the schema is the same on every machine.
"""

import collections.abc
import inspect
import re
import types
import typing
from collections.abc import Callable, Mapping
from typing import Annotated, Any, Literal

import annotated_types
import astropy.units
import numpy as np
import numpy.typing as npt

from gbkfit._catalogue import Family, family_of
from gbkfit.utils import parseutils, typeutils


__all__ = [
    'SchemaWriter',
    'class_schema',
    'schema'
]


# The dialect of the schemas
_DIALECT = 'https://json-schema.org/draft/2020-12/schema'

# The keywords of JSON Schema of the bounds of a range
_BOUNDS = {
    annotated_types.Gt: ('exclusiveMinimum', 'gt'),
    annotated_types.Ge: ('minimum', 'ge'),
    annotated_types.Lt: ('exclusiveMaximum', 'lt'),
    annotated_types.Le: ('maximum', 'le')}

# The types of units, which configurations give as strings (e.g. 'km/s')
_UNITS = (astropy.units.UnitBase, astropy.units.FunctionUnitBase)

# The schemas of the plain types
_PLAIN = {
    bool: {'type': 'boolean'},
    int: {'type': 'integer'},
    float: {'type': 'number'},
    str: {'type': 'string'},
    type(None): {'type': 'null'},
    Any: {}}


def schema() -> dict[str, Any]:
    """
    Return the JSON Schema (2020-12) of the configurations of GBKFIT.

    It is written from the signatures of the constructors that
    configurations call: the options of each, with their types, ranges,
    choices, units (the keyword 'unit'), defaults and descriptions. The
    members of families that need modules that are not installed are in
    it too, with the keywords 'available' (false) and 'requires' (what to
    install): it is the same on every machine. Classes have the schemas
    of their own configurations (see Serializable.schema).

    Returns
    -------
    dict
        The schema.
    """
    # (setup imports the whole library)
    from gbkfit.setup import _SECTIONS
    writer = SchemaWriter()
    properties = {}
    for name, section in _SECTIONS.items():
        value = writer.option(section.type)
        if section.item_options:
            value = writer.with_options(value, section.item_options)
        if section.listed:
            value = {'anyOf': [value, {'type': 'array', 'items': value}]}
        properties[name] = value | {'description': section.desc}
    required = [k for k, v in _SECTIONS.items() if v.required]
    return {
        '$schema': _DIALECT,
        'title': 'GBKFIT configuration',
        **_object(properties, required),
        '$defs': writer.defs()}


def class_schema(cls: type[parseutils.Serializable]) -> dict[str, Any]:
    """
    Return the JSON Schema (2020-12) of the configurations of a class (see
    Serializable.schema).

    Raises
    ------
    TypeError
        If the class has no configurations.
    """
    writer = SchemaWriter()
    return {
        '$schema': _DIALECT,
        'title': cls.__qualname__,
        **writer.option(cls),
        '$defs': writer.defs()}


class SchemaWriter:
    """
    The writer of the schemas of options and classes. The schemas of the
    families and the classes that options name are written once, into its
    definitions ($defs), which their options refer to ($ref).
    """

    def __init__(self):
        self._defs: dict[str, dict[str, Any]] = {}

    def defs(self) -> dict[str, dict[str, Any]]:
        """Return the definitions written so far, by name."""
        return dict(self._defs)

    def option(self, type_: Any) -> dict[str, Any]:
        """
        Return the schema of the values of an option of a type annotation,
        as configurations give them (see parseutils.from_config).

        Raises
        ------
        TypeError
            If the annotation has no form in configurations (e.g. a
            callable).
        """
        if type_ in _PLAIN:
            return dict(_PLAIN[type_])
        if type_ == typeutils.FilePath:
            return {'type': 'string', 'format': 'path'}
        if type_ == typeutils.FitsFile:
            return {'anyOf': [
                {'type': 'string', 'format': 'path'},
                _object(
                    {'file': {'type': 'string', 'format': 'path'},
                     'hdu': {'type': ['integer', 'string']}}, ['file'])]}
        if type_ == npt.ArrayLike:
            return {'type': ['number', 'array']}
        if type_ is np.ndarray:
            return {'type': 'array'}
        if type_ in _UNITS:
            return {'type': 'string', 'format': 'unit'}
        origin = typing.get_origin(type_)
        args = typing.get_args(type_)
        if origin is Annotated:
            return self._annotated(type_)
        if origin in (typing.Union, types.UnionType):
            members = []
            for schema in (self.option(x) for x in args):
                if schema not in members:
                    members.append(schema)
            return members[0] if len(members) == 1 else {'anyOf': members}
        if origin is Literal:
            return {'enum': list(args)}
        if origin in (list, tuple, collections.abc.Sequence):
            return self._sequence(origin, args)
        if origin in (dict, collections.abc.Mapping):
            return {'type': 'object',
                    'additionalProperties': self.option(args[1])}
        if type_ is astropy.units.Quantity:
            return _object({
                'value': {'type': ['number', 'array']},
                'unit': {'type': 'string'}}, ['value', 'unit'])
        if isinstance(type_, type) and parseutils.is_record(type_):
            return self._record(type_)
        if isinstance(type_, type) and (family := family_of(type_)):
            return self._ref(family.name, lambda: self._family(family))
        if isinstance(type_, type) \
                and issubclass(type_, parseutils.Serializable) \
                and not inspect.isabstract(type_):
            return self._ref(
                type_.__qualname__, lambda: self.constructors(type_))
        raise TypeError(
            f"options of the type {typeutils.describe_type(type_)} have no "
            f"form in configurations")

    def constructors(
            self,
            cls: type[parseutils.Serializable]
    ) -> dict[str, Any]:
        """
        Return the schema of the configurations of a class: the options
        of one of its config_constructors (see config_constructor).
        """
        forms = [
            self._constructor(cls, name) for name in cls.config_constructors]
        return forms[0] if len(forms) == 1 else {'oneOf': forms}

    def with_options(
            self,
            schema: dict[str, Any],
            options: Mapping[str, Any]
    ) -> dict[str, Any]:
        """
        Return a schema (a reference to a definition) with options besides
        its own (e.g. the driver of a galaxy in a configuration): of every
        one of its forms, as a definition of its own.
        """
        name = schema['$ref'].removeprefix('#/$defs/')
        extra = {key: self.option(type_) for key, type_ in options.items()}
        return self._ref(
            f'{name}_with_{"_".join(options)}',
            lambda: _with_properties(self._defs[name], extra))

    def _annotated(self, type_: Any) -> dict[str, Any]:
        """Return the schema of an Annotated annotation."""
        inner, *metadata = typing.get_args(type_)
        schema = self.option(inner)
        for marker in metadata:
            if (bound := _BOUNDS.get(type(marker))) is not None:
                keyword, attribute = bound
                schema[keyword] = getattr(marker, attribute)
        return schema

    def _sequence(self, origin: Any, args: tuple) -> dict[str, Any]:
        """Return the schema of a sequence annotation."""
        if origin is tuple and args and args[-1] is not Ellipsis:
            return {'type': 'array',
                    'prefixItems': [self.option(x) for x in args],
                    'minItems': len(args), 'maxItems': len(args)}
        schema: dict[str, Any] = {'type': 'array'}
        if args:
            schema['items'] = self.option(args[0])
        return schema

    def _record(self, cls: type) -> dict[str, Any]:
        """Return the schema of a record (a NamedTuple) in a configuration."""
        return self._properties(cls, cls, _param_docs(cls, None))

    def _family(self, family: Family) -> dict[str, Any]:
        """Return the schema of the configurations of a family."""
        members = []
        for member in family.members:
            schema = self.constructors(member.cls)
            forms = schema.get('oneOf', [schema])
            for form in forms:
                form['properties'] = {
                    'type': {'const': member.type}} | form['properties']
                form['required'] = ['type', *form['required']]
            if not member.available():
                schema['available'] = False
                schema['requires'] = '; '.join(
                    dict.fromkeys(member.missing.values()))
            members.append(schema | {'title': member.cls.__qualname__})
        return {'oneOf': members}

    def _constructor(
            self,
            cls: type[parseutils.Serializable],
            name: str
    ) -> dict[str, Any]:
        """Return the schema of the options of a config constructor."""
        func = cls.__init__ if name == '__init__' else getattr(cls, name)
        docs = _param_docs(cls, func if name != '__init__' else None)
        return self._properties(func, cls, docs)

    def _properties(
            self,
            func: Callable,
            owner: type,
            docs: dict[str, str]
    ) -> dict[str, Any]:
        """
        Return the schema of an object of the options of the parameters of
        a callable (a constructor, or a record), with their descriptions.
        """
        hints = typeutils.option_types(func)
        params = inspect.signature(func).parameters
        properties, required = {}, []
        others: Any = False
        given = getattr(owner, 'given_options', ())
        for name, param in params.items():
            if name in ('self', 'cls', *given) \
                    or param.kind is param.VAR_POSITIONAL:
                continue
            try:
                schema = self.option(hints.get(name, Any))
            except TypeError as e:
                raise TypeError(
                    f"{owner.__qualname__}, option '{name}': {e}") from e
            if param.kind is param.VAR_KEYWORD:
                others = schema
                continue
            if name in getattr(owner, 'VARYING_OPTIONS', ()):
                # (a PSF or an LSF of options that vary along the spectral
                # axis, see PSF.varying, has tables of them)
                from gbkfit.instrument import SpectralTable
                schema = {'anyOf': [schema, self.option(SpectralTable)]}
            if (unit := typeutils.unit_of(hints.get(name))) is not None:
                schema['unit'] = unit
            if name in docs:
                schema['description'] = docs[name]
            if param.default is param.empty:
                required.append(name)
            elif (default := _json(param.default)) is not _NO_JSON:
                schema['default'] = default
            properties[name] = schema
        # (the options of configurations in other forms, or that are not
        # parameters; see Serializable.config_options)
        for name, type_ in getattr(owner, 'config_options', {}).items():
            kept = {key: value for key, value in properties.get(
                name, {}).items() if key in ('description', 'default', 'unit')}
            properties[name] = self.option(type_) | kept
        return _object(properties, required, others)

    def _ref(self, name: str, write: Callable[[], dict]) -> dict[str, Any]:
        """
        Return a reference to a definition, writing it first if it is not
        written (a definition being written is referred to as it is, e.g.
        by the components of a sum of PSFs).
        """
        if name not in self._defs:
            self._defs[name] = {}
            self._defs[name] = write()
        return {'$ref': f'#/$defs/{name}'}


def _with_properties(
        schema: dict[str, Any],
        extra: dict[str, Any]
) -> dict[str, Any]:
    """
    Return a copy of a schema of objects whose every form (of oneOf) has
    the extra properties.
    """
    if 'oneOf' in schema:
        return schema | {
            'oneOf': [_with_properties(x, extra) for x in schema['oneOf']]}
    return schema | {'properties': schema['properties'] | extra}


def _object(
        properties: dict[str, Any],
        required: list[str],
        others: Any = False
) -> dict[str, Any]:
    """Return the schema of an object of the given properties."""
    return {'type': 'object', 'properties': properties,
            'required': required, 'additionalProperties': others}


# The marker of a default value that has no form in JSON
_NO_JSON = object()


def _json(value: Any) -> Any:
    """Return a value as JSON (tuples as lists), or _NO_JSON."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float | np.floating):
        return float(value) if np.isfinite(value) else _NO_JSON
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, list | tuple):
        items = [_json(x) for x in value]
        return _NO_JSON if _NO_JSON in items else items
    return _NO_JSON


def _param_docs(cls: type, func: Callable | None) -> dict[str, str]:
    """
    Return the descriptions of the parameters of a constructor, by name,
    from its numpydoc docstring (of the class for __init__, func None).
    """
    lines = inspect.cleandoc((func or cls).__doc__ or '').split('\n')
    try:
        start = next(
            i for i in range(len(lines) - 1)
            if lines[i].strip() == 'Parameters'
            and set(lines[i + 1].strip()) == {'-'})
    except StopIteration:
        return {}
    docs: dict[str, str] = {}
    names: list[str] = []
    for i in range(start + 2, len(lines)):
        line = lines[i]
        if line.strip() and not line.startswith(' ') and i + 1 < len(lines) \
                and set(lines[i + 1].strip()) == {'-'}:
            break
        if line and not line.startswith(' '):
            head = line.split(' : ')[0]
            names = [n.strip().lstrip('*') for n in head.split(',')
                     if n.strip() not in ('...', '')]
            for name in names:
                docs[name] = ''
        elif line.strip():
            for name in names:
                docs[name] = f'{docs[name]} {line.strip()}'.strip()
    return {
        name: re.sub(r'\s+', ' ', text) for name, text in docs.items() if text}
