"""
Helpers for the type annotations of configuration options.

The options of a configuration are checked against the type annotations
of the callables they are passed to (see
`gbkfit.utils.parseutils.parse_options_for_callable`).
"""

import functools
import typing
import warnings

import pydantic
import pydantic.warnings


__all__ = [
    'describe_type',
    'matches_type'
]


@functools.cache
def _adapter(type_: typing.Any) -> pydantic.TypeAdapter:
    """
    Return the strict pydantic validator of a type annotation.

    Raise TypeError if pydantic cannot validate values of the annotation.
    For an annotation that is not a type (e.g. 5), pydantic would instead
    warn and accept any value.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                'error', pydantic.warnings.ArbitraryTypeWarning)
            return pydantic.TypeAdapter(type_, config=pydantic.ConfigDict(
                strict=True, arbitrary_types_allowed=True))
    except (pydantic.PydanticUserError,
            pydantic.warnings.ArbitraryTypeWarning) as e:
        raise TypeError(
            f"the following type is not recognized: "
            f"{describe_type(type_)}") from e


def matches_type(value: typing.Any, type_: typing.Any) -> bool:
    """
    Check whether a configuration value matches a type annotation.

    A bool is not a number, an int is a float, and a str is not a
    sequence of strings.

    Parameters
    ----------
    value : Any
        The value of a configuration option.
    type_ : Any
        A type annotation (e.g. ``Sequence[float] | None``).

    Returns
    -------
    bool
        Whether the value matches the annotation.

    Raises
    ------
    TypeError
        If pydantic cannot validate values of the annotation.
    """
    try:
        _adapter(type_).validate_python(value)
    except pydantic.ValidationError:
        return False
    return True


def describe_type(type_: typing.Any) -> str:
    """
    Return the label of a type annotation, for messages.

    Parameters
    ----------
    type_ : Any
        A type annotation.

    Returns
    -------
    str
        The label: the name of a type, or the annotation without the
        names of its modules.
    """
    name = getattr(type_, '__name__', None)
    label = str(type_) if typing.get_args(type_) or name is None else name
    return (
        label
        .replace('collections.abc.', '')
        .replace('types.', '')
        .replace('typing.', ''))
