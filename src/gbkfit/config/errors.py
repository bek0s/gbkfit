"""
The errors and warnings of configurations, each with the path to the
part of the configuration that has it (see ConfigError), and strict and
lenient modes for unknown options.
"""

import contextlib
import contextvars
import difflib
import logging
from collections.abc import Generator, Iterable, Sequence
from typing import Literal, NamedTuple


__all__ = [
    'ConfigError',
    'Problem',
    'UnknownNamesError',
    'collect_problems',
    'config_path',
    'describe_unknown',
    'is_strict',
    'report_unknown',
    'strict_mode',
    'warn'
]


_log = logging.getLogger(__name__)


# Whether unknown options are errors (strict, the default) or warnings
# (lenient)
_strict = contextvars.ContextVar('strict', default=True)


@contextlib.contextmanager
def strict_mode(strict: bool = True) -> Generator[None]:
    """
    Make unknown options errors (strict, the default) or warnings
    (lenient), in the block.

    This also applies to unknown sections and parameters. Lenient mode
    leaves them out with a warning, e.g. to load a configuration with an
    option of a newer version of GBKFIT.

    Parameters
    ----------
    strict : bool, optional
        Whether unknown options are errors.
    """
    token = _strict.set(strict)
    try:
        yield
    finally:
        _strict.reset(token)


def is_strict() -> bool:
    """
    Check whether unknown options are errors (see strict_mode).

    Returns
    -------
    bool
        Whether unknown options are errors.
    """
    return _strict.get()


def describe_unknown(unknown: Iterable, known: Iterable[str]) -> str:
    """
    Describe unknown names, each with the most similar known name.

    Parameters
    ----------
    unknown : Iterable
        The unknown names.
    known : Iterable of str
        The known names.

    Returns
    -------
    str
        The unknown names, each with the most similar known name, if any
        (e.g. "'rnmx' (did you mean 'rnmax'?), 'zzz'").
    """
    known = list(known)
    descs = []
    for name in sorted(map(str, unknown)):
        similar = difflib.get_close_matches(name, known, n=1)
        descs.append(
            f"'{name}' (did you mean '{similar[0]}'?)" if similar
            else f"'{name}'")
    return ', '.join(descs)


def report_unknown(
        message: str,
        unknown: Iterable,
        known: Iterable[str]
) -> None:
    """
    Report unknown names: an error, or in lenient mode a warning that they
    are ignored (see strict_mode).

    Parameters
    ----------
    message : str
        What the names are (e.g. 'unknown options').
    unknown : Iterable
        The unknown names.
    known : Iterable of str
        The known names, to suggest similar ones.

    Raises
    ------
    UnknownNamesError
        Unless in lenient mode.
    """
    error = UnknownNamesError(
        f"{message}: {describe_unknown(unknown, known)}")
    if is_strict():
        raise error
    _log_problem(error, "they are ignored (lenient mode)")


def warn(message: str) -> None:
    """
    Log a warning about the part of the configuration being loaded.

    The warning starts with the path to that part, as errors do (see
    config_path). Its record has it as a ConfigError too (its attribute
    config_problem), for programs that collect the warnings (e.g. to
    show them in place).

    Parameters
    ----------
    message : str
        The warning.
    """
    _log_problem(ConfigError(message))


def _log_problem(error: 'ConfigError', note: str | None = None) -> None:
    """
    Log a problem of the part of the configuration being loaded as a
    warning (see warn), with a note after it in the log only.
    """
    # (an error raised here would get the path of each config_path block
    # around it, from the innermost out)
    for segments, context in reversed(_location.get()):
        _add_location(error, segments, context)
    text = str(error) if note is None else f"{error}; {note}"
    _log.warning(text, extra=dict(config_problem=error))
    if (problems := _problems.get()) is not None:
        severity = 'error' if isinstance(error, UnknownNamesError) \
            else 'warning'
        problems.append(_problem(severity, error))


class Problem(NamedTuple):
    """
    A problem of a configuration: an error or a warning, its path in the
    configuration (option names and list indices; e.g. ['galaxies', 0]),
    and its message (with the type of the part, if known).
    """
    severity: Literal['error', 'warning']
    path: list[str | int]
    message: str


@contextlib.contextmanager
def collect_problems() -> Generator[list[Problem]]:
    """
    Collect the problems of the configuration loaded in the block: the
    warnings logged about it (see warn), its unknown names, as errors (see
    report_unknown, in lenient mode), and the ConfigError that ends the
    block, if any, which it stops; each with its path, in order.

    Yields
    ------
    list of Problem
        The problems, as they come.
    """
    problems: list[Problem] = []
    token = _problems.set(problems)
    try:
        yield problems
    except ConfigError as e:
        problems.append(_problem('error', e))
    finally:
        _problems.reset(token)


# The problems collected in this context (thread or task), if any (see
# collect_problems): per context, as strict mode, so that configurations
# loaded at once in two threads do not collect those of each other
_problems: contextvars.ContextVar[list[Problem] | None] = \
    contextvars.ContextVar('gbkfit_problems', default=None)


def _problem(
        severity: Literal['error', 'warning'], error: 'ConfigError'
) -> Problem:
    """Return the problem of an error (or of a warning)."""
    message = error.message if error.context is None \
        else f"[{error.context}] {error.message}"
    return Problem(severity, list(error.path), message)


class ConfigError(Exception):
    """
    An error in a configuration: of what is given, not of a run (not a
    RuntimeError, which code that goes on past failed runs catches, e.g.
    the refits of a bootstrap). The rule: invalid input of a user
    (options, files, arguments of the library's tools) raises it; other
    exceptions are bugs or failed runs.

    Parameters
    ----------
    message : str
        The description of the error.
    path : Sequence of str or int, optional
        The path to the part of the configuration with the error: option
        names and list indices (e.g. models[0].components[1]).
    context : str, optional
        The type of that part (e.g. 'disk'), if any.
    """

    def __init__(
            self,
            message: str,
            path: Sequence[str | int] = (),
            context: str | None = None
    ):
        super().__init__(message)
        self.message = message
        self.path = list(path)
        self.context = context

    def __str__(self):
        path = ''.join(
            f'[{segment}]' if isinstance(segment, int) else f'.{segment}'
            for segment in self.path).lstrip('.')
        if self.context:
            path = f'{path} [{self.context}]' if path else self.context
        return f'{path}: {self.message}' if path else self.message


class UnknownNamesError(ConfigError):
    """
    An error of unknown names in a configuration: options, sections or
    parameters (see report_unknown).
    """


# The parts of the configuration being loaded (see config_path): the
# segments and the context of each
_location = contextvars.ContextVar('location', default=())


def _add_location(
        error: ConfigError,
        segments: Sequence[str | int],
        context: str | None
) -> None:
    """Add the segments and the context of a part to an error."""
    if error.context is None and not error.path:
        error.context = context
    error.path[:0] = segments


@contextlib.contextmanager
def config_path(
        *segments: str | int,
        context: str | None = None
) -> Generator[None]:
    """
    Add to the path of the configuration errors raised in the block.

    Other errors become configuration errors. The warnings of warn get
    the path too.

    Parameters
    ----------
    *segments : str or int
        The option names and list indices to add.
    context : str, optional
        The type of the part of the configuration loaded in the block
        (e.g. 'disk'), for the errors in that part itself (not in its
        options).
    """
    token = _location.set(_location.get() + ((segments, context),))
    try:
        yield
    except ConfigError as e:
        _add_location(e, segments, context)
        raise
    except Exception as e:
        raise ConfigError(str(e), segments, context) from e
    finally:
        _location.reset(token)


def quoted_names(names: Iterable) -> str:
    """Join the names in quotes (e.g. "'a', 'b'")."""
    return ', '.join(f"'{name}'" for name in sorted(names))
