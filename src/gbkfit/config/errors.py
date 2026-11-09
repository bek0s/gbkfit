"""
The errors and warnings of configurations, and the path to the part of
the configuration that each error happens in (see config_path), and
strict and lenient modes for unknown options.
"""

import contextlib
import contextvars
import difflib
import logging
from collections.abc import Generator, Iterable, Sequence
from typing import Literal, NamedTuple


__all__ = [
    'ConfigError',
    'ConfigProblem',
    'UnknownNamesError',
    'collect_problems',
    'config_path',
    'describe_unknown',
    'error_location',
    'error_message',
    'format_location',
    'is_strict',
    'quoted_names',
    'report_unknown',
    'strict_mode',
    'warn'
]


_log = logging.getLogger(__name__)


class ConfigError(Exception):
    """
    An error in a configuration, as its parsing finds it: an unknown or
    missing option, an option of the wrong form, type or range, an
    unknown type, or an invalid name. Its message is all that a user
    needs to fix it (e.g. the CLI shows it without a traceback): that it
    is a mistake of the configuration, and nothing else, is why it is a
    type of its own, and why only the parsing raises it.

    The errors of what a valid configuration is used for are not
    ConfigErrors: of the objects it makes (e.g. a ValueError of a
    constructor, for options that are invalid together), of its files
    and of runs. They pass as they are, with the path in the
    configuration that they happen in (see config_path).

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
        self.config_path = list(path)
        self.config_context = context

    def __str__(self):
        where = format_location(self.config_path, self.config_context)
        return f'{where}: {self.message}' if where else self.message


class UnknownNamesError(ConfigError):
    """
    An error of unknown names in a configuration: options, sections or
    parameters (see report_unknown).
    """


def format_location(
        path: Sequence[str | int], context: str | None = None
) -> str:
    """
    Format a path in a configuration, and the type of its part.

    Parameters
    ----------
    path : Sequence of str or int
        The option names and list indices.
    context : str, optional
        The type of the part (e.g. 'disk').

    Returns
    -------
    str
        The path and the type (e.g. 'models[0].components[1] [disk]'),
        the type alone if there is no path, or an empty string.
    """
    text = ''.join(
        f'[{segment}]' if isinstance(segment, int) else f'.{segment}'
        for segment in path).lstrip('.')
    if context:
        text = f'{text} [{context}]' if text else context
    return text


def error_location(
        error: BaseException
) -> tuple[list[str | int], str | None]:
    """
    Return the path in a configuration of an error raised while it was
    loaded, and the type of its part (see config_path).

    Parameters
    ----------
    error : BaseException
        The error: a ConfigError, or any other.

    Returns
    -------
    tuple of list and (str or None)
        The path (option names and list indices; empty if none) and the
        type of its part, if known.
    """
    return (
        list(getattr(error, 'config_path', [])),
        getattr(error, 'config_context', None))


def error_message(error: BaseException) -> str:
    """
    Return the message of an error raised while a configuration was
    loaded, with its path in the configuration, if any (see config_path).

    Parameters
    ----------
    error : BaseException
        The error: a ConfigError, or any other.

    Returns
    -------
    str
        That of a ConfigError, or that of another error after its type
        (e.g. "galaxies[0]: ValueError: rnodes must start at 0; they
        start at 1").
    """
    if isinstance(error, ConfigError):
        return str(error)
    where = format_location(*error_location(error))
    message = f"{type(error).__name__}: {error}"
    return f"{where}: {message}" if where else message


# The parts of the configuration being loaded (see config_path): the
# segments and the context of each, from the outermost in
_location: contextvars.ContextVar[
    tuple[tuple[Sequence[str | int], str | None], ...]] = \
    contextvars.ContextVar('location', default=())

# The start of the note of the path of an error that is not a ConfigError
# (see config_path)
_NOTE = "in the configuration at "


@contextlib.contextmanager
def config_path(
        *segments: str | int,
        context: str | None = None
) -> Generator[None]:
    """
    Add to the path in the configuration of the errors raised in the
    block, and of the warnings of warn.

    A ConfigError has the path in its message. Any other error passes as
    it is, of its own type, with the path (see error_location) and a note
    of it in its traceback (e.g. "in the configuration at galaxies[0]").

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
    # Errors keep their type: when they became ConfigErrors, bugs, failed
    # runs and failures of devices read as mistakes of the configuration,
    # and code that catches RuntimeError (e.g. a bootstrap) missed them
    except Exception as e:
        _add_location(e, segments, context)
        raise
    finally:
        _location.reset(token)


def _add_location(
        error: BaseException,
        segments: Sequence[str | int],
        context: str | None
) -> None:
    """Add the segments and the context of a part to an error."""
    path = getattr(error, 'config_path', None)
    if path is None:
        path = error.config_path = []
        error.config_context = None
    if error.config_context is None and not path:
        error.config_context = context
    path[:0] = segments
    if not isinstance(error, ConfigError) and path:
        notes = [
            note for note in getattr(error, '__notes__', [])
            if not note.startswith(_NOTE)]
        error.__notes__ = notes + [
            _NOTE + format_location(path, error.config_context)]


def _locate(error: BaseException) -> None:
    """
    Add to an error the path of the parts being loaded, as the blocks of
    config_path around would on its way out of them.
    """
    for segments, context in reversed(_location.get()):
        _add_location(error, segments, context)


# Whether unknown options are errors (strict, the default) or warnings
# (lenient)
_strict: contextvars.ContextVar[bool] = \
    contextvars.ContextVar('strict', default=True)


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


def quoted_names(names: Iterable) -> str:
    """
    Join names in quotes, in order.

    Parameters
    ----------
    names : Iterable
        The names.

    Returns
    -------
    str
        The names, each in quotes (e.g. "'a', 'b'").
    """
    return ', '.join(f"'{name}'" for name in sorted(names))


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
        (e.g. "'rnode' (did you mean 'rnodes'?), 'zzz'").
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
    config_path), and is collected with it (see collect_problems).

    Parameters
    ----------
    message : str
        The warning.
    """
    _log_problem(ConfigError(message))


class ConfigProblem(NamedTuple):
    """
    A problem of a configuration: an error or a warning, its path in the
    configuration (option names and list indices; e.g. ['galaxies', 0]),
    and its message (with the type of the part, if known).
    """
    severity: Literal['error', 'warning']
    path: list[str | int]
    message: str


# The problems collected in this context (thread or task), if any (see
# collect_problems): per context, as strict mode, so that configurations
# loaded at once in two threads do not collect those of each other
_problems: contextvars.ContextVar[list[ConfigProblem] | None] = \
    contextvars.ContextVar('problems', default=None)


@contextlib.contextmanager
def collect_problems() -> Generator[list[ConfigProblem]]:
    """
    Collect the problems of the configuration loaded in the block: the
    warnings logged about it (see warn), its unknown names, as errors (see
    report_unknown, in lenient mode), and the error that ends the block,
    if any, which it stops; each with its full path (that of the blocks
    of config_path around too), in order.

    The error that ends the block is of any type: a ConfigError, or one
    of what the configuration is used for (e.g. a ValueError of a
    constructor, or a bug), whose message starts with its type and whose
    traceback is logged.

    Yields
    ------
    list of ConfigProblem
        The problems, as they come.
    """
    problems: list[ConfigProblem] = []
    token = _problems.set(problems)
    try:
        yield problems
    # Of any type, so that a program that shows the problems (e.g.
    # Studio) shows every error at its path, not only ConfigErrors
    except Exception as e:
        _locate(e)
        if not isinstance(e, ConfigError):
            _log.error(
                "the configuration could not be loaded", exc_info=e)
        problems.append(_problem('error', e))
    finally:
        _problems.reset(token)


def _problem(
        severity: Literal['error', 'warning'], error: Exception
) -> ConfigProblem:
    """Return the problem of an error (or of a warning)."""
    path, context = error_location(error)
    message = error.message if isinstance(error, ConfigError) \
        else f"{type(error).__name__}: {error}"
    if context is not None:
        message = f"[{context}] {message}"
    return ConfigProblem(severity, path, message)


def _log_problem(error: ConfigError, note: str | None = None) -> None:
    """
    Log a problem of the part of the configuration being loaded as a
    warning (see warn), with a note after it in the log only, and collect
    it (see collect_problems).
    """
    _locate(error)
    text = str(error) if note is None else f"{error}; {note}"
    _log.warning(text)
    if (problems := _problems.get()) is not None:
        severity = 'error' if isinstance(error, UnknownNamesError) \
            else 'warning'
        problems.append(_problem(severity, error))
